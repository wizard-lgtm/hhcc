from llvmlite import ir, binding
from typing import TYPE_CHECKING, Dict, Callable, Type
from astnodes import *
from lexer import *


def _cast_value_with_pointer_level(self, value: ir.Value, target_type: ir.Type, builder: ir.IRBuilder,
                                   source_pointer_level: int = 0, target_pointer_level: int = 0) -> ir.Value:
    """
    Cast a value to target type considering pointer levels.
    Handles simple auto-dereference when source has more pointer levels than target.
    For taking address (source_pointer_level < target_pointer_level) this function
    assumes 'value' is an alloca or already an addressable pointer when necessary.
    """
    # If pointer levels match and types are identical, nothing to do
    if source_pointer_level == target_pointer_level and value.type == target_type:
        return value

    # If we need to lower pointer level: load repeatedly
    if source_pointer_level > target_pointer_level:
        result = value
        for _ in range(source_pointer_level - target_pointer_level):
            # If result is a pointer type, load it
            if isinstance(result.type, ir.PointerType):
                result = builder.load(result, name="auto_deref")
            else:
                raise TypeError("Attempted to dereference a non-pointer value during casting.")
        # After derefs, try to cast the plain value to target_type using main caster
        return self._cast_value(result, target_type, builder)

    # If we need to increase pointer level (i.e. take address)
    if source_pointer_level < target_pointer_level:
        # Common simple case: source is a non-pointer SSA value stored in an alloca (addressable).
        # If value is an addressable pointer already, and target expects a pointer, bitcast/inttoptr may be needed.
        # If source_pointer_level == 0 and target_pointer_level == 1, attempt to take address:
        if source_pointer_level == 0 and target_pointer_level == 1:
            # We need the value to be in memory. The caller should arrange that (e.g., via an alloca).
            # If it's already a pointer, return or bitcast.
            if isinstance(value.type, ir.PointerType):
                return builder.bitcast(value, target_type, name="addr_cast")
            # Otherwise, allocate temporary stack slot, store value, and return pointer to it.
            tmp_ptr = builder.alloca(value.type, name="tmp_addr")
            builder.store(value, tmp_ptr)
            if tmp_ptr.type != target_type:
                return builder.bitcast(tmp_ptr, target_type, name="addr_bitcast")
            return tmp_ptr

        # For more complex increases in pointer level, we don't handle automatically.
        raise TypeError("Automatic address-of for multi-level pointer casts is unsupported.")

    # Fallback: same pointer levels but types mismatch -> delegate to main caster
    return self._cast_value(value, target_type, builder)


def _cast_value(self, value: ir.Value, target_type: ir.Type, builder: ir.IRBuilder) -> ir.Value:
    """Casts a value to the target LLVM type, inserting necessary instructions."""
    from llvmlite import ir

    src_type = value.type

    if self.compiler.debug:
        print("=== CASTING DEBUG ===")
        print(f"Value: {value}")
        print(f"Source type: {src_type} ({type(src_type)})")
        print(f"Target type: {target_type} ({type(target_type)})")

    # If already the exact same type, return as-is
    if src_type == target_type:
        return value

    # Integer to Integer
    if isinstance(target_type, ir.IntType) and isinstance(src_type, ir.IntType):
        if target_type.width > src_type.width:
            # extend (signed vs unsigned)
            if Datatypes.is_signed_type(str(src_type)):
                return builder.sext(value, target_type, name="sext")
            else:
                return builder.zext(value, target_type, name="zext")
        else:
            return builder.trunc(value, target_type, name="trunc")

    # Float to Float (handle identical-case above; now handle width changes)
    if isinstance(target_type, (ir.FloatType, ir.DoubleType)) and isinstance(src_type, (ir.FloatType, ir.DoubleType)):
        # f32 -> f64
        if isinstance(target_type, ir.DoubleType) and isinstance(src_type, ir.FloatType):
            return builder.fpext(value, target_type, name="fpext")
        # f64 -> f32
        if isinstance(target_type, ir.FloatType) and isinstance(src_type, ir.DoubleType):
            return builder.fptrunc(value, target_type, name="fptrunc")
        # other float types or unsupported combos
        raise TypeError(f"Unsupported float-to-float cast: {src_type} -> {target_type}")

    # Int -> Float
    if isinstance(target_type, (ir.FloatType, ir.DoubleType)) and isinstance(src_type, ir.IntType):
        if Datatypes.is_signed_type(str(src_type)):
            return builder.sitofp(value, target_type, name="sitofp")
        else:
            return builder.uitofp(value, target_type, name="uitofp")

    # Float -> Int
    if isinstance(target_type, ir.IntType) and isinstance(src_type, (ir.FloatType, ir.DoubleType)):
        # Datatypes.is_signed_type expects a string form (consistent usage)
        if Datatypes.is_signed_type(str(target_type)):
            return builder.fptosi(value, target_type, name="fptosi")
        else:
            return builder.fptoui(value, target_type, name="fptoui")

    # Pointer -> Pointer
    if isinstance(target_type, ir.PointerType) and isinstance(src_type, ir.PointerType):
        # if pointee types are identical-ish, bitcast is ok; else still bitcast to avoid crash
        return builder.bitcast(value, target_type, name="ptr_cast")

    # Integer -> Pointer
    if isinstance(target_type, ir.PointerType) and isinstance(src_type, ir.IntType):
        return builder.inttoptr(value, target_type, name="int_to_ptr")

    # Pointer -> Integer
    if isinstance(target_type, ir.IntType) and isinstance(src_type, ir.PointerType):
        return builder.ptrtoint(value, target_type, name="ptr_to_int")

    # Boolean conversion: i1 <-> other ints
    if isinstance(target_type, ir.IntType) and target_type.width == 1 and isinstance(src_type, ir.IntType):
        zero = ir.Constant(src_type, 0)
        return builder.icmp_unsigned('!=', value, zero, name="int_to_bool")

    if isinstance(target_type, ir.IntType) and isinstance(src_type, ir.IntType) and src_type.width == 1:
        # bool -> wider int
        return builder.zext(value, target_type, name="bool_to_int")

    # Inline structs / struct-pointer handling: provide clearer error and guidance
    if isinstance(src_type, ir.LiteralStructType) and isinstance(target_type, ir.PointerType):
        raise TypeError(f"Cannot cast an aggregate struct value to a pointer type. "
                        f"This usually indicates field access wasn't lowered. src={src_type}, tgt={target_type}")

    if (isinstance(src_type, ir.PointerType) and hasattr(src_type.pointee, 'name')
            and src_type.pointee.name and str(src_type.pointee.name).startswith('%struct.')):
        raise TypeError(f"Struct field access not resolved before casting. src={src_type}, tgt={target_type}")

    # If none matched, raise with rich info
    raise TypeError(f"Incompatible types for assignment: {src_type} cannot be assigned to {target_type}")
