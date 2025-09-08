from llvmlite import ir, binding
from typing import TYPE_CHECKING, Dict, Callable, Type
from astnodes import *
from lexer import *

def _cast_value_with_pointer_level(self, value: ir.Value, target_type: Any, builder: ir.IRBuilder, 
                                source_pointer_level: int = 0, target_pointer_level: int = 0) -> ir.Value:
    """
    Cast a value to target type considering pointer levels.
    
    Args:
        value: The LLVM value to cast
        target_type: The target LLVM type
        builder: LLVM IR builder
        source_pointer_level: Pointer level of the source value
        target_pointer_level: Pointer level of the target type
    
    Returns:
        Casted LLVM value
    """
    # If pointer levels match and types are compatible, return as-is
    if source_pointer_level == target_pointer_level and value.type == target_type:
        return value
    
    # Handle pointer level mismatches
    if source_pointer_level > target_pointer_level:
        # Need to dereference (load from pointer)
        result = value
        for _ in range(source_pointer_level - target_pointer_level):
            result = builder.load(result, name="auto_deref")
        return result
    elif source_pointer_level < target_pointer_level:
        # This case is more complex and might need address-of operations
        # For now, handle simple cases
        if source_pointer_level == 0 and target_pointer_level == 1:
            # Taking address of a value - this typically requires the value to be in memory
            # This might need special handling depending on the context
            pass
    
    # Fall back to regular casting
    return self._cast_value(value, target_type, builder)




def _cast_value(self, value, target_type, builder):
    """Casts a value to the target LLVM type, inserting necessary instructions."""
    from llvmlite import ir
    src_type = value.type
    
    # DEBUG: Print detailed type information
    if self.compiler.debug:
        print(f"=== CASTING DEBUG ===")
        print(f"Source value: {value}")
        print(f"Source type: {src_type} (type: {type(src_type)})")
        print(f"Target type: {target_type} (type: {type(target_type)})")
        print(f"Value representation: {repr(value)}")
    


    
    # Integer to Integer
    if isinstance(target_type, ir.IntType) and isinstance(src_type, ir.IntType):
        if target_type.width > src_type.width:
            return builder.sext(value, target_type, name="sext") if Datatypes.is_signed_type(str(target_type)) else builder.zext(value, target_type, name="zext")
        else:
            return builder.trunc(value, target_type, name="trunc")
    
    # Float to Float
    if isinstance(target_type, (ir.FloatType, ir.DoubleType)) and isinstance(src_type, (ir.FloatType, ir.DoubleType)):
        if isinstance(target_type, ir.DoubleType) and isinstance(src_type, ir.FloatType):
            return builder.fpext(value, target_type, name="fpext")
        elif isinstance(target_type, ir.FloatType) and isinstance(src_type, ir.DoubleType):
            return builder.fptrunc(value, target_type, name="fptrunc")
    
    # Int to Float
    if isinstance(target_type, (ir.FloatType, ir.DoubleType)) and isinstance(src_type, ir.IntType):
        return builder.sitofp(value, target_type, name="sitofp") if Datatypes.is_signed_type(str(src_type)) else builder.uitofp(value, target_type, name="uitofp")
    
    # Float to Int
    if isinstance(target_type, ir.IntType) and isinstance(src_type, (ir.FloatType, ir.DoubleType)):
        return builder.fptosi(value, target_type, name="fptosi") if Datatypes.is_signed_type(target_type) else builder.fptoui(value, target_type, name="fptoui")
    
    # Pointer to Pointer
    if isinstance(target_type, ir.PointerType) and isinstance(src_type, ir.PointerType):
        return builder.bitcast(value, target_type, name="ptr_cast")
    
    # Integer to Pointer
    if isinstance(target_type, ir.PointerType) and isinstance(src_type, ir.IntType):
        return builder.inttoptr(value, target_type, name="int_to_ptr")
    
    # Pointer to Integer
    if isinstance(target_type, ir.IntType) and isinstance(src_type, ir.PointerType):
        return builder.ptrtoint(value, target_type, name="ptr_to_int")
    
    # Boolean to Integer
    if isinstance(target_type, ir.IntType) and isinstance(src_type, ir.IntType) and src_type.width == 1:
        return builder.zext(value, target_type, name="bool_to_int")
    
    # Integer to Boolean
    if isinstance(target_type, ir.IntType) and target_type.width == 1 and isinstance(src_type, ir.IntType):
        # Comparing with zero to convert to boolean (0 = false, non-zero = true)
        zero = ir.Constant(src_type, 0)
        return builder.icmp_unsigned('!=', value, zero, name="int_to_bool")
    
    # NEW: Handle struct to field access (this is probably what you need)
    # If we have a struct value but need a pointer, this might be a field access issue
    if isinstance(src_type, ir.LiteralStructType) and isinstance(target_type, ir.PointerType):
        # This suggests the struct value wasn't properly converted to a field access
        # This is likely a parsing/expression handling issue, not a casting issue
        raise TypeError(f"Cannot cast struct value to pointer - this suggests a field access was not properly handled. "
                       f"Check that 'self.field' expressions are being processed correctly. "
                       f"Got {src_type} but expected {target_type}")
    
    # Handle struct pointer to field type access
    if (isinstance(src_type, ir.PointerType) and 
        hasattr(src_type.pointee, 'name') and 
        src_type.pointee.name and 
        src_type.pointee.name.startswith('%struct.')):
        
        # This case handles when we have a struct pointer but need to access a field
        # However, this should be handled by the expression processor, not the caster
        raise TypeError(f"Struct field access not properly handled in expression processing. "
                       f"Expected field access to be resolved before casting. "
                       f"Got {src_type} but expected {target_type}")
    
    raise TypeError(f"Incompatible types for assignment: {src_type} cannot be assigned to {target_type}")