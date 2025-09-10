from llvmlite import ir 
from astnodes import *
from lexer import *
from .structures import EnumTypeInfo

if TYPE_CHECKING:
    from .base import Codegen


def handle_binary_expression(self: "Codegen", node: ASTNode.ExpressionNode, builder: ir.IRBuilder, var_type, **kwargs):
    
    is_debug = self.compiler.debug
    # Parse the operator
    operator = node.op

    # Determine type characteristics
    is_signed = False
    is_float = False
    is_integer = False
    
    # Try to get type information in different ways
    if hasattr(node, "var_type") and node.var_type:
        is_signed = Datatypes.is_signed_type(node.var_type)
        is_float = Datatypes.is_float_type(node.var_type)
        is_integer = Datatypes.is_integer_type(node.var_type)
        if is_debug:
            print(f"DEBUG - From node var_type: is_signed={is_signed}, is_float={is_float}, is_integer={is_integer}, type={node.var_type}")
    elif hasattr(var_type, "datatype_name") and var_type.datatype_name:
        is_signed = Datatypes.is_signed_type(var_type.datatype_name)
        is_float = Datatypes.is_float_type(var_type.datatype_name)
        is_integer = Datatypes.is_integer_type(var_type.datatype_name)
        if is_debug:
            print(f"DEBUG - From var_type datatype_name: is_signed={is_signed}, is_float={is_float}, is_integer={is_integer}, type={var_type.datatype_name}")
    else:
        # Default values based on LLVM type if we can't determine from name
        if isinstance(var_type, ir.IntType):
            # Check if this type is in the signed types list
            signed_types = [
                self.type_map[t] for t in [Datatypes.I8, Datatypes.I16, Datatypes.I32, Datatypes.I64]
            ]
            is_signed = self.type_signedness.get(var_type, False)
            is_integer = True
            if is_debug:
                print(f"DEBUG - From LLVM IntType: is_signed={is_signed}, width={var_type.width}, type={var_type}")
        elif isinstance(var_type, (ir.FloatType, ir.DoubleType)):
            is_float = True
            is_integer = False
            if is_debug:
                print(f"DEBUG - From LLVM FloatType: is_float={is_float}, type={var_type}")
        else:
            # If we can't determine, default to unsigned integer
            is_signed = False
            is_integer = True
            is_float = False
            if is_debug:
                print(f"DEBUG - Using defaults: is_signed={is_signed}, is_float={is_float}, is_integer={is_integer}, type={var_type}")

    # evaluate left and right expressions
    left = self.handle_expression(node.left, builder, var_type)
    right = self.handle_expression(node.right, builder, var_type)
    
    # Debug the operation
    if is_debug:
        print(f"DEBUG - Operation: {operator}, Left type: {left.type}, Right type: {right.type}")
    
    # Make sure both operands have the same type
    if left.type != right.type:
        if is_debug:
            print(f"DEBUG - Type mismatch: converting right operand from {right.type} to {left.type}")
        
        # For boolean to integer conversions (i1 to i8, etc.)
        if right.type.width < left.type.width:
            if right.type.width == 1:  # Converting from boolean (i1)
                right = builder.zext(right, left.type, name="bool_to_int")
            else:
                # Handle other integer size conversions
                if is_signed:
                    right = builder.sext(right, left.type, name="sext")
                else:
                    right = builder.zext(right, left.type, name="zext")
        elif left.type.width < right.type.width:
            if left.type.width == 1:  # Converting from boolean (i1)
                left = builder.zext(left, right.type, name="bool_to_int")
            else:
                # Handle other integer size conversions
                if is_signed:
                    left = builder.sext(left, right.type, name="sext")
                else:
                    left = builder.zext(left, right.type, name="zext") 
            
        # Check if we need to handle float conversions
        if isinstance(left.type, ir.IntType) and isinstance(right.type, (ir.FloatType, ir.DoubleType)):
            if is_signed:
                left = builder.sitofp(left, right.type, name="int_to_float")
            else:
                left = builder.uitofp(left, right.type, name="uint_to_float")
        elif isinstance(right.type, ir.IntType) and isinstance(left.type, (ir.FloatType, ir.DoubleType)):
            if is_signed:
                right = builder.sitofp(right, left.type, name="int_to_float")
            else:
                right = builder.uitofp(right, left.type, name="uint_to_float")
    
    # After conversion, re-check what types we're working with
    is_float = isinstance(left.type, (ir.FloatType, ir.DoubleType))
    is_integer = isinstance(left.type, ir.IntType)
    
    # Handle operations based on operator type
    if operator == operators["ADD"]:
        return builder.add(left, right, name="sum")
    elif operator == operators["SUBTRACT"]:
        return builder.sub(left, right, name="sub")
    elif operator == operators["MULTIPLY"]:
        return builder.mul(left, right, name="mul")
    elif operator == operators["DIVIDE"]:
        # For integer division
        if is_integer:
            if is_signed:
                if is_debug:
                    print(f"DEBUG - Using SIGNED integer division (sdiv)")
                return builder.sdiv(left, right, name="sdiv")
            else:
                if is_debug:
                    print(f"DEBUG - Using UNSIGNED integer division (udiv)")
                return builder.udiv(left, right, name="udiv")
        # For floating point division
        else:
            if is_debug:
                print(f"DEBUG - Using floating point division (fdiv)")
            return builder.fdiv(left, right, name="fdiv")
    elif operator == operators["MODULO"]:
        # For integer modulo
        if is_integer:
            if is_signed:
                if is_debug:
                    print(f"DEBUG - Using SIGNED integer remainder (srem)")
                return builder.srem(left, right, name="srem")
            else:
                if is_debug:
                    print(f"DEBUG - Using UNSIGNED integer remainder (urem)")
                return builder.urem(left, right, name="urem")
        # For floating point modulo
        else:
            if is_debug:
                print(f"DEBUG - Using floating point remainder (frem)")
            return builder.frem(left, right, name="frem")
    elif operator == operators["BITWISE_AND"]:
        return builder.and_(left, right, name="and")
    elif operator == operators["BITWISE_OR"]:
        return builder.or_(left, right, name="or")
    elif operator == operators["BITWISE_XOR"]:
        return builder.xor(left, right, name="xor")
    elif operator == operators["SHIFT_LEFT"]:
        return builder.shl(left, right, name="shl")
    elif operator == operators["SHIFT_RIGHT"]:
        # Arithmetic shift for signed types
        if is_signed:
            if is_debug:
                print(f"DEBUG - Using arithmetic right shift (ashr) for signed type")
            return builder.ashr(left, right, name="ashr")
        # Logical shift for unsigned types
        else:
            if is_debug:
                print(f"DEBUG - Using logical right shift (lshr) for unsigned type")
            return builder.lshr(left, right, name="lshr")
    # Comparison operators
    elif operator == operators["EQUAL"]:
        # Ensure operands have the same type for comparison
        if left.type != right.type:
            if is_debug:
                print(f"DEBUG - Type mismatch in comparison: converting operands to match")
            if left.type.width > right.type.width:
                right = builder.zext(right, left.type, name="zext_for_cmp") if right.type.width == 1 else right
            else:
                left = builder.zext(left, right.type, name="zext_for_cmp") if left.type.width == 1 else left
        
        if is_integer:
            if is_signed:
                if is_debug:
                    print(f"DEBUG - Using SIGNED integer comparison (==)")
                return builder.icmp_signed('==', left, right, name="seq")  # Will return i1
            else:
                if is_debug:
                    print(f"DEBUG - Using UNSIGNED integer comparison (==)")
                return builder.icmp_unsigned('==', left, right, name="ueq")  # Will return i1
        else:
            if is_debug:
                print(f"DEBUG - Using floating point comparison (==)")
            return builder.fcmp_ordered('==', left, right, name="feq")  # Will return i1
    elif operator == operators["NOT_EQUAL"]:
        if is_integer:
            if is_signed:
                if is_debug:
                    print(f"DEBUG - Using SIGNED integer comparison (!=)")
                return builder.icmp_signed('!=', left, right, name="sne") 
            else:
                if is_debug:
                    print(f"DEBUG - Using UNSIGNED integer comparison (!=)")
                return builder.icmp_unsigned('!=', left, right, name="une")
        else:
            if is_debug:
                print(f"DEBUG - Using floating point comparison (!=)")
            return builder.fcmp_ordered('!=', left, right, name="fne")
    elif operator == operators["LESS_THAN"]:
        if is_integer:
            if is_signed:
                if is_debug:
                    print(f"DEBUG - Using SIGNED integer comparison (<)")
                return builder.icmp_signed('<', left, right, name="slt")
            else:
                if is_debug:
                    print(f"DEBUG - Using UNSIGNED integer comparison (<)")
                return builder.icmp_unsigned('<', left, right, name="ult")
        else:
            if is_debug:
                print(f"DEBUG - Using floating point comparison (<)")
            return builder.fcmp_ordered('<', left, right, name="flt")
    elif operator == operators["LESS_OR_EQUAL"]:
        if is_integer:
            if is_signed:
                if is_debug:
                    print(f"DEBUG - Using SIGNED integer comparison (<=)")
                return builder.icmp_signed('<=', left, right, name="sle")
            else:
                if is_debug:
                    print(f"DEBUG - Using UNSIGNED integer comparison (<=)")
                return builder.icmp_unsigned('<=', left, right, name="ule")
        else:
            if is_debug:
                print(f"DEBUG - Using floating point comparison (<=)")
            return builder.fcmp_ordered('<=', left, right, name="fle")
    elif operator == operators["GREATER_THAN"]:
        if is_integer:
            if is_signed:
                if is_debug:
                    print(f"DEBUG - Using SIGNED integer comparison (>)")
                return builder.icmp_signed('>', left, right, name="sgt")
            else:
                if is_debug:
                    print(f"DEBUG - Using UNSIGNED integer comparison (>)")
                return builder.icmp_unsigned('>', left, right, name="ugt")
        else:
            if is_debug:
                print(f"DEBUG - Using floating point comparison (>)")
            return builder.fcmp_ordered('>', left, right, name="fgt")
    elif operator == operators["GREATER_OR_EQUAL"]:
        if is_integer:
            if is_signed:
                if is_debug:
                    print(f"DEBUG - Using SIGNED integer comparison (>=)")
                return builder.icmp_signed('>=', left, right, name="sge")
            else:
                if is_debug:
                    print(f"DEBUG - Using UNSIGNED integer comparison (>=)") 
                return builder.icmp_unsigned('>=', left, right, name="uge")
        else:
            if is_debug:
                print(f"DEBUG - Using floating point comparison (>=)")
            return builder.fcmp_ordered('>=', left, right, name="fge")
    elif operator == operators["LOGICAL_AND"]:
        # Perform boolean conversion if needed
        if left.type.width > 1:
            if is_signed:
                if is_debug:
                    print(f"DEBUG - Converting SIGNED integer to boolean for LOGICAL_AND")
                left_bool = builder.icmp_signed('!=', left, ir.Constant(left.type, 0), name="tobool_left")
            else:
                if is_debug:
                    print(f"DEBUG - Converting UNSIGNED integer to boolean for LOGICAL_AND")
                left_bool = builder.icmp_unsigned('!=', left, ir.Constant(left.type, 0), name="tobool_left")
        else:
            left_bool = left
            
        if right.type.width > 1:
            if is_signed:
                if is_debug:
                    print(f"DEBUG - Converting SIGNED integer to boolean for LOGICAL_AND")
                right_bool = builder.icmp_signed('!=', right, ir.Constant(right.type, 0), name="tobool_right")
            else:
                if is_debug:
                    print(f"DEBUG - Converting UNSIGNED integer to boolean for LOGICAL_AND")
                right_bool = builder.icmp_unsigned('!=', right, ir.Constant(right.type, 0), name="tobool_right")
        else:
            right_bool = right
            
        return builder.and_(left_bool, right_bool, name="land")
    elif operator == operators["LOGICAL_OR"]:
        # Perform boolean conversion if needed
        if left.type.width > 1:
            if is_signed:
                if is_debug:
                    print(f"DEBUG - Converting SIGNED integer to boolean for LOGICAL_OR")
                left_bool = builder.icmp_signed('!=', left, ir.Constant(left.type, 0), name="tobool_left")
            else:
                if is_debug:
                    print(f"DEBUG - Converting UNSIGNED integer to boolean for LOGICAL_OR")
                left_bool = builder.icmp_unsigned('!=', left, ir.Constant(left.type, 0), name="tobool_left")
        else:
            left_bool = left
            
        if right.type.width > 1:
            if is_signed:
                if is_debug:
                    print(f"DEBUG - Converting SIGNED integer to boolean for LOGICAL_OR")
                right_bool = builder.icmp_signed('!=', right, ir.Constant(right.type, 0), name="tobool_right")
            else:
                if is_debug:
                    print(f"DEBUG - Converting UNSIGNED integer to boolean for LOGICAL_OR")
                right_bool = builder.icmp_unsigned('!=', right, ir.Constant(right.type, 0), name="tobool_right")
        else:
            right_bool = right
            
        return builder.or_(left_bool, right_bool, name="lor")
    else:
        raise ValueError(f"Unsupported binary operator: {operator}")
    
def handle_primary_expression(self: "Codegen", node: ASTNode.ExpressionNode, builder: ir.IRBuilder, var_type, **kwargs):
    if node.node_type == NodeType.REFERENCE and node.value == '&': 
        return self.handle_pointer(node, builder)
    elif node.node_type == NodeType.BINARY_OP:
        return self.handle_binary_expression(node, builder, var_type)
    elif node.node_type == NodeType.POSTFIX_OP:
        return self.handle_postfix_op(node, builder, var_type)
    elif node.node_type == NodeType.LITERAL:
        # First check if this is actually a variable reference
        if node.value in self.symbol_table:
            # It's a variable name, load its value
            var_ptr = self.symbol_table[node.value]
            return builder.load(var_ptr, name=f"load_{node.value}")
        else:
            # It's an actual literal value, create a constant
            try:
                return ir.Constant(var_type, int(node.value))
            except ValueError:
                raise ValueError(f"Invalid literal or undefined variable: '{node.value}'")

def handle_expression(self: "Codegen", node, builder: ir.IRBuilder, var_type, **kwargs):
    pointer_level = kwargs.get('pointer_level', 0)

    if node is None:
        raise ValueError("Node is None, cannot handle expression.")

    # Normalize raw literals into AST nodes
    if not isinstance(node, ASTNode.ExpressionNode):
        # Wrap strings, ints, floats, etc.
        node = ASTNode.ExpressionNode(node_type=NodeType.LITERAL, value=node.value if isinstance(node, Token) else node)

    match node.node_type:
        case NodeType.BINARY_OP:
            return self.handle_binary_expression(node, builder, var_type)

        case NodeType.REFERENCE:
            return self.handle_pointer(node, builder)

        case NodeType.UNARY_OP:
            return self.handle_unary_op(node, builder, var_type)

        case NodeType.FUNCTION_CALL:
            return self.handle_function_call(node, builder, **kwargs)

        case NodeType.STRUCT_ACCESS:
            return self.handle_struct_access(node, builder)

        case NodeType.LITERAL:
            return self._expression_handle_literal(node, builder, var_type, pointer_level=pointer_level)

        case NodeType.ENUM_ACCESS:
            return self.handle_enum_access(node, builder, var_type)

        case NodeType.CAST:
            return self.handle_cast(node, builder)
        case NodeType.POSTFIX_OP:
            return self.handle_postfix_op(node, builder, var_type)

        case NodeType.ARRAY_ACCESS:
            return handle_array_access(self, node, builder, var_type)

        case _:
            raise ValueError(f"Unsupported expression node type: {node.node_type}")

def handle_array_access(self: "Codegen", node: ASTNode.ExpressionNode, builder: ir.IRBuilder, var_type, **kwargs):
    """
    Handle array access operations like arr[index]
    
    Args:
        node: The array access node (should have left=array_name, right=index)
        builder: The LLVM IR builder
        var_type: The expected type of the expression
        
    Returns:
        The LLVM value representing the array element
    """
    debug = getattr(self.compiler, 'debug', False)
    
    if debug:
        print(f"Handling array access")
    
    # Extract array and index from the node
    if not hasattr(node, 'left') or not hasattr(node, 'right'):
        raise ValueError(f"Invalid array access node structure: missing left (array) or right (index)")
    
    array_node = node.left
    index_node = node.right
    
    # Get the array variable
    if array_node.node_type == NodeType.LITERAL and array_node.value in self.symbol_table:
        array_symbol = self.symbol_table.lookup(array_node.value)
        if not array_symbol:
            raise ValueError(f"Array variable '{array_node.value}' not found")
        
        array_ptr = array_symbol.llvm_value
        
        if debug:
            print(f"Array symbol: {array_symbol}")
            print(f"Array type: {array_symbol.data_type}")
    else:
        # Handle more complex array expressions (like struct field arrays, etc.)
        array_ptr = self.handle_expression(array_node, builder, None, **kwargs)
    
    # Evaluate the index expression
    index_value = self.handle_expression(index_node, builder, ir.IntType(32), **kwargs)
    
    if debug:
        print(f"Index value type: {index_value.type}")
    
    # Ensure index is the right type (i32)
    if index_value.type != ir.IntType(32):
        if isinstance(index_value.type, ir.IntType):
            if index_value.type.width < 32:
                index_value = builder.zext(index_value, ir.IntType(32), name="index_zext")
            elif index_value.type.width > 32:
                index_value = builder.trunc(index_value, ir.IntType(32), name="index_trunc")
        else:
            raise ValueError(f"Array index must be an integer type, got {index_value.type}")
    
    # Check what type of pointer we're dealing with
    if debug:
        print(f"Array pointer type: {array_ptr.type}")
    
    # If array_ptr is a pointer to a pointer (like i8**), we need to load it first
    # to get the actual array/string pointer
    if isinstance(array_ptr.type, ir.PointerType) and isinstance(array_ptr.type.pointee, ir.PointerType):
        # This is a pointer to a pointer (like i8** for string parameters)
        # Load the actual string/array pointer first
        actual_array_ptr = builder.load(array_ptr, name="load_array_ptr")
        
        # Now use GEP with just the index (no zero prefix needed for direct pointer arithmetic)
        element_ptr = builder.gep(actual_array_ptr, [index_value], name="array_elem_ptr")
        
    elif isinstance(array_ptr.type, ir.PointerType) and isinstance(array_ptr.type.pointee, ir.ArrayType):
        # This is a pointer to an array type
        # Use GEP with [0, index] - 0 to get the array, index for the element
        zero = ir.Constant(ir.IntType(32), 0)
        element_ptr = builder.gep(array_ptr, [zero, index_value], name="array_elem_ptr")
        
    elif isinstance(array_ptr.type, ir.PointerType):
        # This is a direct pointer (like i8* for a string)
        # Use GEP with just the index
        element_ptr = builder.gep(array_ptr, [index_value], name="array_elem_ptr")
        
    else:
        raise ValueError(f"Unsupported array type for indexing: {array_ptr.type}")
    
    if debug:
        print(f"Element pointer type: {element_ptr.type}")
    
    # Load the element value
    element_value = builder.load(element_ptr, name="array_elem_load")
    
    if debug:
        print(f"Element value type: {element_value.type}")
    
    return element_value


def handle_enum_access(self: "Codegen", node: ASTNode.ExpressionNode, builder: ir.IRBuilder, var_type, **kwargs):
    """
    Handle enum access operations like EnumName::MemberName or EnumName.MemberName
    
    Args:
        node: The enum access node (should have enum_name and member_name attributes)
        builder: The LLVM IR builder
        var_type: The expected type of the expression
        
    Returns:
        The LLVM constant value representing the enum member
    """
    debug = getattr(self.compiler, 'debug', False)
    
    # Extract enum name and member name from the node
    # The exact attribute names depend on how your AST represents enum access
    if hasattr(node, 'enum_name') and hasattr(node, 'member_name'):
        enum_name = node.enum_name
        member_name = node.member_name
    elif hasattr(node, 'left') and hasattr(node, 'right'):
        # Alternative structure where left is enum name, right is member name
        enum_name = node.left.value if hasattr(node.left, 'value') else str(node.left)
        member_name = node.right.value if hasattr(node.right, 'value') else str(node.right)
    else:
        raise ValueError(f"Invalid enum access node structure: {node}")
    
    if debug:
        print(f"Handling enum access: {enum_name}::{member_name}")
    
    # Look up the enum type information
    enum_type_info = Datatypes.get_type(enum_name)
    if not enum_type_info:
        raise ValueError(f"Unknown enum type: {enum_name}")
    
    # Check if it's actually an enum type
    if not isinstance(enum_type_info, self.EnumTypeInfo):
        raise ValueError(f"'{enum_name}' is not an enum type")
    
    # Get the member value
    member_value = enum_type_info.get_member_value(member_name)
    if member_value is None:
        available_members = list(enum_type_info.values.keys())
        raise ValueError(f"Enum member '{member_name}' not found in enum '{enum_name}'. "
                        f"Available members: {available_members}")
    
    if debug:
        print(f"Enum {enum_name}::{member_name} has value {member_value}")
    
    # Determine the target type for the constant
    target_type = var_type if var_type else enum_type_info.get_llvm_type()
    
    # Create and return the constant value
    return ir.Constant(target_type, member_value)




def handle_cast(self: "Codegen", node: ASTNode.ExpressionNode, builder: ir.IRBuilder, **kwargs):
    """
    Handle type casting expressions like (int)value or (float*)ptr
    
    Args:
        node: The cast node (should have left=expression, value=target_type, op="cast")
        builder: The LLVM IR builder
        
    Returns:
        The LLVM value representing the casted expression
    """
    debug = getattr(self.compiler, 'debug', False)
    
    # Extract target type and expression from the cast node
    if not hasattr(node, 'value') or not hasattr(node, 'left'):
        raise ValueError(f"Invalid cast node structure: missing value (target_type) or left (expression)")
    
    target_type_name = node.value  # The target type is stored in 'value'
    expression_node = node.left    # The expression to cast is stored in 'left'
    
    if debug:
        print(f"Handling cast to type: {target_type_name}")
    
    # Handle pointer types (e.g., "U32*", "U32***")
    pointer_level = 0
    base_type_name = target_type_name
    
    # Count and remove pointer indicators
    while base_type_name.endswith('*'):
        pointer_level += 1
        base_type_name = base_type_name[:-1]
    
    # Get the base target LLVM type
    target_type_info = Datatypes.get_type_from_string(base_type_name)
    if not target_type_info:
        raise ValueError(f"Unknown target type for cast: {base_type_name}")
    
    # Get LLVM type and apply pointer levels
    target_llvm_type = Datatypes.get_llvm_type(base_type_name)
    for _ in range(pointer_level):
        target_llvm_type = target_llvm_type.as_pointer()
    
    # Evaluate the expression to be casted
    source_value = self.handle_expression(expression_node, builder, None, **kwargs)
    source_type = source_value.type
    
    if debug:
        print(f"Casting from {source_type} to {target_llvm_type}")
    
    # If types are already the same, no cast needed
    if source_type == target_llvm_type:
        return source_value
    
    # Handle different casting scenarios
    return self._perform_cast(source_value, source_type, target_llvm_type, target_type_info, builder, debug)


def _perform_cast(self: "Codegen", source_value, source_type, target_llvm_type, target_type_info, builder: ir.IRBuilder, debug: bool):
    """
    Perform the actual LLVM cast operation based on source and target types
    
    Args:
        source_value: The LLVM value to cast
        source_type: The source LLVM type
        target_llvm_type: The target LLVM type
        target_type_info: The target type information object
        builder: The LLVM IR builder
        debug: Debug flag
        
    Returns:
        The casted LLVM value
    """
    
    # Pointer casts
    if isinstance(source_type, ir.PointerType) and isinstance(target_llvm_type, ir.PointerType):
        if debug:
            print("Performing pointer-to-pointer cast (bitcast)")
        return builder.bitcast(source_value, target_llvm_type)
    
    # Integer to pointer cast
    elif isinstance(source_type, ir.IntType) and isinstance(target_llvm_type, ir.PointerType):
        if debug:
            print("Performing integer-to-pointer cast (inttoptr)")
        return builder.inttoptr(source_value, target_llvm_type)
    
    # Pointer to integer cast
    elif isinstance(source_type, ir.PointerType) and isinstance(target_llvm_type, ir.IntType):
        if debug:
            print("Performing pointer-to-integer cast (ptrtoint)")
        return builder.ptrtoint(source_value, target_llvm_type)
    
    # Integer to integer cast
    elif isinstance(source_type, ir.IntType) and isinstance(target_llvm_type, ir.IntType):
        source_width = source_type.width
        target_width = target_llvm_type.width
        
        if source_width == target_width:
            # Same width, just bitcast
            if debug:
                print(f"Performing same-width integer cast (bitcast)")
            return builder.bitcast(source_value, target_llvm_type)
        elif source_width < target_width:
            # Sign extend or zero extend based on type information
            is_signed = getattr(target_type_info, 'is_signed', True)  # Default to signed
            if is_signed:
                if debug:
                    print(f"Performing sign extension from {source_width} to {target_width} bits")
                return builder.sext(source_value, target_llvm_type)
            else:
                if debug:
                    print(f"Performing zero extension from {source_width} to {target_width} bits")
                return builder.zext(source_value, target_llvm_type)
        else:
            # Truncate
            if debug:
                print(f"Performing truncation from {source_width} to {target_width} bits")
            return builder.trunc(source_value, target_llvm_type)
    
    # Float to float cast
    elif isinstance(source_type, (ir.FloatType, ir.DoubleType)) and isinstance(target_llvm_type, (ir.FloatType, ir.DoubleType)):
        source_width = self._get_float_width(source_type)
        target_width = self._get_float_width(target_llvm_type)
        
        if source_width < target_width:
            if debug:
                print("Performing float extension (fpext)")
            return builder.fpext(source_value, target_llvm_type)
        elif source_width > target_width:
            if debug:
                print("Performing float truncation (fptrunc)")
            return builder.fptrunc(source_value, target_llvm_type)
        else:
            # Same type, no cast needed (shouldn't reach here due to earlier check)
            return source_value
    
    # Integer to float cast
    elif isinstance(source_type, ir.IntType) and isinstance(target_llvm_type, (ir.FloatType, ir.DoubleType)):
        is_signed = getattr(target_type_info, 'is_signed', True)
        if is_signed:
            if debug:
                print("Performing signed integer-to-float cast (sitofp)")
            return builder.sitofp(source_value, target_llvm_type)
        else:
            if debug:
                print("Performing unsigned integer-to-float cast (uitofp)")
            return builder.uitofp(source_value, target_llvm_type)
    
    # Float to integer cast
    elif isinstance(source_type, (ir.FloatType, ir.DoubleType)) and isinstance(target_llvm_type, ir.IntType):
        is_signed = getattr(target_type_info, 'is_signed', True)
        if is_signed:
            if debug:
                print("Performing float-to-signed-integer cast (fptosi)")
            return builder.fptosi(source_value, target_llvm_type)
        else:
            if debug:
                print("Performing float-to-unsigned-integer cast (fptoui)")
            return builder.fptoui(source_value, target_llvm_type)
    
    # Boolean conversions
    elif isinstance(target_llvm_type, ir.IntType) and target_llvm_type.width == 1:
        # Cast to boolean (i1)
        if isinstance(source_type, ir.IntType):
            if debug:
                print("Performing integer-to-boolean cast")
            zero = ir.Constant(source_type, 0)
            return builder.icmp_ne(source_value, zero)
        elif isinstance(source_type, (ir.FloatType, ir.DoubleType)):
            if debug:
                print("Performing float-to-boolean cast")
            zero = ir.Constant(source_type, 0.0)
            return builder.fcmp_one(source_value, zero)  # one = ordered and not equal
        elif isinstance(source_type, ir.PointerType):
            if debug:
                print("Performing pointer-to-boolean cast")
            null_ptr = ir.Constant(source_type, None)
            return builder.icmp_ne(source_value, null_ptr)
    
    # If we reach here, the cast is not supported
    raise ValueError(f"Unsupported cast from {source_type} to {target_llvm_type}")


def _get_float_width(self: "Codegen", float_type):
    """
    Helper method to get the bit width of floating point types
    
    Args:
        float_type: The LLVM floating point type
        
    Returns:
        The bit width of the type
    """
    if isinstance(float_type, ir.FloatType):
        return 32
    elif isinstance(float_type, ir.DoubleType):
        return 64
    else:
        # For other float types, you might need to add more cases
        # depending on your compiler's type system
        raise ValueError(f"Unknown float type: {float_type}")
    
def handle_unary_op(self: "Codegen", node: ASTNode.ExpressionNode, builder: ir.IRBuilder, var_type, **kwargs):

    """
    Handle unary operations like negation (-), bitwise not (~), logical not (!), etc.
    
    Args:
        node: The unary operation node
        builder: The LLVM IR builder
        var_type: The expected type of the expression
        
    Returns:
        The LLVM value representing the result of the unary operation
    """
    # Process the operand first
    operand = self.handle_expression(node.left, builder, var_type)
    
    match node.op:
        case '-':  # Numeric negation
            if isinstance(var_type, ir.FloatType):
                return builder.fneg(operand, name="neg")
            else:
                # For integers, we can use 0 - value
                zero = ir.Constant(operand.type, 0)
                return builder.sub(zero, operand, name="neg")
        
        case '~':  # Bitwise NOT
            # Only applicable to integer types
            if isinstance(var_type, (ir.IntType)):
                return builder.not_(operand, name="bitnot")
            else:
                raise ValueError(f"Bitwise NOT (~) cannot be applied to type {var_type}")
        
        case '!':  # Logical NOT
            # Convert to boolean (0 or 1) first if not already
            if operand.type != ir.IntType(1):
                # For integers, compare with 0
                if isinstance(operand.type, ir.IntType):
                    zero = ir.Constant(operand.type, 0)
                    bool_val = builder.icmp_ne(operand, zero, name="to_bool")
                # For floats, compare with 0.0
                elif isinstance(operand.type, ir.FloatType):
                    zero = ir.Constant(operand.type, 0.0)
                    bool_val = builder.fcmp_one(operand, zero, name="to_bool")
                else:
                    raise ValueError(f"Cannot convert type {operand.type} to boolean")
            else:
                bool_val = operand
                
            # Invert the boolean value
            return builder.not_(bool_val, name="lognot")
            
        case '*':  # Dereference pointer
            # Check if operand is a pointer type
            if not isinstance(operand.type, ir.PointerType):
                raise ValueError(f"Cannot dereference non-pointer type {operand.type}")
            return builder.load(operand, name="deref")
            
        case '&':  # Reference/Address-of
            # This should be handled elsewhere since we likely need the variable name
            # and not just the expression result
            raise ValueError("Address-of operator (&) should be handled by handle_pointer method")
            
        case _:
            raise ValueError(f"Unsupported unary operator: {node.operator}")

def handle_pointer(self: "Codegen", node: ASTNode.ExpressionNode, builder: ir.IRBuilder, **kwargs):
    var_ptr = self.get_variable_pointer(node.left.value)
    return var_ptr

def _process_escape_sequences(self: "Codegen", string_content: str) -> str:
    """
    Process escape sequences in string literals.
    
    Args:
        string_content: The string content without quotes
        
    Returns:
        String with escape sequences processed
    """
    # Handle common escape sequences
    escape_map = {
        '\\n': '\n',
        '\\t': '\t',
        '\\r': '\r',
        '\\\\': '\\',
        '\\"': '"',
        "\\'": "'",
        '\\0': '\0'
    }
    
    result = string_content
    for escape_seq, replacement in escape_map.items():
        result = result.replace(escape_seq, replacement)
    
    return result


def _create_string_literal(self: "Codegen", quoted_string: str, builder: ir.IRBuilder, target_type, is_pointer: bool = False) -> ir.Value:
    """
    Create a string literal in LLVM IR with proper PIC support.
    
    Args:
        quoted_string: The string including quotes (e.g., '"Hello"')
        builder: The LLVM IR builder
        target_type: The target type (should be pointer type)
        
    Returns:
        LLVM value representing the string literal
    """
    # Remove the quotes
    string_content = quoted_string[1:-1]
    
    # Handle escape sequences
    string_content = self._process_escape_sequences(string_content)
    
    # Special handling for single byte types (U8/I8)
    if isinstance(target_type, ir.IntType) and target_type.width == 8 and is_pointer is False:
        if len(string_content) == 0:
            # Empty string -> null character
            return ir.Constant(target_type, 0)
        elif len(string_content) == 1:
            # Single character -> return its ASCII value
            return ir.Constant(target_type, ord(string_content[0]))
        else:
            # Multi-character string -> take first character with warning
            print(f"Warning: String literal '{quoted_string}' truncated to first character for U8 variable")
            return ir.Constant(target_type, ord(string_content[0]))
    
    # For pointer types, create a proper string with PIC support
    # Add null terminator
    string_with_null = string_content + '\0'
    
    # Create the array type for the string
    string_array_type = ir.ArrayType(ir.IntType(8), len(string_with_null))
    
    # Create a global variable for the string with proper linkage for PIC
    string_global = ir.GlobalVariable(
        self.module, 
        string_array_type, 
        name=f"str_{len(getattr(self, 'string_literals', []))}"
    )
    
    # Set proper attributes for PIC compilation
    string_global.linkage = 'private'  # Changed from 'internal' to 'private'
    string_global.global_constant = True
    string_global.unnamed_addr = True  # Allow merging of identical constants
    
    # Initialize with the string content
    string_global.initializer = ir.Constant(
        string_array_type, 
        bytearray(string_with_null.encode('utf-8'))
    )
    
    # Keep track of string literals
    if not hasattr(self, 'string_literals'):
        self.string_literals = []
    self.string_literals.append(string_global)
    
    # Create a GEP instruction to get pointer to the first character
    zero = ir.Constant(ir.IntType(32), 0)
    string_ptr = builder.gep(string_global, [zero, zero], name="str_ptr")
    
    return string_ptr

# Alternative approach - if you want to be more strict about type checking
def _create_string_literal_strict(self: "Codegen", quoted_string: str, builder: ir.IRBuilder, target_type):
    """
    Strict version that provides better error messages for type mismatches.
    """
    # Remove the quotes
    string_content = quoted_string[1:-1]
    
    # Handle escape sequences
    string_content = self._process_escape_sequences(string_content)
    
    # Strict type checking for single byte types
    if isinstance(target_type, ir.IntType) and target_type.width == 8:
        if len(string_content) == 0:
            return ir.Constant(target_type, 0)  # Empty string -> null char
        elif len(string_content) == 1:
            return ir.Constant(target_type, ord(string_content[0]))  # Single char
        else:
            # Provide helpful error message
            raise TypeError(f"Cannot assign string literal '{quoted_string}' to U8 variable. "
                        f"String has {len(string_content)} characters but U8 can only hold 1. "
                        f"Consider:\n"
                        f"  - Using a single character: '\"T\"' (first char of '{string_content}')\n"
                        f"  - Declaring variable as string pointer: 'var a: *U8 = {quoted_string}'\n"
                        f"  - Using an array: 'var a: [4]U8 = ...' for fixed-size strings")

        
def _expression_handle_literal(self: "Codegen", node: ASTNode.ExpressionNode, builder: ir.IRBuilder, var_type, **kwargs):
    pointer_level = kwargs.get('pointer_level', 0)
    
    if self.compiler.debug:
        print(f"Handling literal: '{node.value}', target type: {var_type}, pointer_level: {pointer_level}")
    
    """
    Handle literal expressions like numbers, booleans, strings, etc.
    """
    # Print debug info
    debug = getattr(self, 'debug', False)
    if debug:
        print(f"Handling literal: '{node.value}', target type: {var_type}, pointer_level: {pointer_level}")
    
    # Handle special node types that might be misidentified as literals
    if not hasattr(node, 'value'):
        if debug:
            print(f"Node doesn't have a 'value' attribute: {node}")
        if hasattr(node, 'node_type') and node.node_type == NodeType.STRUCT_ACCESS:
            return self.handle_struct_access(node, builder)
        else:
            raise ValueError(f"Invalid literal node: {node}")
    
    # Check for enum access (EnumName::MemberName format in string)
    if isinstance(node.value, str) and '::' in node.value:
        parts = node.value.split('::', 1)
        if len(parts) == 2:
            enum_name, member_name = parts
            enum_type_info = Datatypes.get_type(enum_name)
            if enum_type_info and isinstance(enum_type_info, self.EnumTypeInfo):
                member_value = enum_type_info.get_member_value(member_name)
                if member_value is not None:
                    target_type = var_type if var_type else enum_type_info.get_llvm_type()
                    return ir.Constant(target_type, member_value)
                else:
                    available_members = list(enum_type_info.values.keys())
                    raise ValueError(f"Enum member '{member_name}' not found in enum '{enum_name}'. "
                                f"Available members: {available_members}")
    
    # Handle variable reference if it's in the symbol table
    if isinstance(node.value, str) and node.value in self.symbol_table:
        var_info = self.symbol_table.lookup(node.value)
        if var_info:
            # For pointer assignments, we might need to handle dereferencing
            if pointer_level > 0 and var_info.pointer_level > pointer_level:
                # Need to dereference the variable
                loaded_value = builder.load(var_info.llvm_value, name=f"load_{node.value}")
                # Apply additional dereferencing if needed
                for _ in range(var_info.pointer_level - pointer_level):
                    loaded_value = builder.load(loaded_value, name=f"deref_{node.value}")
                return loaded_value
            elif pointer_level > 0 and var_info.pointer_level == 0:
                # Taking address of a regular variable
                return var_info.llvm_value  # Return the alloca (address)
            else:
                # Regular load
                return builder.load(var_info.llvm_value, name=f"load_{node.value}")
    
    # Check if this is a string literal (quoted string)
    is_string_literal = (isinstance(node.value, str) and 
                        len(node.value) >= 2 and 
                        node.value.startswith('"') and 
                        node.value.endswith('"'))

    # Check if this is a char literal (single char in single quotes)
    # Fixed: More robust character literal detection
    is_char_literal = (isinstance(node.value, str) and
                      len(node.value) >= 2 and  # Changed from >= 3 to >= 2
                      node.value.startswith("'") and
                      node.value.endswith("'"))
 
    
    # Infer type if not specified
    if var_type is None:
        if isinstance(node.value, str):
            if is_string_literal:
                # String literal - return pointer to i8 array
                var_type = ir.PointerType(ir.IntType(8))
            elif is_char_literal:
                # Character literal - return i8
                var_type = ir.IntType(8)
            elif node.value.isdigit():
                var_type = ir.IntType(32)  # default to i32
            elif node.value.lower() in ['true', 'false']:
                var_type = ir.IntType(1)
            elif '.' in node.value and all(c.isdigit() or c == '.' or c == '-' or c == '+' or c.lower() == 'e' 
                                        for c in node.value):
                var_type = ir.DoubleType()# or FloatType
                
            # Hexadecimal literals
            elif node.value.startswith('0x'):
                # Handle
                var_type = ir.IntType(64)
                pass
            else:
                # Could be a variable name or other identifier
                raise ValueError(f"Cannot infer type for literal value: '{node.value}'. Node: {node}")
        else:
            # If not a string, what is it?
            raise ValueError(f"Unsupported literal type: {type(node.value)}")

    try:
        # Handle string literals
        if is_string_literal:
            # String literals are inherently pointers to char arrays
            return self._create_string_literal(node.value, builder, var_type, pointer_level)

        if debug:
            print("IS CHAR:", is_char_literal)
        if is_char_literal:
            inner = node.value[1:-1]  # strip quotes

            # Handle escape sequences
            if inner.startswith("\\"):
                escapes = {
                    "n": 10,   # newline
                    "t": 9,    # tab
                    "r": 13,   # carriage return
                    "0": 0,    # null character
                    "'": 39,   # single quote
                    '"': 34,   # double quote
                    "\\": 92,  # backslash
                }
                
                if len(inner) >= 2:
                    esc = inner[1:]
                    if esc in escapes:
                        char_value = escapes[esc]
                    else:
                        raise ValueError(f"Unknown escape sequence: '\\{esc}'")
                else:
                    raise ValueError(f"Invalid escape sequence in char literal: '{node.value}'")
            else:
                # Normal char - but handle case where inner might be a single actual character
                if len(inner) == 1:
                    char_value = ord(inner)
                elif len(inner) == 0:
                    raise ValueError(f"Empty char literal: '{node.value}'")
                else:
                    # This might be an already-processed escape sequence (like actual newline character)
                    # Take the first character's ASCII value
                    char_value = ord(inner[0])

            return ir.Constant(ir.IntType(8), char_value)
        
        # Boolean literals (true/false)
        if isinstance(var_type, ir.IntType) and var_type.width == 1:
            if isinstance(node.value, str) and node.value.lower() == 'true':
                return ir.Constant(var_type, 1)
            elif isinstance(node.value, str) and node.value.lower() == 'false':
                return ir.Constant(var_type, 0)
            elif isinstance(node.value, (int, float)):
                # Convert numeric value to boolean (0 = false, non-zero = true)
                return ir.Constant(var_type, 1 if node.value != 0 else 0)

        # Handle NULL pointer literals
        if isinstance(var_type, ir.PointerType) and (
            (isinstance(node.value, str) and (node.value == "0" or node.value.upper() == "NULL")) or
            (isinstance(node.value, (int, float)) and node.value == 0)
        ):
            return ir.Constant(var_type, None)

        # Handle multilevel NULL pointers
        if pointer_level > 0 and (
            (isinstance(node.value, str) and (node.value == "0" or node.value.upper() == "NULL")) or
            (isinstance(node.value, (int, float)) and node.value == 0)
        ):
            # Create appropriate null pointer type based on pointer level
            null_type = var_type
            for _ in range(pointer_level):
                null_type = ir.PointerType(null_type)
            return ir.Constant(null_type, None)

        # Integer literals
        if isinstance(var_type, ir.IntType):
            # Handle different types of input for integer literals
            if hasattr(self, '_expression_parse_integer_literal'):
                return self._expression_parse_integer_literal(node.value, var_type)
            else:
                # Fallback if the helper method doesn't exist
                if isinstance(node.value, str):
                    # Handle hexadecimal, octal, binary literals
                    try:
                        if node.value.startswith('0x') or node.value.startswith('0X'):
                            int_val = int(node.value, 16)
                        elif node.value.startswith('0b') or node.value.startswith('0B'):
                            int_val = int(node.value, 2)
                        elif node.value.startswith('0') and len(node.value) > 1 and all(c.isdigit() for c in node.value[1:]):
                            int_val = int(node.value, 8)
                        else:
                            # Try parsing as decimal
                            int_val = int(float(node.value))
                    except ValueError as e:
                        raise ValueError(f"Invalid integer literal: '{node.value}' - {e}")
                else:
                    # Already a numeric value
                    int_val = int(node.value)
                return ir.Constant(var_type, int_val)

        # Floating point literals
        if isinstance(var_type, (ir.FloatType, ir.DoubleType)):
            if isinstance(node.value, str):
                float_val = float(node.value)
            else:
                float_val = float(node.value)
            return ir.Constant(var_type, float_val)

        raise ValueError(f"Unsupported literal type for value: '{node.value}' with pointer_level: {pointer_level}")
    except ValueError as e:
        if debug:
            print(f"Error handling literal: {e}")
        raise ValueError(f"Invalid literal or undefined variable: '{node.value}'")
    
def _expression_parse_integer_literal(self: "Codegen", value: str, var_type: ir.IntType):
    """Parse integer literals supporting decimal, hexadecimal, octal, and binary formats"""
    try:
        # Handle hexadecimal literals
        if value.startswith('0x') or value.startswith('0X'):
            val = int(value, 16)
        # Handle binary literals  
        elif value.startswith('0b') or value.startswith('0B'):
            val = int(value, 2)
        # Handle octal literals (starts with 0 and has only digits)
        elif value.startswith('0') and len(value) > 1 and all(c.isdigit() for c in value[1:]):
            val = int(value, 8)
        # Handle decimal literals
        else:
            val = int(value)
    except ValueError as e:
        raise ValueError(f"Invalid integer literal: '{value}' - {e}")
    
    # Handle signed vs unsigned integer types
    if var_type in self.signed_int_types:
        return ir.Constant(var_type, val)
    
    # For unsigned integers, convert negatives to 2's complement
    if val < 0:
        val = (1 << var_type.width) + val
    return ir.Constant(var_type, val)

def handle_struct_access(self: "Codegen", node: ASTNode.ExpressionNode, builder: ir.IRBuilder, load_final=True):
    """
    Handles struct access chains safely.
    load_final: whether to load the final field (False if assigning)
    """
    debug = self.compiler.debug
    access_chain = self._flatten_struct_access(node)

    if debug:
        print(f"Access chain: {access_chain}")

    # Base variable
    base_name = access_chain[0]
    base_info = self.symbol_table.lookup(base_name)
    if not base_info:
        raise ValueError(f"Unknown struct variable: {base_name}")

    # Start with base pointer/value
    current_ptr = base_info.llvm_value
    current_type = base_info.data_type

    if debug:
        print(f"Starting with base: {base_name} of type {current_type}")

    # Walk through each field in the access chain
    for i, field_name in enumerate(access_chain[1:]):
        if debug:
            print(f"Accessing field: {field_name} in type: {current_type}")

        class_info = self.struct_table.get(current_type)
        if not class_info:
            raise ValueError(f"Unknown struct type: {current_type}")

        class_type = class_info.get("class_type_info")
        if field_name not in class_type.field_names:
            raise ValueError(f"Field '{field_name}' not found in struct '{current_type}'")

        # Get pointer to field via GEP
        field_ptr = self.get_struct_field_ptr(current_ptr, current_type, field_name, builder)

        # Update current_ptr:
        # - For intermediate fields: keep pointer
        # - For final field: load if load_final=True
        if i == len(access_chain[1:]) - 1:
            if load_final:
                current_ptr = builder.load(field_ptr, name=f"access_{field_name}")
            else:
                current_ptr = field_ptr  # For assignment, we return pointer
        else:
            current_ptr = field_ptr

        # Update type for next iteration
        current_type = class_type.field_types[class_type.field_names.index(field_name)]

    return current_ptr


def _flatten_struct_access(self: "Codegen", node):

    """
    Convert a nested struct access tree into a flat list of field names
    For example, a.b.c becomes ['a', 'b', 'c']
    """
    if node.node_type != NodeType.STRUCT_ACCESS:
        # If it's just a variable reference
        return [node.value]
    
    # If it's a struct access node
    left_parts = self._flatten_struct_access(node.left)
    right_part = node.right.value
    return left_parts + [right_part]


def _get_variable_address(self, var_name: str, builder: ir.IRBuilder, required_pointer_level: int = 1) -> ir.Value:
    """
    Get the address of a variable with appropriate pointer level.
    
    Args:
        var_name: Name of the variable
        builder: LLVM IR builder
        required_pointer_level: Required pointer level for the result
    
    Returns:
        LLVM value representing the address with correct pointer level
    """
    symbol = self.symbol_table.lookup(var_name)
    if not symbol:
        raise ValueError(f"Variable '{var_name}' not found")
    
    # The symbol's llvm_value is always an alloca (pointer to the variable's type)
    base_address = symbol.llvm_value
    
    if symbol.pointer_level == 0:
        # Regular variable - return its address
        if required_pointer_level == 1:
            return base_address  # Address of the variable
        else:
            # Need multiple levels of indirection - this is complex
            raise ValueError(f"Cannot create {required_pointer_level}-level pointer to non-pointer variable")
    else:
        # Pointer variable
        if required_pointer_level == symbol.pointer_level + 1:
            return base_address  # Address of the pointer variable
        elif required_pointer_level == symbol.pointer_level:
            return builder.load(base_address, name=f"load_{var_name}")  # Value of the pointer
        elif required_pointer_level < symbol.pointer_level:
            # Need to dereference
            result = builder.load(base_address, name=f"load_{var_name}")
            for _ in range(symbol.pointer_level - required_pointer_level):
                result = builder.load(result, name=f"deref_{var_name}")
            return result
        else:
            # Need more levels - complex case
            raise ValueError(f"Cannot create {required_pointer_level}-level pointer from {symbol.pointer_level}-level pointer")


def _handle_pointer_arithmetic(self, left_val: ir.Value, operator: str, right_val: ir.Value, 
                            builder: ir.IRBuilder, left_pointer_level: int = 0, 
                            right_pointer_level: int = 0) -> ir.Value:
    """
    Handle arithmetic operations involving pointers.
    
    Args:
        left_val: Left operand LLVM value
        operator: Arithmetic operator ('+', '-', etc.)
        right_val: Right operand LLVM value
        builder: LLVM IR builder
        left_pointer_level: Pointer level of left operand
        right_pointer_level: Pointer level of right operand
    
    Returns:
        Result of pointer arithmetic
    """
    # Pointer + integer or integer + pointer
    if (left_pointer_level > 0 and right_pointer_level == 0) or \
    (left_pointer_level == 0 and right_pointer_level > 0):
        
        if operator == '+':
            # Use GEP (GetElementPtr) for pointer arithmetic
            if left_pointer_level > 0:
                return builder.gep(left_val, [right_val], name="ptr_add")
            else:
                return builder.gep(right_val, [left_val], name="ptr_add")
        elif operator == '-' and left_pointer_level > 0:
            # Subtract integer from pointer
            neg_right = builder.neg(right_val, name="neg_offset")
            return builder.gep(left_val, [neg_right], name="ptr_sub")
    
    # Pointer - pointer (both operands are pointers)
    elif left_pointer_level > 0 and right_pointer_level > 0 and operator == '-':
        # Convert pointers to integers, subtract, then divide by element size
        left_int = builder.ptrtoint(left_val, ir.IntType(64), name="ptr_to_int_left")
        right_int = builder.ptrtoint(right_val, ir.IntType(64), name="ptr_to_int_right")
        diff = builder.sub(left_int, right_int, name="ptr_diff")
        
        # Get element size (this is simplified - real implementation needs type info)
        element_size = ir.Constant(ir.IntType(64), 1)  # Assuming byte-sized elements
        return builder.sdiv(diff, element_size, name="ptr_distance")
    
    # Regular arithmetic for non-pointer cases
    return None  # Let caller handle regular arithmetic

def get_struct_field_ptr(self, struct_ptr, struct_type_name: str, field_name: str, builder: ir.IRBuilder):
    """
    Returns the pointer to a struct field using GEP.
    struct_ptr: llvm pointer to the struct (e.g., self or local variable)
    struct_type_name: type name of the struct
    field_name: field to access
    """
    struct_type_info = self.struct_table[struct_type_name]["class_type_info"]

    if field_name not in struct_type_info.field_names:
        raise ValueError(f"Field '{field_name}' not found in struct '{struct_type_name}'.")

    field_index = struct_type_info.field_names.index(field_name)
    zero = ir.Constant(ir.IntType(32), 0)
    field_idx = ir.Constant(ir.IntType(32), field_index)

    # If struct_ptr is pointer to struct*, load once
    if isinstance(struct_ptr.type, ir.PointerType) and isinstance(struct_ptr.type.pointee, ir.PointerType):
        struct_ptr = builder.load(struct_ptr, name=f"{struct_type_name}_loaded")

    result = builder.gep(struct_ptr, [zero, field_idx], name=f"{struct_type_name}_{field_name}_ptr")
    
    return result



def handle_enum_access(self, node: ASTNode.ExpressionNode, builder: ir.IRBuilder, var_type, **kwargs):
    """
    Enhanced enum access handler that supports both integer and string enums
    """
    debug = getattr(self.compiler, 'debug', False)
    
    # Extract enum name and member name from the node
    if hasattr(node, 'enum_name') and hasattr(node, 'member_name'):
        enum_name = node.enum_name
        member_name = node.member_name
    elif hasattr(node, 'left') and hasattr(node, 'right'):
        enum_name = node.left.value if hasattr(node.left, 'value') else str(node.left)
        member_name = node.right.value if hasattr(node.right, 'value') else str(node.right)
    else:
        raise ValueError(f"Invalid enum access node structure: {node}")
    
    if debug:
        print(f"Handling enum access: {enum_name}::{member_name}")
    
    # Look up the enum type information
    if hasattr(self, 'enum_table'):
        enum_type_info = self.enum_table.get_enum(enum_name)
    else:
        enum_type_info = Datatypes.get_type(enum_name)
    
    if not enum_type_info:
        raise ValueError(f"Unknown enum type: {enum_name}")
    
    if not isinstance(enum_type_info, EnumTypeInfo):
        raise ValueError(f"'{enum_name}' is not an enum type")
    
    # Check if the member exists
    if not enum_type_info.has_member(member_name):
        available_members = list(enum_type_info.values.keys())
        raise ValueError(f"Enum member '{member_name}' not found in enum '{enum_name}'. "
                        f"Available members: {available_members}")
    
    if debug:
        member_value = enum_type_info.get_member_value(member_name)
        print(f"Enum {enum_name}::{member_name} has value {member_value} (type: {enum_type_info.enum_value_type.value})")
    
    # Return the appropriate LLVM constant
    if enum_type_info.is_string_enum():
        # For string enums, return the stored LLVM constant (i8*)
        llvm_constant = enum_type_info.get_llvm_constant(member_name)
        if llvm_constant is None:
            raise ValueError(f"LLVM constant not found for string enum member {enum_name}::{member_name}")
        return llvm_constant
    else:
        # For integer enums, create the constant
        member_value = enum_type_info.get_member_value(member_name)
        target_type = var_type if var_type else enum_type_info.get_llvm_type()
        return ir.Constant(target_type, member_value)

        
# Helper method additions to your existing class
def _evaluate_enum_constant(self, expr):
    """
    Enhanced constant expression evaluator (your existing implementation with minor improvements)
    """
    if expr.node_type == NodeType.LITERAL:
        if isinstance(expr.value, str):
            # Check if it's a quoted string (should not be evaluated as integer)
            if (expr.value.startswith('"') and expr.value.endswith('"')) or \
            (expr.value.startswith("'") and expr.value.endswith("'")):
                return None  # String literals are not integer constants
            
            try:
                # Handle different number formats
                if expr.value.startswith('0x') or expr.value.startswith('0X'):
                    return int(expr.value, 16)
                elif expr.value.startswith('0b') or expr.value.startswith('0B'):
                    return int(expr.value, 2)
                elif expr.value.startswith('0') and len(expr.value) > 1 and expr.value[1].isdigit():
                    return int(expr.value, 8)
                else:
                    return int(expr.value)
            except ValueError:
                return None  # Not a valid integer
        elif isinstance(expr.value, (int, float)):
            return int(expr.value)
    
    elif expr.node_type == NodeType.BINARY_OP:
        # Your existing binary operation handling
        left_val = self._evaluate_enum_constant(expr.left)
        right_val = self._evaluate_enum_constant(expr.right)
        
        if left_val is None or right_val is None:
            return None
            
        match expr.op:
            case '+':
                return left_val + right_val
            case '-':
                return left_val - right_val
            case '*':
                return left_val * right_val
            case '/':
                return left_val // right_val
            case '%':
                return left_val % right_val
            case '<<':
                return left_val << right_val
            case '>>':
                return left_val >> right_val
            case '&':
                return left_val & right_val
            case '|':
                return left_val | right_val
            case '^':
                return left_val ^ right_val
            case _:
                return None
    
    elif expr.node_type == NodeType.UNARY_OP:
        # Your existing unary operation handling
        operand_val = self._evaluate_enum_constant(expr.left)
        if operand_val is None:
            return None
            
        match expr.op:
            case '-':
                return -operand_val
            case '+':
                return operand_val
            case '~':
                return ~operand_val
            case _:
                return None
    
    return None 

# New handler for postfix operations
def handle_postfix_op(self: "Codegen", node: ASTNode.ExpressionNode, builder: ir.IRBuilder, var_type):
    """
    Handle postfix increment (++) and decrement (--) operations.
    Returns the original value before the operation.
    """
    operand = node.left
    op = node.op
    
    # The operand should be a variable or addressable expression
    if operand.node_type == NodeType.LITERAL and operand.value in self.symbol_table:
        # Simple variable case
        var_ptr = self.symbol_table[operand.value]
        
        # Load the current value
        current_value = builder.load(var_ptr.llvm_value, name=f"load_{operand.value}")
        
        # Perform the increment/decrement
        if op == '++':
            new_value = builder.add(current_value, ir.Constant(var_type, 1), name="postfix_inc")
        elif op == '--':
            new_value = builder.sub(current_value, ir.Constant(var_type, 1), name="postfix_dec")
        else:
            raise ValueError(f"Unsupported postfix operator: {op}")
        
        # Store the new value back
        builder.store(new_value, var_ptr.llvm_value)

        
        # Return the original value (postfix semantics)
        return current_value
        
    elif operand.node_type == NodeType.ARRAY_ACCESS:
        # Array element case: arr[index]++
        array_node = operand.left
        index_node = operand.right
        
        # Get the array pointer
        if array_node.node_type == NodeType.LITERAL and array_node.value in self.symbol_table:
            array_ptr = self.symbol_table[array_node.value]
        else:
            raise ValueError(f"Complex array expressions not supported in postfix operations")
        
        # Evaluate the index
        index_value = self.handle_expression(index_node, builder, ir.IntType(32))
        
        # Get the element pointer
        element_ptr = builder.gep(array_ptr, [ir.Constant(ir.IntType(32), 0), index_value], 
                                name=f"arr_elem_ptr")
        
        # Load current value
        current_value = builder.load(element_ptr, name="current_elem_value")
        
        # Perform the operation
        if op == '++':
            new_value = builder.add(current_value, ir.Constant(var_type, 1), name="postfix_inc")
        elif op == '--':
            new_value = builder.sub(current_value, ir.Constant(var_type, 1), name="postfix_dec")
        else:
            raise ValueError(f"Unsupported postfix operator: {op}")
        
        # Store the new value back
        builder.store(new_value, element_ptr)
        
        # Return the original value
        return current_value
        
    else:
        raise ValueError(f"Postfix operation not supported on expression type: {operand.node_type}")
