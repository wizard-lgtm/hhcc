from llvmlite import ir, binding
from typing import TYPE_CHECKING, Dict, Callable, Type
from astnodes import *
from lexer import *


if TYPE_CHECKING:
    from compiler import Compiler  # Only for type hints


class Codegen:
    def __init__(self, compiler: "Compiler"):

        self.symbol_table = {}           # Maps variable names to their allocations
        self.function_table = {}         # Maps function names to their definitions
        self.struct_table = {}           # Maps struct type names to their definitions
        self.variable_types = {}         # Maps variable names to their type names

        self.compiler = compiler
        self.astnodes = compiler.astnodes
        if self.compiler.triple:
            self.triple = self.compiler.target.get_llvm_triple()
        else:
            self.triple = ""

        self.node_index = 0

        self.current_node = self.astnodes[self.node_index]

        # Node type to handler mapping
        self.node_handlers: Dict[Type, Callable] = {
            ASTNode.ExpressionNode: self.handle_expression,
            ASTNode.Block: self.handle_block,
            ASTNode.VariableDeclaration: self.handle_variable_declaration,
            ASTNode.VariableAssignment: self.handle_variable_assignment,
            ASTNode.Return: self.handle_return,
            ASTNode.FunctionDefinition: self.handle_function_definition,
            ASTNode.IfStatement: self.handle_if_statement,
            ASTNode.WhileLoop: self.handle_while_loop,
            ASTNode.ForLoop: self.handle_for_loop,
            ASTNode.Comment: self.handle_comment,
            ASTNode.FunctionCall: self.handle_function_call,
            ASTNode.Class: self.handle_class,
            ASTNode.Union: self.handle_union,
            ASTNode.Break: self.handle_break,
            ASTNode.Continue: self.handle_continue,
            ASTNode.ArrayDeclaration: self.handle_array_declaration,
            ASTNode.ArrayInitialization: self.handle_array_initialization,
            ASTNode.Pointer: self.handle_pointer,
            ASTNode.Reference: self.handle_reference
        }

        # Define correct LLVM types with appropriate signedness
        # Boolean is represented as i1
        bool_type = ir.IntType(1)
        # Unsigned types
        u8_type = ir.IntType(8)
        u16_type = ir.IntType(16)
        u32_type = ir.IntType(32)
        u64_type = ir.IntType(64)
        # Signed types - in LLVM IR, the types are the same but operations differ
        i8_type = ir.IntType(8)
        i16_type = ir.IntType(16)
        i32_type = ir.IntType(32)
        i64_type = ir.IntType(64)
        # Other types
        void_type = ir.VoidType()
        f32_type = ir.FloatType()
        f64_type = ir.DoubleType()

        self.type_map = {
            Datatypes.BOOL: bool_type,
            Datatypes.U8: u8_type,
            Datatypes.U16: u16_type,
            Datatypes.U32: u32_type,
            Datatypes.U64: u64_type,
            Datatypes.I8: i8_type,
            Datatypes.I16: i16_type,
            Datatypes.I32: i32_type,
            Datatypes.I64: i64_type,
            Datatypes.U0: void_type,
            Datatypes.F32: f32_type,
            Datatypes.F64: f64_type
        }

        self.type_signedness = {
            self.type_map[Datatypes.I8]: True,
            self.type_map[Datatypes.I16]: True,
            self.type_map[Datatypes.I32]: True,
            self.type_map[Datatypes.I64]: True,
            self.type_map[Datatypes.U8]: False,
            self.type_map[Datatypes.U16]: False,
            self.type_map[Datatypes.U32]: False,
            self.type_map[Datatypes.U64]: False,
            self.type_map[Datatypes.BOOL]: False,
        }


        self.signed_int_types = {i8_type, i16_type, i32_type, i64_type}
        self.unsigned_int_types = {bool_type, u8_type, u16_type, u32_type, u64_type}
        self.float_types = {f32_type, f64_type}

    def generation_error(self, message: str, node: 'ASTNode'):
        """Report an error with a formatted message, based on the ASTNode (instead of token)."""
        
        # Retrieve line and column information from the ASTNode (assuming these attributes exist)
        if hasattr(node, 'line') and hasattr(node, 'column'):
            line = node.line
            column = node.column
        else:
            line = column = -1  # If line/column info is missing, set to -1 for safety

        # If source code is available, try to get the line of code where the error occurred
        try:
            error_line = self.compiler.code.splitlines()[line - 1]  # Subtract 1 for 0-based index
        except IndexError:
            error_line = "[ERROR: Line out of range]"

        # Align the caret with the column position (ensure it's within bounds)
        caret_position = " " * (min(column, len(error_line))) + "^"

        # Print error message and source line context
        print(f"Generation Error: {message} at line {line}, column {column}")
        print(f"{error_line}")
        print(f"{caret_position}")
        
        # Print detailed node information
        print(f"Caused by ASTNode: {repr(node)}")

        # Raise an exception with the full error message
        raise Exception(f"{message} at line {line}, column {column}\n"
                        f"{error_line}\n"
                        f"{caret_position}\n"
                        f"Caused by ASTNode: {repr(node)}")

        
    def add_function(self, function: ASTNode.FunctionCall):
        self.function_map[function.name] = function
        
    def lookup_function(self, name: str) -> Optional[ASTNode.FunctionDefinition]:
        """
        Look up a function by name and return the FunctionDefinition.
        Returns None if the function is not found.
        """
        return self.function_map.get(name)

    def current_node(self):
        self.current_node = self.astnodes[self.node_index]
        return self.current_node

    def next_node(self):
        self.node_index += 1
        self.current_node = self.astnodes[self.node_index]
        return self.current_node


    def get_llvm_type(self, type: str):
        if type in self.type_map:
            return self.type_map[type]
        # handle pointer types
        elif type.endswith('*'):
            base_type = self.get_llvm_type(type[:-1]) # delete the * symbol and get it's type
            return ir.PointerType(base_type) 
        else: # other types (classes)
            # TODO! implement
            print("Non-Primitive types did not implemented. Returning a generic type (U8*)")
            return ir.PointerType(ir.IntType(8)) # return generic u8 pointer

    def gen(self):
        # Initialize LLVM
        binding.initialize()
        binding.initialize_native_target()
        binding.initialize_native_asmprinter()
        
        try:
            # Try to use the specified target
            print(self.triple)
            target = binding.Target.from_triple(self.triple)
        except RuntimeError:
            # get native target if specified target doesn't available
            print(f"Warning: Target '{self.triple}' not available, using native target instead")
            target = binding.Target.from_default_triple()
        
        target_machine = target.create_target_machine()
        
        # data layout string
        data_layout = target_machine.target_data
        
        # create the module 
        module = ir.Module(name=self.compiler.file)
        
        # Use the actual triple from the target machine to ensure compatibility
        module.triple = target_machine.triple
        module.data_layout = str(data_layout)
        
        # Build context for code generation
        self.module = module
        self.context = ir.context.Context()
        self.builder = None
        self.function = None
        
        # iterate astnodes for handler 
        for node in self.astnodes:
            self.process_node(node)
            
        return module
    
    def process_node(self, node, **kwargs):
        # Get the node's class type
        node_class = type(node)
        
        # Look up and call the appropriate handler
        if node_class in self.node_handlers:
            return self.node_handlers[node_class](node, **kwargs)
        else:
            print(f"Warning: No handler for node type {node_class.__name__}")
            return None
    
    def turn_variable_type_to_llvm_type(self, type: Datatypes):
        llvm_type  = self.type_map[type]
        return llvm_type

    def handle_function_definition(self, node: ASTNode.FunctionDefinition, **kwargs):
        name = node.name
        return_type = Datatypes.to_llvm_type(node.return_type)

        node_params: List[ASTNode.VariableDeclaration] = node.parameters

        llvm_params = [] 

        # Parse args
        for param in node_params:
            if param.is_user_typed:
                print("NOT IMPLEMENTED! user typed parameters")
            if param.is_pointer:
                print("NOT IMPLEMNTED, function pointer types")
            llvm_params.append(self.type_map[param.var_type])
        
        node.parameters
        func_type = ir.FunctionType(return_type, llvm_params)
        func = ir.Function(self.module, func_type, name)
        # Handle function body if we have
        if node.body:
            entry_block = func.append_basic_block("entry")
            builder = ir.IRBuilder(entry_block)
            # Parse block
            for body_node in node.body.nodes:
                self.process_node(body_node, builder=builder)
                
        self.symbol_table[func.name] = func


    def handle_binary_expression(self, node: ASTNode.ExpressionNode, builder: ir.IRBuilder, var_type, **kwargs):
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

    def get_variable_value(name: str):
        # Find variable and get it's value
        pass

    def handle_primary_expression(self, node: ASTNode.ExpressionNode, builder: ir.IRBuilder, var_type, **kwargs):
        if node.node_type == NodeType.REFERENCE and node.value == '&': 
            return self.handle_pointer(node, builder)
        elif node.node_type == NodeType.BINARY_OP:
            return self.handle_binary_expression(node, builder, var_type)
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

    
    def handle_expression(self, node: ASTNode.ExpressionNode, builder: ir.IRBuilder, var_type, **kwargs):
        if node is None:
            raise ValueError("Node is None, cannot handle expression.")

        print(node)
    
        match node.node_type:
            case NodeType.BINARY_OP:
                return self.handle_binary_expression(node, builder, var_type)

            case NodeType.REFERENCE:
                return self.handle_pointer(node, builder)

            case NodeType.FUNCTION_CALL:
                return self.handle_function_call(node, builder, **kwargs)

            case NodeType.STRUCT_ACCESS:
                return self.handle_struct_access(node, builder)

            case NodeType.LITERAL:
                return self._expression_handle_literal(node, builder, var_type)

            case _:
                raise ValueError(f"Unsupported expression node type: {node.node_type}")


    def _expression_handle_literal(self, node: ASTNode.ExpressionNode, builder: ir.IRBuilder, var_type):
        # Handle variable reference if it's in the symbol table
        if node.value in self.symbol_table:
            var_ptr = self.symbol_table[node.value].get("var")
            return builder.load(var_ptr, name=f"load_{node.value}")

        try:
            # Boolean literals (true/false)
            if isinstance(var_type, ir.IntType) and var_type.width == 1:
                if node.value.lower() == 'true':
                    return ir.Constant(var_type, 1)
                elif node.value.lower() == 'false':
                    return ir.Constant(var_type, 0)

            # Integer literals
            if isinstance(var_type, ir.IntType):
                return self._expression_parse_integer_literal(node.value, var_type)

            # Floating point literals
            if isinstance(var_type, (ir.FloatType, ir.DoubleType)):
                return ir.Constant(var_type, float(node.value))

            raise ValueError(f"Unsupported literal type for value: '{node.value}'")
        except ValueError:
            raise ValueError(f"Invalid literal or undefined variable: '{node.value}'")


    def _expression_parse_integer_literal(self, value: str, var_type: ir.IntType):
        val = int(value)

        if var_type in self.signed_int_types:
            return ir.Constant(var_type, val)

        # For unsigned integers, convert negatives to 2's complement
        if val < 0:
            val = (1 << var_type.width) + val

        return ir.Constant(var_type, val)

    def handle_struct_access(self, node: ASTNode.ExpressionNode, builder: ir.IRBuilder):
        # For example 't.a' where 't' is the struct variable and 'a' is the field
        struct_name = node.left.value       # 't'
        field_name = node.right.value       # 'a'

        # Lookup struct variable
        struct_info = self.symbol_table.get(struct_name)
        if not struct_info:
            raise ValueError(f"Unknown struct variable: {struct_name}")

        struct_ptr = struct_info.get("var")
        struct_type_name = struct_info.get("type_name")

        if struct_ptr is None or struct_type_name is None:
            raise ValueError(f"Invalid struct entry for '{struct_name}'.")

        # Get field index and type info
        class_info = self.struct_table.get(struct_type_name)
        if not class_info:
            raise ValueError(f"Unknown struct type: {struct_type_name}")

        class_type = class_info.get("class_type_info")
        if not class_type or field_name not in class_type.field_names:
            raise ValueError(f"Field '{field_name}' not found in struct '{struct_type_name}'.")

        field_index = class_type.field_names.index(field_name)

        field_ptr = self.get_struct_field_ptr(struct_name, field_name, builder)

        # Load field value
        dxx = builder.load(field_ptr, name=f"{struct_name}_{field_name}")
        print(dxx)
        return dxx

    def handle_block(self, node):
        pass

    def handle_variable_declaration(self, node: ASTNode.VariableDeclaration, builder: ir.IRBuilder, **kwargs):
            # Check if this is a pointer type declaration
            is_pointer = False
            base_type_name = node.var_type
            
            if node.var_type.endswith('*'):
                is_pointer = True
                base_type_name = node.var_type[:-1]  # Remove the asterisk
            
            # Get the base type
            base_type = Datatypes.to_llvm_type(base_type_name)
            
            # If it's a pointer, create a pointer to the base type
            if is_pointer:
                var_type = ir.PointerType(base_type)
            else:
                var_type = base_type
                
            print("var_type2", var_type)
            
            # Allocate space for the variable
            var = builder.alloca(var_type, name=node.name)
            
            # Store variable in symbol table
            self.symbol_table[node.name] = {
                "type_name": node.var_type,
                "var": var,
                "alloca": var,  # Add the alloca directly for pointer operations
                "is_pointer": is_pointer
            }
            
            # Handle initial value if present
            if node.value:
                value = self.handle_expression(node.value, builder, var_type)
                print("value from expression", value)
                
                if not value:
                    if is_pointer:
                        # Initialize to null pointer
                        value = ir.Constant(var_type, None)
                    else:
                        # Default to zero for non-pointer types
                        value = ir.Constant(var_type, 0)
                        
                builder.store(value, var)
            else:
                # Initialize pointers to null by default if no value is provided
                if is_pointer:
                    null_ptr = ir.Constant(var_type, None)
                    builder.store(null_ptr, var)
            
            return var  # Return the variable pointer

    def get_struct_field_ptr(self, struct_name: str, field_name: str, builder: ir.IRBuilder):
        """
        Returns the pointer to a struct field using GEP.
        """
        # Ensure the struct variable exists
        if struct_name not in self.symbol_table:
            raise ValueError(f"Struct variable '{struct_name}' not found.")

        struct_info = self.symbol_table[struct_name]
        struct_ptr = struct_info["var"]
        struct_type_name = struct_info["type_name"]

        if struct_type_name not in self.struct_table:
            raise ValueError(f"Struct type '{struct_type_name}' not found.")

        struct_type_info = self.struct_table[struct_type_name]["class_type_info"]

        if field_name not in struct_type_info.field_names:
            raise ValueError(f"Field '{field_name}' not found in struct '{struct_type_name}'.")

        field_index = struct_type_info.field_names.index(field_name)
        zero = ir.Constant(ir.IntType(32), 0)
        field_idx = ir.Constant(ir.IntType(32), field_index)

        return builder.gep(struct_ptr, [zero, field_idx], name=f"{struct_name}_{field_name}_ptr")


    def handle_variable_assignment(self, node: ASTNode.VariableAssignment, builder: ir.IRBuilder, **kwargs):
        # 1. Get variable name and check if it's a struct field access
        var_name = node.name
        
        # Check if this is a struct field assignment (contains a dot)
        if '.' in var_name:
            struct_name, field_name = var_name.split('.')
            
            # Ensure the struct variable exists
            if struct_name not in self.symbol_table:
                raise ValueError(f"Struct variable '{struct_name}' not found in symbol table.")
            
            # Get the struct type information
            struct_info = self.symbol_table[struct_name]
            struct_ptr = struct_info["var"]
            struct_type_name = struct_info["type_name"]
            
            # Ensure the struct type exists in the struct table
            if struct_type_name not in self.struct_table:
                raise ValueError(f"Struct type '{struct_type_name}' not found in struct table.")
            
            # Get the struct type definition
            struct_type_info = self.struct_table[struct_type_name]["class_type_info"]
            
            # Find the field index in the struct
            if field_name not in struct_type_info.field_names:
                raise ValueError(f"Field '{field_name}' not found in struct '{struct_type_name}'.")
            
            field_index = struct_type_info.field_names.index(field_name)
            
            # Create a GEP instruction to get the field pointer
            # First, get the struct pointer
            zero = ir.Constant(ir.IntType(32), 0)
            field_idx = ir.Constant(ir.IntType(32), field_index)
            field_ptr = builder.gep(struct_ptr, [zero, field_idx], name=f"{struct_name}_{field_name}_ptr")
            
            field_ptr = self.get_struct_field_ptr(struct_name, field_name, builder)

            
            # Get the field type from the struct type
            field_type = struct_type_info.llvm_type.elements[field_index]
            
            # Evaluate right-hand side expression
            value = self.handle_expression(node.value, builder, field_type)
            
            # Handle type casting if needed
            if value.type != field_type:
                value = self._cast_value(value, field_type, builder)
            
            # Store the value in the field
            builder.store(value, field_ptr)
        else:
            # Regular variable assignment (existing code)
            if var_name not in self.symbol_table:
                raise ValueError(f"Variable '{var_name}' not found in symbol table. It must be declared before assignment.")
            
            # Retrieve variable pointer and type
            var_ptr = self.symbol_table[var_name]["var"]
            var_type = var_ptr.type.pointee
            
            # Evaluate right-hand side expression
            value = self.handle_expression(node.value, builder, var_type)
            
            print("value from expression", value)
            
            # Handle type casting if types don't match
            if value.type != var_type:
                value = self._cast_value(value, var_type, builder)
            
            # Store the evaluated value into the variable
            builder.store(value, var_ptr)

    def _cast_value(self, value, target_type, builder):
        """Casts a value to the target LLVM type, inserting necessary instructions."""
        from llvmlite import ir

        src_type = value.type

        # Integer to Integer
        if isinstance(target_type, ir.IntType) and isinstance(src_type, ir.IntType):
            if target_type.width > src_type.width:
                return builder.sext(value, target_type, name="sext") if Datatypes.is_signed_type(target_type) else builder.zext(value, target_type, name="zext")
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
            return builder.sitofp(value, target_type, name="sitofp") if Datatypes.is_signed_type(src_type) else builder.uitofp(value, target_type, name="uitofp")

        # Float to Int
        if isinstance(target_type, ir.IntType) and isinstance(src_type, (ir.FloatType, ir.DoubleType)):
            return builder.fptosi(value, target_type, name="fptosi") if Datatypes.is_signed_type(target_type) else builder.fptoui(value, target_type, name="fptoui")

        # Pointer to Pointer
        if isinstance(target_type, ir.PointerType) and isinstance(src_type, ir.PointerType):
            return builder.bitcast(value, target_type, name="ptr_cast")

        raise TypeError(f"Incompatible types for assignment: {src_type} cannot be assigned to {target_type}")

        

    def handle_return(self, node: ASTNode.Return, builder: ir.IRBuilder, **kwargs):
        # If the value is a function
        # Get return type
        function_return_type = builder.function.function_type.return_type
        if node.expression:
            return_value = self.handle_expression(node.expression, builder, function_return_type)
            builder.ret(return_value)
        else:
            builder.ret_void()
            
    def handle_if_statement(self, node: ASTNode.IfStatement, builder: ir.IRBuilder, **kwargs):
        # Evaluate the condition
        condition = self.handle_expression(node.condition, builder, self.type_map[Datatypes.BOOL])

        condition = self.handle_expression(node.condition, builder, self.type_map[Datatypes.BOOL])
        if condition.type != ir.IntType(1):
            condition = builder.trunc(condition, ir.IntType(1))


        # Create basic blocks for the 'then', 'else', and 'merge' sections
        then_block = builder.append_basic_block("if.then")
        else_block = builder.append_basic_block("if.else") if node.else_body else None
        merge_block = builder.append_basic_block("if.end")

        # Branch based on the condition
        if else_block:
            builder.cbranch(condition, then_block, else_block)
        else:
            builder.cbranch(condition, then_block, merge_block)

        # Generate code for the 'then' block
        builder.position_at_end(then_block)
        for stmt in node.if_body.nodes:
            self.process_node(stmt, builder=builder)
        if not builder.block.is_terminated:
            builder.branch(merge_block)

        # Generate code for the 'else' block if it exists
        if else_block:
            builder.position_at_end(else_block)
            for stmt in node.else_body.nodes:
                self.process_node(stmt, builder=builder)
            if not builder.block.is_terminated:
                builder.branch(merge_block)

        # Position the builder at the merge block
        builder.position_at_end(merge_block)

    def handle_while_loop(self, node: ASTNode.WhileLoop, builder: ir.IRBuilder):
        # Create basic blocks for the loop
        loop_cond_block = builder.append_basic_block("while.cond")
        loop_body_block = builder.append_basic_block("while.body")
        loop_end_block = builder.append_basic_block("while.end")

        # Branch to the condition block
        builder.branch(loop_cond_block)

        # Generate code for the condition block
        builder.position_at_end(loop_cond_block)
        condition = self.handle_expression(node.condition, builder, self.type_map[Datatypes.BOOL])
        builder.cbranch(condition, loop_body_block, loop_end_block)

        # Generate code for the body block
        builder.position_at_end(loop_body_block)
        for stmt in node.body.nodes:
            self.process_node(stmt, builder=builder)
        builder.branch(loop_cond_block)

        # Position the builder at the end block
        builder.position_at_end(loop_end_block)

    def handle_for_loop(self, node: ASTNode.ForLoop, builder: ir.IRBuilder, **kwargs):
        # Create basic blocks for the for loop
        loop_cond_block = builder.append_basic_block("for.cond")
        loop_body_block = builder.append_basic_block("for.body")
        loop_update_block = builder.append_basic_block("for.update")
        loop_end_block = builder.append_basic_block("for.end")

        # Initialize the loop variable(s) (if any)
        if node.initialization:
            self.process_node(node.initialization, builder=builder)

        # Branch to the condition block
        builder.branch(loop_cond_block)
        
        # Generate code for the condition block
        builder.position_at_end(loop_cond_block)
        if node.condition:
            condition = self.handle_expression(node.condition, builder, self.type_map[Datatypes.BOOL])
            builder.cbranch(condition, loop_body_block, loop_end_block)
        else:
            # If there's no condition, assume an infinite loop unless we break somewhere
            builder.branch(loop_body_block)

        # Generate code for the body block
        builder.position_at_end(loop_body_block)
        for stmt in node.body.nodes:
            self.process_node(stmt, builder=builder)

        # After the body, move to the update block
        builder.branch(loop_update_block)

        # Generate code for the update block (if any)
        builder.position_at_end(loop_update_block)
        if node.update:
            self.process_node(node.update, builder=builder)

        # Return to the condition block for the next iteration
        builder.branch(loop_cond_block)
        
        # Position the builder at the end block
        builder.position_at_end(loop_end_block)

    def handle_comment(self, node: ASTNode.Comment, **kwargs):
        if 'builder' in kwargs and kwargs['builder'] is not None:
            builder = kwargs['builder']
            comment_text = node.text
            if node.is_inline:
                comment_text = "INLINE: " + comment_text
                
            # Add a custom metadata node that we can convert to a comment when printing
            comment_md = builder.module.add_metadata([ir.MetaDataString(builder.module, comment_text)])
            
    def handle_function_call(self, node, builder: ir.IRBuilder, var_type=None, **kwargs):
        # Handle either dedicated FunctionCall nodes or ExpressionNode with FUNCTION_CALL type
        if isinstance(node, ASTNode.FunctionCall):
            func_name = node.name
            arguments = node.arguments
        elif node.node_type == NodeType.FUNCTION_CALL:
            # Extract function name from the left part of the expression
            if node.left and node.left.node_type == NodeType.LITERAL:
                func_name = node.left.value
            else:
                raise ValueError("Invalid function call format: function name not found")
            
            # Get arguments from node.right or node.arguments depending on your AST structure
            # This might need adjustment based on your specific AST structure
            arguments = node.arguments if hasattr(node, 'arguments') else []
        else:
            raise TypeError("Expected a function call node")
        
        # Get the function from symbol table
        func = self.symbol_table.get(func_name)
        
        if not func:
            # If the function isn't found in the symbol table, raise an error
            raise ValueError(f"Function {func_name} not defined")
        
        # Prepare arguments for the function call
        llvm_args = []
        for arg in arguments:
            # Process each argument as an expression
            llvm_arg = self.handle_expression(arg, builder, None)
            
            # Type checking can be done here if you have type information
            llvm_args.append(llvm_arg)
        
        # Make the function call in LLVM IR
        if hasattr(func, 'return_value') and func.return_value == str(Datatypes.U0):
            # For void functions
            builder.call(func, llvm_args)
            return None
        else:
            # For functions that return a value
            result = builder.call(func, llvm_args)
            return result

    class ClassTypeInfo:
                def __init__(self, llvm_type, field_names, parent_type=None, node: ASTNode.Class = None):
                    self.llvm_type = llvm_type
                    self.field_names = field_names
                    self.parent = parent_type
                    self.node = node
                
                def get_llvm_type(self):
                    return self.llvm_type
                
                def get_fields(self):
                    return [(name, self.llvm_type.elements[i]) for i, name in enumerate(self.field_names)]
                
                def get_field_index(self, field_name):
                    try:
                        return self.field_names.index(field_name)
                    except ValueError:
                        if self.parent:
                            # Check if the field exists in the parent class
                            for i, (name, _) in enumerate(self.parent.get_fields()):
                                if name == field_name:
                                    return i
                        raise Exception(f"Unknown field '{field_name}' in class '{self.node}'")
                def __repr__(self):
                    return f"<ClassTypeInfo: fields={self.field_names}, parent={self.parent}, llvm_type={self.llvm_type}>"

    def handle_class(self, node: ASTNode.Class, **kwargs):
        """
        Handles the definition of a class in the source code and generates
        the corresponding identified LLVM struct type.

        Args:
            node: The AST node representing the class definition.
            **kwargs: Additional keyword arguments (not used in this snippet).
        """
        # Get the parent class info if any
        parent_type_info = None
        if node.parent:
            parent_type_info = Datatypes.get_type(node.parent)
            if not parent_type_info:
                raise Exception(f"Unknown parent class '{node.parent}'")

        # --- START: Handling Identified LLVM Struct Types ---

        # 1. Create the identified (named) struct type in the LLVM context.
        # This type is initially 'opaque' (its contents are not yet defined).
        # We use a standard naming convention like '%struct.ClassName'.
        llvm_struct_name = f"%struct.{node.name}"
        # Use global_context to get the type by name. If it doesn't exist, it's created.
        struct_type = ir.global_context.get_identified_type(llvm_struct_name)

        # 2. Collect the LLVM types for each field in the class.
        # We need to do this *after* creating the identified type so that
        # self-referential fields (like 'Node next' in a Node class) can
        # correctly refer to a pointer to 'struct_type'.
        field_llvm_types = []
        field_names = []

        # If there's a parent, include its field types first (inheritance).
        # Ensure inherited_fields provides LLVM types compatible with the parent's struct layout.
        if parent_type_info and hasattr(parent_type_info, 'get_fields'):
            inherited_fields = parent_type_info.get_fields() # Assuming this returns [(name, llvm_type)]
            for field_name, field_type in inherited_fields:
                field_names.append(field_name)
                field_llvm_types.append(field_type)

        # Process each field defined directly in this class.
        for field in node.fields:
            # Convert the source type name to its corresponding LLVM type.
            # Special handling is needed here for fields that are pointers to *this* class.
            if field.var_type == node.name:
                # If the field type is the same as the class being defined,
                # it should be a pointer to this identified struct type.
                field_llvm_type = struct_type.as_pointer()
            else:
                # For other types, use the standard conversion.
                # Datatypes.to_llvm_type should handle base types (U8, etc.)
                # and potentially look up other defined class types (usually returning pointers).
                field_llvm_type = Datatypes.to_llvm_type(field.var_type)

            field_llvm_types.append(field_llvm_type)
            field_names.append(field.name)

        # 3. Set the body of the identified struct type.
        # This defines the actual layout (the sequence of field types) for the struct.
        # This step completes the definition of the 'opaque' type created earlier.
        struct_type.set_body(*field_llvm_types) # Use * to unpack the list of types

        # --- END: Handling Identified LLVM Struct Types ---

        # Create a wrapper object to store additional information about our class,
        # including the now-defined LLVM struct type.
        class_type_info = self.ClassTypeInfo(struct_type, field_names, parent_type_info, node)

        # Register the class type info in your type system.
        # This makes the 'Node' source type name map to the 'class_type_info' object,
        # which contains the LLVM 'struct_type'.
        Datatypes.add_type(node.name, class_type_info)

        # Store the struct info in your internal table (if needed).
        # Ensure you are storing the class name as the key.
        self.struct_table[node.name] = {'name': node.name, 'class_type_info': class_type_info}

        # This function typically doesn't return an LLVM value, just defines the type.
        return None


    def handle_union(self, node, **kwargs):
        pass

    def handle_break(self, node, **kwargs):
        pass

    def handle_continue(self, node, **kwargs):
        pass

    def handle_array_declaration(self, node, **kwargs):
        pass

    def handle_array_initialization(self, node, **kwargs):
        pass


    def handle_pointer(self, node, builder, **kwargs):
        """
        Handle pointer operations (&) to get the address of a variable.
        
        Args:
            node: The AST node representing the pointer operation
            builder: The LLVM IR builder
            **kwargs: Additional keyword arguments
            
        Returns:
            The pointer value as an LLVM IR value
        """
        # The child node should be the variable we're getting the address of
        child_node = node.left [0] if node.left else None
        if not child_node:
            raise ValueError("Invalid pointer operation: missing target operand")
        
        # Get the variable reference
        if child_node.node_type == NodeType.REFERENCE:
            # Get the variable name
            var_name = child_node.value
            
            # Look up the variable in the symbol table
            if var_name not in self.symbol_table:
                raise ValueError(f"Cannot take address of undefined variable: {var_name}")
            
            var_alloca = self.symbol_table[var_name]['alloca']
            return var_alloca  # The alloca instruction itself is the pointer
        elif child_node.node_type == NodeType.STRUCT_ACCESS:
            # Handle getting address of a struct field
            struct_ptr = self.handle_struct_access(child_node, builder, get_pointer=True)
            return struct_ptr
        else:
            raise ValueError(f"Cannot take address of {child_node.node_type}")

    def handle_reference(self, node, builder, **kwargs):
        """
        Handle variable references (dereferencing pointers with *).
        
        Args:
            node: The AST node representing the reference operation
            builder: The LLVM IR builder
            **kwargs: Additional keyword arguments
            
        Returns:
            The dereferenced value as an LLVM IR value
        """
        # Check if this is a dereference operation (*)
        if node.value == '*':
            # The child node should be the pointer we're dereferencing
            child_node = node.children[0] if node.children else None
            if not child_node:
                raise ValueError("Invalid dereference operation: missing pointer operand")
            
            # Get the pointer value
            ptr_value = self.handle_expression(child_node, builder, None)
            
            # Determine the pointee type
            ptr_type = ptr_value.type
            if not isinstance(ptr_type, ir.PointerType):
                raise ValueError("Cannot dereference a non-pointer value")
            
            # Load the value from the pointer
            return builder.load(ptr_value, name="deref")
        
        # If it's a normal variable reference (not a dereference)
        var_name = node.value
        
        # Look up the variable in the symbol table
        if var_name not in self.symbol_table:
            raise ValueError(f"Reference to undefined variable: {var_name}")
        
        var_info = self.symbol_table[var_name]
        var_alloca = var_info['alloca']
        
        # Load the value from the alloca
        return builder.load(var_alloca, name=f"{var_name}_value")