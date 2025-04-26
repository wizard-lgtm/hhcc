from llvmlite import ir, binding
from typing import TYPE_CHECKING, Dict, Callable, Type
from astnodes import *
from lexer import *


if TYPE_CHECKING:
    from compiler import Compiler  # Only for type hints


class Codegen:
    def __init__(self, compiler: "Compiler"):
        self.symbol_table = {} 
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
        outer_scope = self.symbol_table

        # Create a new symbol table for this function
        self.symbol_table = {}

        name = node.name
        return_type = Datatypes.to_llvm_type(node.return_type)
        print(f"RETURN TYPE: {node.return_type}")

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

    
        # Restore the outer scope when function processing is complete
        self.symbol_table = outer_scope

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
            if is_integer:
                if is_signed:
                    if is_debug:
                        print(f"DEBUG - Using SIGNED integer comparison (==)")
                    return builder.icmp_signed('==', left, right, name="seq")
                else:
                    if is_debug:
                        print(f"DEBUG - Using UNSIGNED integer comparison (==)")
                    return builder.icmp_unsigned('==', left, right, name="ueq")
            else:
                if is_debug:
                    print(f"DEBUG - Using floating point comparison (==)")
                return builder.fcmp_ordered('==', left, right, name="feq")
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
            return None
            
        if node.node_type == NodeType.BINARY_OP:
            return self.handle_binary_expression(node, builder, var_type)
        elif node.node_type == NodeType.REFERENCE and node.value == '&':
            return self.handle_pointer(node, builder)
        elif node.node_type == NodeType.LITERAL:
            # First check if this is actually a variable reference
            if node.value in self.symbol_table:
                # It's a variable name, load its value
                var_ptr = self.symbol_table[node.value]
                # Get the actual variable type to pass to any further expressions
                return builder.load(var_ptr, name=f"load_{node.value}")
            else:
                # It's an actual literal value, create a constant
                try:
                    # Handle different literal types based on var_type
                    if isinstance(var_type, ir.IntType):
                        # Check if the type should be signed or unsigned for proper parsing
                        if var_type in self.signed_int_types:
                            # Parse as signed integer
                            return ir.Constant(var_type, int(node.value))
                        else:
                            # Parse as unsigned integer, ensure no negative values
                            val = int(node.value)
                            if val < 0:
                                # Convert negative values to their 2's complement representation
                                val = (1 << var_type.width) + val
                            return ir.Constant(var_type, val)
                    elif isinstance(var_type, (ir.FloatType, ir.DoubleType)):
                        return ir.Constant(var_type, float(node.value))
                    else:
                        raise ValueError(f"Unsupported literal type for value: '{node.value}'")
                except ValueError:
                    raise ValueError(f"Invalid literal or undefined variable: '{node.value}'")
        else:
            raise ValueError(f"Unsupported expression node type: {node.node_type}")

    def handle_block(self, node):
        pass

    def handle_variable_declaration(self, node: ASTNode.VariableDeclaration, builder: ir.IRBuilder, **kwargs):
        # Allocate space for the variable
        var_type = Datatypes.to_llvm_type(node.var_type)
        var = builder.alloca(var_type, name=node.name)  

        # Store variable in symbol table
        self.symbol_table[node.name] = var
        
        # Handle initial value if present
        if node.value:
            value = self.handle_expression(node.value, builder, var_type)
            builder.store(value, var)
        
        return var # Return the variable pointer
    
    def handle_variable_assignment(self, node: ASTNode.VariableAssignment, builder: ir.IRBuilder, **kwargs):
        # Get the variable name
        var_name = node.name
        
        # Check if the variable exists in the symbol table
        if var_name not in self.symbol_table:
            raise ValueError(f"Variable '{var_name}' not found in symbol table. It must be declared before assignment.")
        
        # Get the pointer to the variable
        var_ptr = self.symbol_table[var_name]
        
        # Get the variable's LLVM type
        var_type = var_ptr.type.pointee
        
        # Evaluate the right-hand side expression
        value = self.handle_expression(node.value, builder, var_type)
        
        # If types don't match, try to insert a cast
        if value.type != var_type:
            if isinstance(var_type, ir.IntType) and isinstance(value.type, ir.IntType):
                # Integer to integer cast
                if var_type.width > value.type.width:
                    # Extending the integer
                    if Datatypes.is_signed_type(var_type):
                        value = builder.sext(value, var_type, name="sext")
                    else:
                        value = builder.zext(value, var_type, name="zext")
                else:
                    # Truncating the integer
                    value = builder.trunc(value, var_type, name="trunc")
            elif isinstance(var_type, (ir.FloatType, ir.DoubleType)) and isinstance(value.type, (ir.FloatType, ir.DoubleType)):
                # Float to float cast
                if isinstance(var_type, ir.DoubleType) and isinstance(value.type, ir.FloatType):
                    value = builder.fpext(value, var_type, name="fpext")
                elif isinstance(var_type, ir.FloatType) and isinstance(value.type, ir.DoubleType):
                    value = builder.fptrunc(value, var_type, name="fptrunc")
            elif isinstance(var_type, (ir.FloatType, ir.DoubleType)) and isinstance(value.type, ir.IntType):
                # Integer to float cast
                if Datatypes.is_signed_type(value.type):
                    value = builder.sitofp(value, var_type, name="sitofp")
                else:
                    value = builder.uitofp(value, var_type, name="uitofp")
            elif isinstance(var_type, ir.IntType) and isinstance(value.type, (ir.FloatType, ir.DoubleType)):
                # Float to integer cast
                if Datatypes.is_signed_type(var_type):
                    value = builder.fptosi(value, var_type, name="fptosi")
                else:
                    value = builder.fptoui(value, var_type, name="fptoui")
            else:
                # Handle pointer types or other more complex cast scenarios
                if isinstance(var_type, ir.PointerType) and isinstance(value.type, ir.PointerType):
                    value = builder.bitcast(value, var_type, name="ptr_cast")
                else:
                    raise TypeError(f"Incompatible types for assignment: {value.type} cannot be assigned to {var_type}")
        
        # Store the value in the variable
        builder.store(value, var_ptr)

    def handle_return(self, node: ASTNode.Return, builder: ir.IRBuilder, **kwargs):
        # If the value is a function
        # Get return type
        function_return_type = builder.function.function_type.return_type
        if node.expression:
            return_value = self.handle_expression(node.expression, builder, function_return_type)
            builder.ret(return_value)
        else:
            builder.ret_void()
            
    def handle_if_statement(self, node: ASTNode.IfStatement, builder: ir.IRBuilder):
        # Evaluate the condition
        condition = self.handle_expression(node.condition, builder, self.type_map[Datatypes.BOOL])

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
            builder.branch(merge_block)

        # Position the builder at the merge block
        builder.position_at_end(merge_block)

    def handle_while_loop(self, node, **kwargs):
        pass

    def handle_for_loop(self, node, **kwargs):
        pass

    def handle_comment(self, node: ASTNode.Comment, **kwargs):
        if 'builder' in kwargs and kwargs['builder'] is not None:
            builder = kwargs['builder']
            comment_text = node.text
            if node.is_inline:
                comment_text = "INLINE: " + comment_text
                
            # Add a custom metadata node that we can convert to a comment when printing
            comment_md = builder.module.add_metadata([ir.MetaDataString(builder.module, comment_text)])
            
        

    def handle_function_call(self, node, **kwargs):
        pass

    def handle_class(self, node, **kwargs):
        pass

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

    def handle_pointer(self, node, **kwargs):
        pass

    def handle_reference(self, node, **kwargs):
        pass