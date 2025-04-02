
from llvmlite import ir, binding
from typing import TYPE_CHECKING, Dict, Callable, Type
from astnodes import *
from lexer import *


if TYPE_CHECKING:
    from compiler import Compiler  # Only for type hints


class Codegen:
    def __init__(self, compiler: "Compiler"):
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

        self.type_map = {
            Datatypes.BOOL : ir.IntType(0),
            Datatypes.U8   : ir.IntType(8),
            Datatypes.U16  : ir.IntType(16),
            Datatypes.U32  : ir.IntType(32),
            Datatypes.U64  : ir.IntType(64),
            Datatypes.I8   : ir.IntType(8),
            Datatypes.I16  : ir.IntType(16),
            Datatypes.I32  : ir.IntType(32),
            Datatypes.I64  : ir.IntType(64),
            Datatypes.U0   : ir.VoidType(),
            Datatypes.F32  : ir.FloatType(),
            Datatypes.F64  : ir.DoubleType()
        }


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
        print(llvm_type)
        return llvm_type

    def handle_function_definition(self, node: ASTNode.FunctionDefinition, **kwargs):
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

    def handle_binary_expression(self, node: ASTNode.ExpressionNode, builder: ir.IRBuilder, var_size, **kwargs):
        # Parse the operator
        operator = node.op

        # evaluate left and right expressions to get LLVM values
        left = self.handle_expression(node.left, builder, var_size)
        right = self.handle_expression(node.right, builder, var_size)
         
        if operator == operators["ADD"]:
            return builder.add(left, right, name="sum")
        elif operator == operators["SUBTRACT"]:
            return builder.sub(left, right, name="sub")


    def handle_primary_expression(self, node: ASTNode.ExpressionNode, builder: ir.IRBuilder, var_type, **kwargs):
        if node.node_type == NodeType.REFERENCE and node.value == '&': 
            # TODO! we don't handle pointers now
            return self.handle_pointer(node, builder)
        elif node.node_type == NodeType.BINARY_OP:
            return self.handle_binary_expression(node, builder, var_type)
        elif node.node_type == NodeType.LITERAL:
            # Create an LLVM constant integer from the literal value
            # Find the variable from variables and get their value 
            return ir.Constant(var_type, int(node.value))
        # Handle other node types...


    def handle_expression(self, node: ASTNode.ExpressionNode, builder: ir.IRBuilder, var_type, **kwargs):

        # Define operator precedence
        precedence_map = {
            '+': 1, '-': 1,  # Addition, subtraction
            '*': 2, '/': 2, '%': 2,  # Multiplication, division, modulo
            '<': 3, '>': 3, '<=': 3, '>=': 3,  # Comparison
            '==': 4, '!=': 4,  # Equality
            '&&': 5,  # Logical AND
            '||': 6,  # Logical OR
        }
        
        print(node)

        # Start with parsing a primary expression
        return self.handle_primary_expression(node, builder, var_type)

        

    def handle_block(self, node):
        pass

    def handle_variable_declaration(self, node: ASTNode.VariableDeclaration, builder: ir.IRBuilder, **kwargs):
        # local variable 
        var_type = Datatypes.to_llvm_type(node.var_type)
        print(var_type)
        var = builder.alloca(var_type, name=node.name)  

        llvm_type = self.turn_variable_type_to_llvm_type(node.var_type)

        # Parse value from expression
        value = self.handle_expression(node.value, builder, llvm_type)

        builder.store(ir.Constant(var_type, value), var) 

        # Load the local variable's value
        local_value = builder.load(var, name="loaded_local")

        print("varible declaration llvm")

    def handle_variable_assignment(self, node, **kwargs):
        pass

    def handle_return(self, node: ASTNode.Return, builder: ir.IRBuilder, **kwargs):
        # If the value is a function
        # Get return type
        function_return_type = builder.function.function_type.return_type
        if node.expression:
            return_value = self.handle_expression(node.expression, builder, function_return_type)
            builder.ret(return_value)
        else:
            builder.ret_void()
            
    def handle_if_statement(self, node, **kwargs):
        pass

    def handle_while_loop(self, node, **kwargs):
        pass

    def handle_for_loop(self, node, **kwargs):
        pass

    def handle_comment(self, node, **kwargs):
        pass

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