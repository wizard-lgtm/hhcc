
from llvmlite import ir, binding
from typing import TYPE_CHECKING, Dict, Callable, Type
from astnodes import *
from lexer import *
from src.codegen.symboltable import Symbol, SymbolKind, create_function_symbol

def handle_function_definition(self, node: ASTNode.FunctionDefinition, builder: Optional[ir.IRBuilder] = None, **kwargs):
    """Handle function definition with the new symbol table."""
    name = node.name
    return_type = Datatypes.to_llvm_type(node.return_type)

    node_params: List[ASTNode.VariableDeclaration] = node.parameters
    llvm_params = []
    param_types = []

    # Parse args
    for param in node_params:
        if param.is_user_typed:
            print("NOT IMPLEMENTED! user typed parameters")
        if param.is_pointer:
            print("NOT IMPLEMENTED, function pointer types")
        
        param_type = self.type_map[param.var_type]
        llvm_params.append(param_type)
        param_types.append(param.var_type)
    
    # Create the function type and function
    func_type = ir.FunctionType(return_type, llvm_params)
    func = ir.Function(self.module, func_type, name)
    
    # Create and store the function symbol
    func_symbol = create_function_symbol(
        name=name,
        ast_node=node,
        return_type=node.return_type,
        parameter_types=param_types,
        llvm_function=func
    )
    self.symbol_table.define(func_symbol)
    
    # Also store in function map for backward compatibility
    self.function_map[name] = func
    
    # Handle function body if we have one
    if node.body:
        # Create a new scope for the function body
        self.symbol_table.enter_scope()
        
        # Create the entry block and builder
        entry_block = func.append_basic_block("entry")
        local_builder = ir.IRBuilder(entry_block)
        
        # Define function parameters in the symbol table
        for i, (param, llvm_param) in enumerate(zip(node_params, func.args)):
            # Allocate space for the parameter
            param_ptr = local_builder.alloca(llvm_param.type, name=f"{param.name}_param")
            local_builder.store(llvm_param, param_ptr)
            
            # Add parameter to symbol table
            param_symbol = Symbol(
                name=param.name,
                kind=SymbolKind.PARAMETER,
                ast_node=param,
                data_type=param.var_type,
                llvm_type=llvm_param.type,
                llvm_value=param_ptr,
                scope_level=self.symbol_table.current_scope_level
            )
            self.symbol_table.define(param_symbol)
        
        # Process the function body
        self.process_node(node.body, builder=local_builder)
        
        # Ensure the function has a return statement if needed
        last_block = local_builder.block
        if not last_block.is_terminated:
            if return_type == ir.VoidType():
                local_builder.ret_void()
            else:
                # For non-void functions, add a default return value
                local_builder.ret(ir.Constant(return_type, 0))
        
        # Exit the function scope
        self.symbol_table.exit_scope()
    
    return func


def handle_function_call(self, node, builder: ir.IRBuilder, var_type=None, **kwargs):
    """Handle function call with the new symbol table."""
    # Debug the AST node structure to help understand its format

    
    # Handle either dedicated FunctionCall nodes or ExpressionNode with FUNCTION_CALL type
    if isinstance(node, ASTNode.FunctionCall):
        func_name = node.name
        arguments = node.arguments
    elif node.node_type == NodeType.FUNCTION_CALL:
        # Extract function name
        if hasattr(node, 'name'):
            func_name = node.name
        elif hasattr(node, 'value') and node.value:
            # Sometimes the function name is stored in value
            func_name = node.value
        elif hasattr(node, 'left') and node.left:
            if hasattr(node.left, 'value'):
                func_name = node.left.value
            elif hasattr(node.left, 'name'):
                func_name = node.left.name
            else:
                raise ValueError("Invalid function call format: function name not found")
        else:
            raise ValueError("Invalid function call format: function name not found")
        
        # Extract arguments - handle the various ways they might be stored
        arguments = []
        
        # Check if arguments are stored directly as a property
        if hasattr(node, 'arguments') and node.arguments is not None:
            arguments = node.arguments
        # Check if arguments are stored in a right property
        elif hasattr(node, 'right') and node.right is not None:
            if isinstance(node.right, list):
                arguments = node.right
            else:
                # If right is a single node but represents multiple arguments
                if hasattr(node.right, 'arguments') and node.right.arguments is not None:
                    arguments = node.right.arguments
                else:
                    arguments = [node.right]
    else:
        raise TypeError("Expected a function call node")
    
    # Look up the function in the symbol table
    func_symbol = self.symbol_table.lookup(func_name)
    
    # For backward compatibility, also check the function map
    func = None
    if func_symbol:
        func = func_symbol.llvm_value
    elif func_name in self.function_map:
        func = self.function_map[func_name]
    
    if not func:
        raise ValueError(f"Function {func_name} not defined")
    
    # Get the expected argument types from the function type
    expected_arg_types = func.function_type.args
    
    # Check if the number of arguments matches
    if len(arguments) != len(expected_arg_types):
        raise ValueError(f"Function {func_name} expects {len(expected_arg_types)} arguments, but {len(arguments)} were provided")
    
    # Prepare arguments for the function call
    llvm_args = []
    for i, arg in enumerate(arguments):
        # Process each argument as an expression
        llvm_arg = self.handle_expression(arg, builder, None)
        
        # Make sure the argument types match - cast if necessary
        expected_type = expected_arg_types[i]
        if llvm_arg.type != expected_type:
            # Cast the argument to the expected type
            if isinstance(expected_type, ir.IntType) and isinstance(llvm_arg.type, ir.IntType):
                # For integer types, perform bit casting if needed
                if expected_type.width > llvm_arg.type.width:
                    llvm_arg = builder.zext(llvm_arg, expected_type)
                elif expected_type.width < llvm_arg.type.width:
                    llvm_arg = builder.trunc(llvm_arg, expected_type)
            # Add more type casting as needed
        
        llvm_args.append(llvm_arg)
    
    # Before calling, let's log what we're about to do for debugging
    if self.compiler.debug:
        print(f"Calling function {func_name} with {len(llvm_args)} arguments")
        for i, arg in enumerate(llvm_args):
            print(f"  Argument {i}: {arg}, Type: {arg.type}")
    
    # Make the function call in LLVM IR
    if func.function_type.return_type == ir.VoidType():
        # For void functions
        builder.call(func, llvm_args)
        return None
    else:
        # For functions that return a value
        result = builder.call(func, llvm_args)
        return result