from .symboltable import create_function_symbol
from llvmlite import ir 
from astnodes import *
from lexer import *

def handle_extern(self, node: ASTNode.Extern, **kwargs):
    """Handle extern function declarations."""
    # Get function name, return type and parameter information
    func_name = node.declaration.name
    return_type = self.get_llvm_type(node.declaration.return_type)
    if return_type is None:
        raise ValueError(f"Unknown return type for function '{func_name}'")
    
    # Process parameters
    param_types = []
    for param in node.declaration.parameters:
        param_type = self.get_llvm_type(param.var_type)
        # Handle pointer types
        if param.pointer_level > 0:
            param_type = ir.PointerType(param_type)
        param_types.append(param_type)
    
    # Create function type
    func_type = ir.FunctionType(return_type, param_types)
    
    # Check if function already exists in the module
    if func_name in self.module.globals:
        # Function already declared, return the existing function
        return self.module.globals[func_name]
    
    # Create the function declaration
    func = ir.Function(self.module, func_type, name=func_name)
    
    # Set parameter names (optional but useful for debugging)
    for i, param in enumerate(node.declaration.parameters):
        func.args[i].name = param.name
    
    # Register the function in the symbol table
    symbol = create_function_symbol(
        func_name, 
        node.declaration, 
        node.declaration.return_type, 
        [param.var_type for param in node.declaration.parameters],
        func, 
        0  # Global scope
    )
    self.symbol_table.define(symbol)
    
    # Return the function reference
    return func

