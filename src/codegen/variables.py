from llvmlite import ir, binding
from typing import TYPE_CHECKING, Dict, Callable, Type
from astnodes import *
from lexer import *
from .symboltable import create_variable_symbol


# Utility functions for pointer level handling
def count_pointer_level(type_str: str) -> tuple[str, int]:
    """
    Count the pointer level from a type string.
    Returns (base_type, pointer_level)
    
    Examples:
    - "int" -> ("int", 0)
    - "int*" -> ("int", 1)
    - "char**" -> ("char", 2)
    - "MyStruct***" -> ("MyStruct", 3)
    """
    base_type = type_str.rstrip('*')
    pointer_level = len(type_str) - len(base_type)
    return base_type, pointer_level


def apply_pointer_level(base_llvm_type: Any, pointer_level: int) -> Any:
    """
    Apply pointer level to an LLVM base type.
    
    Args:
        base_llvm_type: Base LLVM type (e.g., ir.IntType(32))
        pointer_level: Number of pointer levels (0 = no pointer, 1 = pointer, etc.)
    
    Returns:
        LLVM type with appropriate pointer levels applied
    """
    result_type = base_llvm_type
    for _ in range(pointer_level):
        result_type = ir.PointerType(result_type)
    return result_type
def handle_variable_declaration(self, node: ASTNode.VariableDeclaration, builder: ir.IRBuilder, **kwargs):
    """Handle variable declaration with multilevel pointer support."""
    
    # Parse the type string to determine base type and pointer level
    base_type_name, pointer_level = count_pointer_level(node.var_type)
    
    # Also check the node's pointer-related attributes if they exist
    if hasattr(node, 'is_pointer') and node.is_pointer and pointer_level == 0:
        pointer_level = 1  # Backward compatibility
    elif hasattr(node, 'pointer_level'):
        pointer_level = max(pointer_level, node.pointer_level)
    
    if self.compiler.debug:
        print(f"Variable declaration: {node.name}, base_type: {base_type_name}, pointer_level: {pointer_level}")
    
    # Get the base LLVM type
    base_type = Datatypes.to_llvm_type(base_type_name)
    if base_type is None:
        raise ValueError(f"Unknown type: {base_type_name}")
    
    # Apply pointer levels to get the final variable type
    var_type = apply_pointer_level(base_type, pointer_level)
        
    
    # Allocate space for the variable (this creates a pointer to var_type)
    var = builder.alloca(var_type, name=node.name)
    
    # Create and store the symbol in our table
    symbol = create_variable_symbol(
        name=node.name,
        ast_node=node,
        data_type=node.var_type,  # Keep original type string
        llvm_type=var_type,
        llvm_value=var,
        pointer_level=pointer_level
    )
    self.symbol_table.define(symbol)
    
    # Handle initial value if present
    if node.value:
        if self.compiler.debug:
            print(f"Processing initial value for {node.name}, pointer_level: {pointer_level}")
        
        # Pass pointer level information to expression handler
        value = self.handle_expression(node.value, builder, base_type, pointer_level=pointer_level)
            
        if not value:
            if pointer_level > 0:
                # Initialize to null pointer
                value = ir.Constant(var_type, None)
            else:
                # Default to zero for non-pointer types
                value = ir.Constant(var_type, 0)
        else:
            # Apply proper type casting before storing
            value = self._cast_value(value, var_type, builder)
                
        builder.store(value, var)
    else:
        # Initialize with default values
        if pointer_level > 0:
            # Initialize pointers to null by default
            null_ptr = ir.Constant(var_type, None)
            builder.store(null_ptr, var)
        # Non-pointer types can be left uninitialized or initialized to zero
        # depending on language semantics

    
    return var  # Return the variable pointer
def handle_variable_assignment(self, node: ASTNode.VariableAssignment, builder: ir.IRBuilder, **kwargs):
    """Handle variable assignment with the new symbol table."""
    # Get variable name and check if it's a struct field access
    var_name = node.name
    
    # Check if this is a struct field assignment (contains a dot)
    if '.' in var_name:
        struct_name, field_name = var_name.split('.')
        
        # Look up the struct in the symbol table
        struct_symbol = self.symbol_table.lookup(struct_name)
        if not struct_symbol:
            raise ValueError(f"Struct variable '{struct_name}' not found in symbol table.")
        
        # Get struct information
        struct_ptr = struct_symbol.llvm_value
        struct_type_name = struct_symbol.data_type
        
        # Ensure the struct type exists in the struct table
        if struct_type_name not in self.struct_table:
            raise ValueError(f"Struct type '{struct_type_name}' not found in struct table.")
        
        # Get the struct type definition
        struct_type_info = self.struct_table[struct_type_name]["class_type_info"]
        
        # Find the field index in the struct
        if field_name not in struct_type_info.field_names:
            raise ValueError(f"Field '{field_name}' not found in struct '{struct_type_name}'.")
        
        # Get pointer to the field
        field_ptr = self.get_struct_field_ptr(struct_name, field_name, builder)
        
        # Get the field type from the struct type
        field_index = struct_type_info.field_names.index(field_name)
        field_type = struct_type_info.llvm_type.elements[field_index]
        
        # Evaluate right-hand side expression
        value = self.handle_expression(node.value, builder, field_type)
        
        # Handle type casting if needed
        if value.type != field_type:
            value = self._cast_value(value, field_type, builder)
        
        # Store the value in the field
        builder.store(value, field_ptr)
    else:
        # Regular variable assignment
        symbol = self.symbol_table.lookup(var_name)
        if not symbol:
            raise ValueError(f"Variable '{var_name}' not found in symbol table. It must be declared before assignment.")
        
        # Get variable pointer and type
        var_ptr = symbol.llvm_value
        var_type = var_ptr.type.pointee
        
        # Evaluate right-hand side expression
        value = self.handle_expression(node.value, builder, var_type)
        
        # Handle type casting if types don't match
        if value.type != var_type:
            value = self._cast_value(value, var_type, builder)
        
        # Store the evaluated value into the variable
        builder.store(value, var_ptr)

def handle_variable_increment(self, node, builder, **kwargs):
    # Find variable in symbol table
    self.symbol_table.lookup(node.name)
    # Get the variable pointer
    var_ptr = self.get_variable_pointer(node.name)
    # Load the current value
    current_value = builder.load(var_ptr, name=f"load_{node.name}")
    # Increment the value
    incremented_value = builder.add(current_value, ir.Constant(current_value.type, 1), name=f"increment_{node.name}")
    # Store the incremented value back to the variable
    builder.store(incremented_value, var_ptr)
    # Return the incremented value
    return incremented_value

def handle_variable_decrement(self, node, builder, **kwargs):
    # Find variable in symbol table
    self.symbol_table.lookup(node.name)
    # Get the variable pointer
    var_ptr = self.get_variable_pointer(node.name)
    # Load the current value
    current_value = builder.load(var_ptr, name=f"load_{node.name}")
    # Decrement the value
    decremented_value = builder.sub(current_value, ir.Constant(current_value.type, 1), name=f"decrement_{node.name}")
    # Store the decremented value back to the variable
    builder.store(decremented_value, var_ptr)
    # Return the decremented value
    return decremented_value

def handle_compound_variable_declaration(self, node: ASTNode.CompoundVariableDeclaration, builder: ir.IRBuilder, **kwargs):
    """Handle compound variable declarations (like structs or unions)."""
    # Get the type of the compound variable
    for declaration in node.declarations:
        self.handle_variable_declaration(declaration, builder)
    
def handle_compound_variable_assignment(self, node: ASTNode.CompoundVariableAssigment, builder: ir.IRBuilder, **kwargs):
    for assignment in node.assignments:
        self.handle_variable_assignment(assignment, builder)
    