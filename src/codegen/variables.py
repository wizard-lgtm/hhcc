from llvmlite import ir, binding
from typing import TYPE_CHECKING, Dict, Callable, Type
from astnodes import *
from lexer import *
from .symboltable import create_array_symbol, create_variable_symbol

if TYPE_CHECKING:
    from .base import Codegen

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
    
def handle_array_element_assignment(self: "Codegen", node: ASTNode.ArrayElementAssignment, builder: ir.IRBuilder, **kwargs):
    """Handle assignment to an array element."""
    # Look up the array variable in the symbol table
    array_symbol = self.symbol_table.lookup(node.array_name)
    
    if not array_symbol:
        raise ValueError(f"Array variable '{node.array_name}' not found in symbol table.")
    
    print(f"Array symbol: {array_symbol}")
    
    # Get the pointer to the array
    array_ptr = array_symbol.llvm_value
    
    # Determine the element type based on the array/pointer type
    if isinstance(array_ptr.type.pointee, ir.ArrayType):
        # This is an actual array type [N x element_type]
        element_type = array_ptr.type.pointee.element
        # Get pointer to the specific array element
        index_value = self.handle_expression(node.index_expr, builder, ir.IntType(64))
        element_ptr = builder.gep(array_ptr, [ir.Constant(ir.IntType(32), 0), index_value], name=f"{node.array_name}_elem_ptr")
    elif isinstance(array_ptr.type.pointee, ir.IntType):
        # This is a pointer to elements (like U8* pointing to a string)
        element_type = array_ptr.type.pointee
        # Load the pointer first, then index into it
        loaded_ptr = builder.load(array_ptr, name=f"load_{node.array_name}")
        index_value = self.handle_expression(node.index_expr, builder, ir.IntType(64))
        element_ptr = builder.gep(loaded_ptr, [index_value], name=f"{node.array_name}_elem_ptr")
    elif isinstance(array_ptr.type.pointee, ir.PointerType):
        # This is a pointer to pointer (like U8* stored in a variable)
        # The element type is what the inner pointer points to
        element_type = array_ptr.type.pointee.pointee
        # Load the pointer first, then index into it
        loaded_ptr = builder.load(array_ptr, name=f"load_{node.array_name}")
        index_value = self.handle_expression(node.index_expr, builder, ir.IntType(64))
        element_ptr = builder.gep(loaded_ptr, [index_value], name=f"{node.array_name}_elem_ptr")
    else:
        raise ValueError(f"Unsupported array/pointer type: {array_ptr.type.pointee}")
    
    print(f"Element type: {element_type}")
    
    # Evaluate the right-hand side expression with the correct element type
    value = self.handle_expression(node.value_expr, builder, element_type)
    
    print(f"Value type: {value.type}, Element type: {element_type}")
    
    # Handle type casting if needed
    if value.type != element_type:
        value = self._cast_value(value, element_type, builder)
    
    # Store the value in the array element
    builder.store(value, element_ptr)
    
    return element_ptr  # Return pointer to the assigned element

def handle_array_declaration(self: "Codegen", node: ASTNode.ArrayDeclaration, builder: ir.IRBuilder, **kwargs):
    """
    Handle array declaration with multi-dimensional support.
    
    Creates LLVM array types and allocates memory for arrays.
    Supports both global and local scope allocation.
    """
    if self.compiler.debug:
        print(f"Handling array declaration: {node.name}, base_type: {node.base_type}")
    
    # Step 1: Get the base LLVM type
    base_llvm_type = Datatypes.to_llvm_type(node.base_type)
    if base_llvm_type is None:
        raise ValueError(f"Unknown base type: {node.base_type}")
    
    # Step 2: Evaluate dimensions to get concrete sizes
    dimension_sizes = []
    has_undefined_dimensions = False
    
    for dim_expr in node.dimensions:
        if dim_expr is None:
            # Mark that we have undefined dimensions - we'll infer from initialization
            has_undefined_dimensions = True
            dimension_sizes.append(None)  # Placeholder
        else:
            # Evaluate the dimension expression based on ExpressionNode structure
            if dim_expr.node_type == NodeType.LITERAL:
                # Direct integer literal
                size = int(dim_expr.value)
            else:
                # Complex expression - evaluate it
                size_value = self.handle_expression(dim_expr, builder, ir.IntType(64))
                if isinstance(size_value, ir.Constant):
                    size = int(size_value.constant)
                else:
                    raise ValueError(f"Array dimension must be a compile-time constant, got runtime expression")
            
            if size <= 0:
                raise ValueError(f"Array dimension must be positive, got {size}")
            
            dimension_sizes.append(size)
    
    # Step 2.5: Handle size inference from initialization
    if has_undefined_dimensions:
        if not node.initialization:
            raise ValueError(f"Array '{node.name}' has undefined dimensions but no initialization to infer size from")
        
        # Infer size from initialization
        dimension_sizes = self._infer_array_dimensions(node.initialization, dimension_sizes)
        
        if self.compiler.debug:
            print(f"Inferred array dimensions: {dimension_sizes}")
    
    if self.compiler.debug:
        print(f"Array dimensions: {dimension_sizes}")
    
    # Step 3: Create nested LLVM array type for multi-dimensional arrays
    # Build from innermost to outermost: [cols x [rows x element_type]]
    llvm_array_type = base_llvm_type
    for size in reversed(dimension_sizes):
        llvm_array_type = ir.ArrayType(llvm_array_type, size)
    
    if self.compiler.debug:
        print(f"Final LLVM array type: {llvm_array_type}")
    
    # Step 4: Allocate memory based on scope level
    array_ptr = None
    if self.symbol_table.current_scope_level == 0:
        # Global scope - create global variable
        global_array = ir.GlobalVariable(self.module, llvm_array_type, name=node.name)
        global_array.linkage = 'internal'
        
        # Initialize global array
        if node.initialization:
            # Handle initialization (we'll implement this part next)
            init_value = self._create_array_initializer(node.initialization, llvm_array_type, dimension_sizes)
            global_array.initializer = init_value
        else:
            # Zero-initialize by default
            global_array.initializer = ir.Constant(llvm_array_type, None)  # Zero initializer
        
        array_ptr = global_array
    else:
        # Local scope - allocate on stack
        array_ptr = builder.alloca(llvm_array_type, name=node.name)
        
        # Handle initialization for local arrays
        if node.initialization:
            self._initialize_local_array(array_ptr, node.initialization, llvm_array_type, dimension_sizes, builder)
        # Note: Local arrays can be left uninitialized or we could zero them out
        # depending on language semantics
    
    # Step 5: Create array symbol and add to symbol table
    array_symbol = create_array_symbol(
        name=node.name,
        ast_node=node,
        element_type=node.base_type,  # Base element type
        dimensions=dimension_sizes,   # Concrete dimension sizes
        llvm_type=llvm_array_type,   # The full array type
        llvm_value=array_ptr,        # Pointer to the array
        scope_level=self.symbol_table.current_scope_level
    )
    
    # Add to symbol table
    self.symbol_table.define(array_symbol)
    
    if self.compiler.debug:
        print(f"Array symbol created: {array_symbol}")
    
    return array_ptr


def _create_array_initializer(self: "Codegen", initialization: ASTNode.ArrayInitialization, 
                             array_type: ir.ArrayType, dimensions: List[int]) -> ir.Constant:
    """
    Create LLVM constant initializer for global arrays.
    
    Args:
        initialization: The ArrayInitialization node
        array_type: The LLVM array type
        dimensions: List of dimension sizes
    
    Returns:
        LLVM constant for initialization
    """
    if not initialization or not initialization.values:
        # Return zero initializer
        return ir.Constant(array_type, None)
    
    # Handle 1D arrays
    if len(dimensions) == 1:
        element_type = array_type.element
        init_values = []
        
        for i, value_expr in enumerate(initialization.elements):
            if i >= dimensions[0]:
                break  # Don't exceed array bounds
            
            # Handle both regular elements and string literals
            if (len(initialization.elements) == 1 and 
                initialization.elements[0].node_type == NodeType.LITERAL and
                isinstance(initialization.elements[0].value, str)):
                # This is a string literal initialization
                string_val = initialization.elements[0].value
                # Convert string to individual character constants
                for j, char in enumerate(string_val):
                    if j >= dimensions[0] - 1:  # Leave space for null terminator
                        break
                    const_val = ir.Constant(element_type, ord(char))
                    init_values.append(const_val)
                # Add null terminator
                if len(init_values) < dimensions[0]:
                    init_values.append(ir.Constant(element_type, 0))
                break  # We've processed the entire string
            else:
                # Regular element initialization
                if value_expr.node_type == NodeType.LITERAL:
                    if element_type == ir.IntType(8):  # U8
                        const_val = ir.Constant(element_type, int(value_expr.value))
                    elif element_type == ir.IntType(32):  # U32/I32
                        const_val = ir.Constant(element_type, int(value_expr.value))
                    elif element_type == ir.IntType(64):  # U64/I64
                        const_val = ir.Constant(element_type, int(value_expr.value))
                    elif element_type == ir.FloatType():  # F32
                        const_val = ir.Constant(element_type, float(value_expr.value))
                    elif element_type == ir.DoubleType():  # F64
                        const_val = ir.Constant(element_type, float(value_expr.value))
                    else:
                        raise ValueError(f"Unsupported element type for array initialization: {element_type}")
                else:
                    # For complex expressions, we might need to evaluate them at compile time
                    # This is more complex and might require constant folding
                    raise ValueError("Complex expressions in global array initialization not yet supported")
                
                init_values.append(const_val)
        
        # Pad with zeros if not enough values provided
        zero_val = ir.Constant(element_type, 0)
        while len(init_values) < dimensions[0]:
            init_values.append(zero_val)
        
        return ir.Constant(array_type, init_values)
    
    else:
        # Multi-dimensional arrays - more complex, would need recursive handling
        # For now, return zero initializer
        return ir.Constant(array_type, None)


def _initialize_local_array(self: "Codegen", array_ptr: ir.Value, 
                           initialization: ASTNode.ArrayInitialization,
                           array_type: ir.ArrayType, dimensions: List[int], 
                           builder: ir.IRBuilder) -> None:
    """Initialize a local array with runtime values."""
    
    if not initialization or not initialization.elements:
        return  # Leave uninitialized or zero-fill if desired
    
    # Handle 1D arrays
    if len(dimensions) == 1:
        element_type = array_type.element
        
        # Check if this is a single string literal initialization
        first_elem = initialization.elements[0]
        is_string_literal = (
            isinstance(first_elem, str) or  # Direct string case
            (hasattr(first_elem, 'node_type') and 
             first_elem.node_type == NodeType.LITERAL and
             isinstance(first_elem.value, str))  # AST node case
        )
        
        if len(initialization.elements) == 1 and is_string_literal:
            
            # Handle string literal initialization
            if isinstance(first_elem, str):
                string_val = first_elem
            else:
                string_val = first_elem.value
            
            if self.compiler.debug:
                print(f"Initializing array with string literal: '{string_val}'")
            
            # Store each character individually
            for i, char in enumerate(string_val):
                if i >= dimensions[0] - 1:  # Leave space for null terminator
                    break
                
                # Get pointer to array element
                index_const = ir.Constant(ir.IntType(32), i)
                element_ptr = builder.gep(array_ptr, [ir.Constant(ir.IntType(32), 0), index_const], 
                                          name=f"{array_ptr.name}_elem_{i}")
                
                # Store the character
                char_value = ir.Constant(element_type, ord(char))
                builder.store(char_value, element_ptr)
            
            # Add null terminator if there's space
            if len(string_val) < dimensions[0]:
                null_index = ir.Constant(ir.IntType(32), len(string_val))
                null_ptr = builder.gep(array_ptr, [ir.Constant(ir.IntType(32), 0), null_index], 
                                       name=f"{array_ptr.name}_null")
                null_char = ir.Constant(element_type, 0)
                builder.store(null_char, null_ptr)
        
        else:
            # Handle regular element-by-element initialization
            for i, elem in enumerate(initialization.elements):
                if i >= dimensions[0]:
                    break  # Avoid overflow
                
                # Get pointer to array element
                index_const = ir.Constant(ir.IntType(32), i)
                element_ptr = builder.gep(array_ptr, [ir.Constant(ir.IntType(32), 0), index_const], 
                                          name=f"{array_ptr.name}_elem_{i}")
                
                # Evaluate the element expression
                if isinstance(elem, ASTNode.ExpressionNode):
                    value = self.handle_expression(elem, builder, element_type)
                    if value.type != element_type:
                        value = self._cast_value(value, element_type, builder)
                    builder.store(value, element_ptr)
                else:
                    raise TypeError(f"Unsupported array element type: {type(elem)}")
    
    else:
        if self.compiler.debug:
            print("Multi-dimensional array initialization not yet implemented")


def _create_array_initializer(self: "Codegen", initialization: ASTNode.ArrayInitialization, 
                             array_type: ir.ArrayType, dimensions: List[int]) -> ir.Constant:
    """
    Create LLVM constant initializer for global arrays.
    
    Args:
        initialization: The ArrayInitialization node
        array_type: The LLVM array type
        dimensions: List of dimension sizes
    
    Returns:
        LLVM constant for initialization
    """
    if not initialization or not initialization.elements:
        # Return zero initializer
        return ir.Constant(array_type, None)
    
    # Handle 1D arrays
    if len(dimensions) == 1:
        element_type = array_type.element
        init_values = []
        
        # Check if this is a single string literal initialization
        first_elem = initialization.elements[0]
        is_string_literal = (
            isinstance(first_elem, str) or  # Direct string case
            (hasattr(first_elem, 'node_type') and 
             first_elem.node_type == NodeType.LITERAL and
             isinstance(first_elem.value, str))  # AST node case
        )
        
        if len(initialization.elements) == 1 and is_string_literal:
            
            # Handle string literal initialization
            if isinstance(first_elem, str):
                string_val = first_elem
            else:
                string_val = first_elem.value
            
            if self.compiler.debug:
                print(f"Creating global string initializer for: '{string_val}'")
            
            # Convert string to individual character constants
            for i, char in enumerate(string_val):
                if i >= dimensions[0] - 1:  # Leave space for null terminator
                    break
                const_val = ir.Constant(element_type, ord(char))
                init_values.append(const_val)
            
            # Add null terminator if there's space
            if len(init_values) < dimensions[0]:
                init_values.append(ir.Constant(element_type, 0))
            
            # Pad with zeros if needed
            zero_val = ir.Constant(element_type, 0)
            while len(init_values) < dimensions[0]:
                init_values.append(zero_val)
        
        else:
            # Handle regular element initialization
            for i, value_expr in enumerate(initialization.elements):
                if i >= dimensions[0]:
                    break  # Don't exceed array bounds
                
                # Regular element initialization
                if value_expr.node_type == NodeType.LITERAL:
                    if element_type == ir.IntType(8):  # U8
                        const_val = ir.Constant(element_type, int(value_expr.value))
                    elif element_type == ir.IntType(32):  # U32/I32
                        const_val = ir.Constant(element_type, int(value_expr.value))
                    elif element_type == ir.IntType(64):  # U64/I64
                        const_val = ir.Constant(element_type, int(value_expr.value))
                    elif element_type == ir.FloatType():  # F32
                        const_val = ir.Constant(element_type, float(value_expr.value))
                    elif element_type == ir.DoubleType():  # F64
                        const_val = ir.Constant(element_type, float(value_expr.value))
                    else:
                        raise ValueError(f"Unsupported element type for array initialization: {element_type}")
                else:
                    # For complex expressions, we might need to evaluate them at compile time
                    # This is more complex and might require constant folding
                    raise ValueError("Complex expressions in global array initialization not yet supported")
                
                init_values.append(const_val)
            
            # Pad with zeros if not enough values provided
            zero_val = ir.Constant(element_type, 0)
            while len(init_values) < dimensions[0]:
                init_values.append(zero_val)
        
        return ir.Constant(array_type, init_values)
    
    else:
        # Multi-dimensional arrays - more complex, would need recursive handling
        # For now, return zero initializer
        return ir.Constant(array_type, None)


def _infer_array_dimensions(self: "Codegen", initialization: ASTNode.ArrayInitialization, 
                           dimension_template: List[Optional[int]]) -> List[int]:
    """
    Infer array dimensions from initialization data.
    
    Args:
        initialization: The ArrayInitialization node
        dimension_template: List with None for dimensions to infer, concrete values for known dims
    
    Returns:
        List of concrete dimension sizes
    """
    if not initialization:
        raise ValueError("Cannot infer array dimensions without initialization")
    
    result_dimensions = []
    
    # Handle the first (and possibly only) dimension
    if len(dimension_template) >= 1 and dimension_template[0] is None:
        # Infer first dimension from initialization elements
        if hasattr(initialization, 'elements') and initialization.elements:
            first_elem = initialization.elements[0]

            # Check if this is a string literal - handle both AST node and direct string cases
            if isinstance(first_elem, str):
                # Direct string case
                string_value = first_elem
                inferred_size = len(string_value) + 1
                if self.compiler.debug:
                    print(f"Inferred string array size: {inferred_size} for '{string_value}'")
            
            elif (hasattr(first_elem, 'node_type') and 
                  first_elem.node_type == NodeType.LITERAL and
                  isinstance(first_elem.value, str)):
                
                # AST node with string literal
                string_value = first_elem.value
                inferred_size = len(string_value) + 1
                if self.compiler.debug:
                    print(f"Inferred string array size: {inferred_size} for '{string_value}'")
            
            else:
                # Regular array like {1, 2, 3}
                inferred_size = len(initialization.elements)
                if self.compiler.debug:
                    print(f"Inferred array size from {inferred_size} elements")

        else:
            raise ValueError("Cannot infer array size from initialization - no elements found")
        
        result_dimensions.append(inferred_size)
    else:
        # Use the provided dimension
        result_dimensions.append(dimension_template[0])
    
    # For multi-dimensional arrays, add the rest of the dimensions
    for i in range(1, len(dimension_template)):
        if dimension_template[i] is None:
            # For now, only support inferring the first dimension
            raise ValueError("Can only infer the first dimension of multi-dimensional arrays")
        result_dimensions.append(dimension_template[i])
    
    return result_dimensions

# Helper function to calculate total array size
def _calculate_total_array_size(dimensions: List[int]) -> int:
    """Calculate total number of elements in a multi-dimensional array."""
    total = 1
    for dim in dimensions:
        total *= dim
    return total


# Example usage in your AST visitor pattern:
def visit_array_declaration(self, node: ASTNode.ArrayDeclaration, builder: ir.IRBuilder):
    """Visitor method for array declarations."""
    return self.handle_array_declaration(node, builder)