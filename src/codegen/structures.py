from llvmlite import ir, binding
from typing import TYPE_CHECKING, Dict, Callable, Type
from astnodes import *
from lexer import *
from enum import Enum 

class EnumValueType(Enum):
    """Enum to represent the type of enum values"""
    INTEGER = "integer"
    STRING = "string"

class EnumTypeInfo:
    """Enhanced enum type information that handles both integer and string enums"""
    
    def __init__(self, name, enum_value_type, llvm_type, values, ast_node):
        self.name = name                    # Name of the enum
        self.enum_value_type = enum_value_type  # EnumValueType.INTEGER or EnumValueType.STRING
        self.llvm_type = llvm_type         # The underlying LLVM type (i32 for int, i8* for string)
        self.values = values               # Dictionary mapping member names to their values
        self.ast_node = ast_node           # Reference to the original AST node
        
        # For string enums, we also need to store the LLVM constants
        self.llvm_constants = {}           # Maps member names to LLVM constants
    
    def get_member_value(self, member_name):
        """Get the value for a specific enum member"""
        return self.values.get(member_name)
    
    def get_llvm_constant(self, member_name):
        """Get the LLVM constant for a specific enum member"""
        if self.enum_value_type == EnumValueType.STRING:
            return self.llvm_constants.get(member_name)
        else:
            # For integer enums, create the constant on-the-fly
            value = self.values.get(member_name)
            if value is not None:
                return ir.Constant(self.llvm_type, value)
        return None
    
    def get_llvm_type(self):
        """Return the LLVM type for this enum"""
        return self.llvm_type
    
    def has_member(self, member_name):
        """Check if a member exists in this enum"""
        return member_name in self.values
    
    def is_string_enum(self):
        """Check if this is a string enum"""
        return self.enum_value_type == EnumValueType.STRING
    
    def is_integer_enum(self):
        """Check if this is an integer enum"""
        return self.enum_value_type == EnumValueType.INTEGER

class EnumTable:
    """Centralized enum table to manage all enum types"""
    
    def __init__(self):
        self.enums = {}  # Maps enum names to EnumTypeInfo objects
    
    def add_enum(self, name, enum_type_info):
        """Add an enum to the table"""
        self.enums[name] = enum_type_info
    
    def get_enum(self, name):
        """Get enum information by name"""
        return self.enums.get(name)
    
    def has_enum(self, name):
        """Check if an enum exists"""
        return name in self.enums
    
    def remove_enum(self, name):
        """Remove an enum from the table"""
        if name in self.enums:
            del self.enums[name]
    
    def clear(self):
        """Clear all enums"""
        self.enums.clear()

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

            def get_field_ptr(self, struct_ptr, field_name: str, builder: ir.IRBuilder):
                """
                Returns an LLVM pointer to the field inside the struct.
                
                struct_ptr: %struct.ClassName* (LLVM pointer to struct)
                class_info: ClassTypeInfo instance for the class
                field_name: str, name of the field
                builder: IRBuilder to emit instructions
                """
                # Get the field index
                field_index = self.get_field_index(field_name)
                
                # LLVM GEP: first 0 for the struct itself, then field index
                ptr = builder.gep(
                    struct_ptr,
                    [ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), field_index)],
                    inbounds=True
                )
                
                return ptr

            def __repr__(self):
                return f"<ClassTypeInfo: fields={self.field_names}, parent={self.parent}, llvm_type={self.llvm_type}>"

def handle_class(self, node: ASTNode.Class, **kwargs):
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
            field_llvm_type = Datatypes.to_llvm_type(field.var_type, field.pointer_level)

        field_llvm_types.append(field_llvm_type)
        field_names.append(field.name)

    # 3. Set the body of the identified struct type.
    # This defines the actual layout (the sequence of field types) for the struct.
    # This step completes the definition of the 'opaque' type created earlier.
    struct_type.set_body(*field_llvm_types) # Use * to unpack the list of types

    # --- END: Handling Identified LLVM Struct Types ---

    # Create a wrapper object to store additional information about our class,
    # including the now-defined LLVM struct type.
    class_type_info = ClassTypeInfo(struct_type, field_names, parent_type_info, node)

    # Register the class type info in your type system.
    # This makes the 'Node' source type name map to the 'class_type_info' object,
    # which contains the LLVM 'struct_type'.
    Datatypes.add_type(node.name, class_type_info)

    # Store the struct info in your internal table (if needed).
    # Ensure you are storing the class name as the key.
    self.struct_table[node.name] = {'name': node.name, 'class_type_info': class_type_info}

    # --- NEW: Process class methods ---
    # Register each method as a function with the naming convention ClassName_methodName
    for method in node.methods:
        # Create the mangled method name
        mangled_name = f"{node.name}_{method.name}"
        
        # Create a new function definition node with the mangled name
        # The first parameter should be a pointer to the class instance (self)
        
        # Create self parameter
        self_param = ASTNode.VariableDeclaration(
            var_type=node.name,
            name="self",
            pointer_level=1,  # self is always a pointer to the class
            is_user_typed=True
        )
        

        # Combine self parameter with existing parameters
        if not method.parameters or method.parameters[0].name != "self":
            all_params = [self_param] + method.parameters
        else:
            all_params = method.parameters
        
        # Create a new function definition with the mangled name and updated parameters
        mangled_method = ASTNode.FunctionDefinition(
            name=mangled_name,
            return_type=method.return_type,
            parameters=all_params,
            body=method.body
        )
        # Process the method as a regular function using the existing handler
        from .functions import handle_function_definition
        handle_function_definition(self, mangled_method, **kwargs)
        
        if self.compiler.debug:
            print(f"Registered method {method.name} as function {mangled_name}")

    # This function typically doesn't return an LLVM value, just defines the type.
    return None

def handle_class_method_call(self, node, builder: ir.IRBuilder, **kwargs):
    """Handle class method calls - automatically injects self parameter."""
    # Extract components
    object_name = node.object_name
    method_name = node.method_name
    
    provided_args = node.args if node.args else []
    
    # Look up object
    object_symbol = self.symbol_table.lookup(object_name)
    if not object_symbol:
        raise ValueError(f"Object '{object_name}' not found in symbol table")
    
    # Create actual function name (however you're naming methods)
    actual_function_name = f"{object_symbol.data_type}_{method_name}"
    
    # Verify method exists
    func_symbol = self.symbol_table.lookup(actual_function_name)
    if not func_symbol:
        raise ValueError(f"Method '{actual_function_name}' not found")
    
    # Check parameter count
    func = func_symbol.llvm_value
    expected_total_args = len(func.function_type.args)  # Total expected (including self)
    provided_user_args = len(provided_args)             # User-provided args (excluding self)
    
    # The method expects: self + user_provided_args
    # So expected_total_args should equal 1 + provided_user_args
    expected_user_args = expected_total_args - 1  # Subtract 1 for self
    
    if provided_user_args != expected_user_args:
        raise ValueError(f"Method '{method_name}' expects {expected_user_args} arguments "
                        f"(plus self), but {provided_user_args} were provided")
    
    # Create self reference node
    self_ref = ASTNode.ExpressionNode(NodeType.REFERENCE, left=ASTNode.ExpressionNode(node_type=NodeType.LITERAL, value=object_name))

    
    # Create modified function call node with self as first argument
    modified_call = ASTNode.FunctionCall(
        name=actual_function_name,
        arguments=[self_ref] + provided_args,  # self + user args
        has_parentheses=True
    )
    
    # Use existing function call handler
    from .functions import handle_function_call
    return handle_function_call(self, modified_call, builder, **kwargs)

def _determine_enum_type(self, members):
    """
    Determine if this is a string enum or integer enum based on the first member
    
    Args:
        members: List of (member_name, value_expr) tuples
        
    Returns:
        EnumValueType indicating the type of enum
    """
    if not members:
        return EnumValueType.INTEGER  # Default to integer for empty enums
    
    # Check the first member that has a value
    for member_name, value_expr in members:
        if value_expr is not None:
            if value_expr.node_type == NodeType.LITERAL:
                # Check if the literal value is a string (contains quotes)
                if isinstance(value_expr.value, str):
                    # Check if it's a quoted string
                    if (value_expr.value.startswith('"') and value_expr.value.endswith('"')) or \
                    (value_expr.value.startswith("'") and value_expr.value.endswith("'")):
                        return EnumValueType.STRING
                    else:
                        # Try to parse as integer
                        try:
                            int(value_expr.value)
                            return EnumValueType.INTEGER
                        except ValueError:
                            # If it's not a valid integer, treat as string
                            return EnumValueType.STRING
    
    # Default to integer enum
    return EnumValueType.INTEGER

def handle_enum(self, node: ASTNode.Enum, **kwargs):
    """
    Enhanced enum handler that supports both integer and string enums
    """
    debug = getattr(self.compiler, 'debug', False)
    
    if debug:
        print(f"Processing enum: {node.name}")
    
    # Initialize enum table if it doesn't exist
    if not hasattr(self, 'enum_table'):
        self.enum_table = EnumTable()
    
    # Determine the type of enum (integer or string)
    enum_type = self._determine_enum_type(node.members)
    
    if debug:
        print(f"Enum type determined: {enum_type.value}")
    
    # Set up the underlying LLVM type
    if enum_type == EnumValueType.STRING:
        underlying_type = ir.PointerType(ir.IntType(8))  # i8* for strings
    else:
        underlying_type = ir.IntType(32)  # i32 for integers
    
    # Process enum members
    enum_values = {}
    llvm_constants = {}
    current_value = 0  # For auto-incrementing integer enums
    
    for member_name, value_expr in node.members:
        if debug:
            print(f"Processing member: {member_name}")
        
        if enum_type == EnumValueType.STRING:
            if value_expr is not None and value_expr.node_type == NodeType.LITERAL:
                string_value = value_expr.value
                enum_values[member_name] = string_value
                
                # Create the LLVM string constant
                string_constant = self._create_string_constant(self.builder, string_value)
                llvm_constants[member_name] = string_constant
                
                if debug:
                    print(f"String enum member {member_name} = {string_value}")
            else:
                raise ValueError(f"String enum member '{member_name}' must have a string literal value")
        
        else:  # Integer enum
            if value_expr is not None:
                # Evaluate the constant expression
                evaluated_value = self._evaluate_enum_constant(value_expr)
                if evaluated_value is not None:
                    current_value = evaluated_value
            
            enum_values[member_name] = current_value
            if debug:
                print(f"Integer enum member {member_name} = {current_value}")
            
            current_value += 1  # Auto-increment for next member
    
    # Create the EnumTypeInfo object
    enum_type_info = EnumTypeInfo(
        name=node.name,
        enum_value_type=enum_type,
        llvm_type=underlying_type,
        values=enum_values,
        ast_node=node
    )
    
    # For string enums, store the LLVM constants
    if enum_type == EnumValueType.STRING:
        enum_type_info.llvm_constants = llvm_constants
    
    # Register the enum in the type system
    Datatypes.add_type(node.name, enum_type_info)
    
    # Add to our enum table
    self.enum_table.add_enum(node.name, enum_type_info)
    
    if debug:
        print(f"Enum {node.name} registered successfully")
    
    return None
def handle_union(self, node, **kwargs):
    pass