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

class ClassFieldInfo:
    """Enhanced field information for class fields"""
    
    def __init__(self, name: str, field_type: str, llvm_type, ast_node: 'ASTNode.VariableDeclaration', 
                 index: int, is_mutable: bool = True, default_value: Any = None, 
                 default_llvm_value = None, pointer_level: int = 0):
        self.name = name                        # Field name
        self.field_type = field_type           # Source type string (e.g., "U8", "U8*")
        self.llvm_type = llvm_type             # LLVM type
        self.ast_node = ast_node               # Original AST node
        self.index = index                     # Index in struct
        self.is_mutable = is_mutable           # Whether field can be modified
        self.default_value = default_value     # Raw default value from source
        self.default_llvm_value = default_llvm_value  # Compiled LLVM constant
        self.pointer_level = pointer_level     # Number of pointer levels
    
    def has_default_value(self):
        """Check if this field has a default value"""
        return self.default_value is not None
    
    def get_default_llvm_value(self):
        """Get the LLVM constant for the default value"""
        return self.default_llvm_value
    
    def __repr__(self):
        return f"<ClassFieldInfo: {self.name}:{self.field_type}, mutable={self.is_mutable}, default={self.default_value}>"

class ClassTypeInfo:
    """Enhanced class type information with better field handling"""

    def __init__(self, name: str, llvm_type, parent_type=None, node: 'ASTNode.Class' = None):
        self.name = name                       # Class name
        self.llvm_type = llvm_type            # LLVM struct type
        self.parent = parent_type             # Parent class info
        self.node = node                      # Original AST node
        self.fields = {}                      # Maps field names to ClassFieldInfo
        self.field_order = []                 # Ordered list of field names
        self.methods = {}                     # Maps method names to method info
        self._next_field_index = 0            # Track field indices

        # If we have a parent, inherit its fields first
        if self.parent:
            self._inherit_parent_fields()

    # -----------------------------
    # 🔙 Backward compatibility
    # -----------------------------
    @property
    def field_names(self):
        """Old API compatibility: returns list of field names"""
        return self.field_order

    @property
    def field_types(self):
        """Old API compatibility: returns list of LLVM types"""
        return [self.fields[name].llvm_type for name in self.field_order]

    @property
    def field_mutable(self):
        """Old API compatibility: returns list of mutability flags"""
        return [self.fields[name].is_mutable for name in self.field_order]

    @property
    def field_variables(self):
        """Old API compatibility: return underlying AST nodes for fields"""
        return [self.fields[name].ast_node for name in self.field_order]

    # -----------------------------
    # Existing methods
    # -----------------------------
    def _inherit_parent_fields(self):
        """Inherit fields from parent class"""
        if not self.parent:
            return

        for field_name in self.parent.field_order:
            parent_field = self.parent.fields[field_name]
            inherited_field = ClassFieldInfo(
                name=parent_field.name,
                field_type=parent_field.field_type,
                llvm_type=parent_field.llvm_type,
                ast_node=parent_field.ast_node,
                index=self._next_field_index,
                is_mutable=parent_field.is_mutable,
                default_value=parent_field.default_value,
                default_llvm_value=parent_field.default_llvm_value,
                pointer_level=parent_field.pointer_level,
            )
            self.fields[field_name] = inherited_field
            self.field_order.append(field_name)
            self._next_field_index += 1

    def add_field(self, field_info: ClassFieldInfo):
        field_info.index = self._next_field_index
        self.fields[field_info.name] = field_info
        self.field_order.append(field_info.name)
        self._next_field_index += 1

    def get_field(self, field_name: str) -> Optional[ClassFieldInfo]:
        return self.fields.get(field_name)

    def has_field(self, field_name: str) -> bool:
        return field_name in self.fields

    def get_field_index(self, field_name: str) -> int:
        field = self.fields.get(field_name)
        if field:
            return field.index
        raise Exception(f"Unknown field '{field_name}' in class '{self.name}'")

    def get_field_ptr(self, struct_ptr, field_name: str, builder: ir.IRBuilder):
        field_index = self.get_field_index(field_name)
        return builder.gep(
            struct_ptr,
            [ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), field_index)],
            inbounds=True,
        )

    def get_llvm_type(self):
        return self.llvm_type

    def get_fields_with_defaults(self):
        return [field for field in self.fields.values() if field.has_default_value()]

    def get_all_field_types(self):
        return [self.fields[name].llvm_type for name in self.field_order]

    def __repr__(self):
        return f"<ClassTypeInfo: {self.name}, fields={list(self.fields.keys())}, parent={self.parent.name if self.parent else None}>"


def handle_class(self, node: ASTNode.Class, **kwargs):
    """Enhanced class handler with proper field value handling"""
    
    if self.compiler.debug:
        print(f"Processing class: {node.name}")
    
    # Get the parent class info if any
    parent_type_info = None
    if node.parent:
        parent_type_info = Datatypes.get_type(node.parent)
        if not parent_type_info:
            raise Exception(f"Unknown parent class '{node.parent}'")

    # Create the identified struct type
    llvm_struct_name = f"%struct.{node.name}"
    struct_type = ir.global_context.get_identified_type(llvm_struct_name)

    # Create the enhanced class type info
    class_type_info = ClassTypeInfo(
        name=node.name,
        llvm_type=struct_type,
        parent_type=parent_type_info,
        node=node
    )

    # Process each field defined in this class
    field_llvm_types = []
    
    # First, collect inherited field types if any
    if parent_type_info:
        field_llvm_types.extend(parent_type_info.get_all_field_types())

    # Process each field defined directly in this class
    for field_ast in node.fields:
        if self.compiler.debug:
            print(f"Processing field: {field_ast.name} of type {field_ast.var_type}")
        
        # Determine the LLVM type for this field
        if field_ast.var_type == node.name:
            # Self-referential field (pointer to same class)
            field_llvm_type = struct_type.as_pointer()
        else:
            # Regular field type
            field_llvm_type = Datatypes.to_llvm_type(field_ast.var_type, field_ast.pointer_level)

        field_llvm_types.append(field_llvm_type)
        
        # Process default value if present
        default_llvm_value = None
        if field_ast.value:
            if self.compiler.debug:
                print(f"Processing default value for field {field_ast.name}")
            
            # Evaluate the default value expression to get LLVM constant
            try:
                # Use a temporary builder or evaluate as constant
                default_llvm_value = self._evaluate_field_default_value(field_ast.value, field_llvm_type)
            except Exception as e:
                if self.compiler.debug:
                    print(f"Warning: Could not evaluate default value for {field_ast.name}: {e}")
        
        # Create field info
        field_info = ClassFieldInfo(
            name=field_ast.name,
            field_type=field_ast.var_type,
            llvm_type=field_llvm_type,
            ast_node=field_ast,
            index=0,  # Will be set by add_field
            is_mutable=field_ast.is_mutable,
            default_value=field_ast.value,
            default_llvm_value=default_llvm_value,
            pointer_level=field_ast.pointer_level
        )
        
        # Add field to class type info
        class_type_info.add_field(field_info)

    # Set the body of the struct type
    struct_type.set_body(*field_llvm_types)

    # Register the class type info in the type system
    Datatypes.add_type(node.name, class_type_info)

    # Store in struct table
    self.struct_table[node.name] = {
        'name': node.name, 
        'class_type_info': class_type_info
    }

    # Process class methods (unchanged)
    for method in node.methods:
        mangled_name = f"{node.name}_{method.name}"
        
        # Create self parameter
        self_param = ASTNode.VariableDeclaration(
            var_type=node.name,
            name="self",
            pointer_level=1,
            is_user_typed=True
        )
        
        # Combine self parameter with existing parameters
        if not method.parameters or method.parameters[0].name != "self":
            all_params = [self_param] + method.parameters
        else:
            all_params = method.parameters
        
        # Create mangled method
        mangled_method = ASTNode.FunctionDefinition(
            name=mangled_name,
            return_type=method.return_type,
            parameters=all_params,
            body=method.body
        )
        
        from .functions import handle_function_definition
        handle_function_definition(self, mangled_method, **kwargs)
        
        if self.compiler.debug:
            print(f"Registered method {method.name} as function {mangled_name}")

    if self.compiler.debug:
        print(f"Class {node.name} processed successfully")
        print(f"Fields: {[f.name for f in class_type_info.fields.values()]}")
        fields_with_defaults = class_type_info.get_fields_with_defaults()
        if fields_with_defaults:
            print(f"Fields with defaults: {[f.name for f in fields_with_defaults]}")

    return None

def _evaluate_field_default_value(self, value_expr, target_llvm_type):
    """Evaluate a field's default value expression to an LLVM constant"""
    
    if value_expr.node_type == NodeType.LITERAL:
        # Handle literal values
        if isinstance(value_expr.value, str):
            # String literal
            if value_expr.value.startswith('"') and value_expr.value.endswith('"'):
                # Remove quotes and create string constant
                string_val = value_expr.value[1:-1]  # Remove quotes
                return self._create_string_constant_for_field(string_val, target_llvm_type)
            else:
                # Try to parse as number
                try:
                    if '.' in value_expr.value:
                        # Float
                        float_val = float(value_expr.value)
                        return ir.Constant(target_llvm_type, float_val)
                    else:
                        # Integer
                        int_val = int(value_expr.value)
                        return ir.Constant(target_llvm_type, int_val)
                except ValueError:
                    # If can't parse as number, treat as string
                    return self._create_string_constant_for_field(value_expr.value, target_llvm_type)
        else:
            # Direct numeric value
            return ir.Constant(target_llvm_type, value_expr.value)
    
    elif value_expr.node_type == NodeType.REFERENCE:
        # Handle references to other constants/enums
        # This would need to be implemented based on your symbol table
        pass
    
    # For complex expressions, you might need to evaluate them at compile time
    # or defer initialization to runtime
    return None

def _create_string_constant_for_field(self, string_value, target_llvm_type):
    """Create a string constant for field initialization"""
    
    if isinstance(target_llvm_type, ir.PointerType) and \
       isinstance(target_llvm_type.pointee, ir.IntType) and \
       target_llvm_type.pointee.width == 8:
        
        # This is a char* type, create a global string constant
        # Note: This is a simplified version, you might need to adapt it
        # to your existing string constant creation method
        
        string_type = ir.ArrayType(ir.IntType(8), len(string_value) + 1)  # +1 for null terminator
        string_data = bytearray(string_value.encode('utf-8') + b'\0')
        
        # Create global constant
        global_string = ir.GlobalVariable(
            self.module, 
            string_type, 
            name=f".str.field.{len(string_value)}"
        )
        global_string.initializer = ir.Constant(string_type, string_data)
        global_string.global_constant = True
        global_string.linkage = 'private'
        
        # Return pointer to the first element
        return global_string.gep([ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), 0)])
    
    return None

def create_class_instance(self, class_name: str, builder: ir.IRBuilder, initialize_defaults: bool = True):
    """Create a new instance of a class with optional default value initialization"""
    
    class_info = Datatypes.get_type(class_name)
    if not isinstance(class_info, ClassTypeInfo):
        raise ValueError(f"'{class_name}' is not a valid class type")
    
    # Allocate memory for the class instance
    instance_ptr = builder.alloca(class_info.llvm_type, name=f"{class_name.lower()}_instance")
    
    if initialize_defaults:
        # Initialize fields that have default values
        fields_with_defaults = class_info.get_fields_with_defaults()
        
        for field in fields_with_defaults:
            if field.default_llvm_value:
                # Get pointer to field
                field_ptr = class_info.get_field_ptr(instance_ptr, field.name, builder)
                
                # Store the default value
                builder.store(field.default_llvm_value, field_ptr)
                
                if self.compiler.debug:
                    print(f"Initialized field {field.name} with default value")
    
    return instance_ptr

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
    print("WARNING: union codegen handler is not defined!")
    pass