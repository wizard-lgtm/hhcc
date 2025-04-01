from typing import List, Union
from enum import Enum, auto
from lexer import *
class Variable:
    def __init__(self, name, value):
        self.name = name
        self.value = value

class NodeType(Enum):
    LITERAL = auto()
    BINARY_OP = auto()
    UNARY_OP = auto()
    ASSIGNMENT = auto()
    CALL = auto()
    ARRAY_ACCESS = auto()
    STRUCT_ACCESS = auto()
    REFERENCE = auto()

class ASTNodeType(Enum):
    # Expression nodes
    EXPRESSION = auto()
    
    # Block structure
    BLOCK = auto()
    
    # Variable operations
    VARIABLE_DECLARATION = auto()
    VARIABLE_ASSIGNMENT = auto()
    
    # Control flow
    RETURN = auto()
    IF_STATEMENT = auto()
    ELSE_STATEMENT = auto()
    WHILE_LOOP = auto()
    FOR_LOOP = auto()
    BREAK = auto()
    CONTINUE = auto()
    
    # Function related
    FUNCTION_DEFINITION = auto()
    FUNCTION_CALL = auto()
    
    # Classes and custom types
    CLASS_DEFINITION = auto()
    UNION_DEFINITION = auto()
    
    # Arrays
    ARRAY_DECLARATION = auto()
    ARRAY_INITIALIZATION = auto()
    
    # Pointers and references
    POINTER = auto()
    REFERENCE = auto()
    
    # Metadata
    COMMENT = auto()

class ASTNode:
    class ExpressionNode:
        def __init__(self, node_type: NodeType, value=None, left=None, right=None, op=None):
            self.node_type = node_type
            self.value = value  # For literals and identifiers
            self.left = left    # For binary and unary ops
            self.right = right  # For binary ops
            self.op = op        # Operator for binary/unary ops

        def print_tree(self, prefix=""):
            result = f"{prefix}ExpressionNode ({self.node_type}"
            if self.op:
                result += f": {self.op}"
            if self.value is not None:
                result += f": {self.value}"
            result += ")\n"

            if self.left:
                result += f"{prefix}├── left: {self.left.print_tree(prefix + '│   ')}"
            if self.right:
                result += f"{prefix}└── right: {self.right.print_tree(prefix + '    ')}"

            return result

        def __repr__(self):
            return self.print_tree()


    class Block:
        nodes: List

        def __init__(self, nodes):
            self.nodes = nodes

        def print_tree(self, prefix=""):
            result = f"{prefix}Block\n"
            for node in self.nodes:
                # Check if the node has a print_tree method (i.e., it's a valid AST node)
                if hasattr(node, 'print_tree'):
                    result += node.print_tree(prefix + "    ")
                else:
                    result += f"{prefix}    {str(node)}\n"  # For non-structured nodes (simple types)
            return result

        def __repr__(self):
            return self.print_tree()


    class VariableDeclaration:
        def __init__(self, var_type, name, value=None, is_user_typed=False, is_pointer=False):
            self.var_type = var_type
            self.name = name
            self.value = value
            self.is_user_typed = is_user_typed
            self.is_pointer = is_pointer

        def print_tree(self, prefix=""):
            result = f"{prefix}VariableDeclaration\n"
            result += f"{prefix}├── var_type: {self.var_type}\n"
            result += f"{prefix}├── name: {self.name}\n"
            if self.value:
                if isinstance(self.value, str):
                    result += f"{prefix}└── value: {self.value + "\n"}"
                else:
                    result += f"{prefix}└── value: {self.value.print_tree(prefix + '    ')}"
            if self.is_user_typed:
                result += f"{prefix}└── user_typed: {self.is_user_typed}"
            if self.is_pointer:
                result += f"{prefix}└── is_pointer: {self.is_pointer}"
            return result

        def __repr__(self):
            return self.print_tree()

    class VariableAssignment:
        def __init__(self, name, value=None):
            self.name = name
            self.value = value

        def print_tree(self, prefix=""):
            result = f"{prefix}VariableAssignment\n"
            result += f"{prefix}├── name: {self.name}\n"
            if self.value:
                result += f"{prefix}└── value: {self.value.print_tree(prefix + '    ')}"
            return result

        def __repr__(self):
            return self.print_tree()

    class Return:
        def __init__(self, expression):
            self.expression = expression 

        def print_tree(self, prefix=""):
            return f"{prefix}Return\n{prefix}└── expression: {self.expression}\n"

        def __repr__(self):
            return self.print_tree()

    class FunctionDefinition:
        def __init__(self, name, return_type, body, parameters ): 
            self.name = name
            self.return_type = return_type
            self.parameters = parameters or []
            self.body = body 

        def print_tree(self, prefix=""):
            result = f"{prefix}FunctionDefinition\n"
            result += f"{prefix}├── name: {self.name}\n"
            result += f"{prefix}├── return_type: {self.return_type}\n"
            if self.parameters:
                result += f"{prefix}├── parameters:\n"
                if self.parameters:
                    for param in self.parameters:
                        result += param.print_tree(prefix + "│  ")
            if self.body:
                result += f"{prefix}└── body: {self.body}\n"
                
            else:
                result += "No Function body"
            return result

        def __repr__(self):
            return self.print_tree() 
    class IfStatement:
        def __init__(self, condition, if_body, else_body = None):
            self.condition = condition 
            self.if_body = if_body      
            self.else_body = else_body  # A block Optinal Optinal Optinal

        def print_tree(self, prefix=""):
            result = f"{prefix}IfStatement\n"
            result += f"{prefix}├── condition: {self.condition.print_tree(prefix + '│   ')}"
            
            result += f"{prefix}├── if_body: {self.if_body.print_tree(prefix + '│   ')}"

            if self.else_body:
                result += f"{prefix}├── else_body: {self.else_body.print_tree(prefix + '│   ')}"

            return result

        def __repr__(self):
            return self.print_tree()

    class WhileLoop:
        def __init__(self, condition, body):
            self.condition = condition
            self.body = body

        def print_tree(self, prefix=""):
            result = f"{prefix}WhileLoop\n"
            result += f"{prefix}├── condition: {self.condition.print_tree(prefix + '│   ')}"
            result += f"{prefix}└── body: {self.body.print_tree(prefix + '    ')}"
            return result

        def __repr__(self):
            return self.print_tree()
        
    class ForLoop:
        def __init__(self, initialization, condition, update, body):
            self.initialization = initialization 
            self.condition = condition           
            self.update = update                 # TODO! didn't implemented ++ or -- feature so it's just variable assigmnet now 
            self.body = body                     
        
        def print_tree(self, prefix=""):
            result = f"{prefix}ForLoop\n"
            
            if self.initialization:
                result += f"{prefix}├── initialization: {self.initialization.print_tree(prefix + '│   ')}"
            else:
                result += f"{prefix}├── initialization: None\n"
                
            if self.condition:
                result += f"{prefix}├── condition: {self.condition.print_tree(prefix + '│   ')}"
            else:
                result += f"{prefix}├── condition: None\n"
                
            if self.update:
                result += f"{prefix}├── update: {self.update.print_tree(prefix + '│   ')}"
            else:
                result += f"{prefix}├── update: None\n"
                
            result += f"{prefix}└── body: {self.body.print_tree(prefix + '    ')}"
            
            return result
        
        def __repr__(self):
            return self.print_tree()
        
    class Comment:
        def __init__(self, text):
            self.text = text

        def print_tree(self, prefix=""):
            return f"{prefix}Comment: {self.text}\n"

        def __repr__(self):
            return self.print_tree()    
        
    class FunctionCall:
        def __init__(self, name, arguments, has_parentheses=False):
            self.name = name
            self.arguments = arguments  # List of expressions
            self.has_parentheses = has_parentheses  # Flag to indicate if parentheses were used

        def print_tree(self, prefix=""):
            result = f"{prefix}FunctionCall\n"
            result += f"{prefix}├── name: {self.name}\n"
            result += f"{prefix}├── has_parentheses: {self.has_parentheses}\n"
            
            if self.arguments:
                result += f"{prefix}└── arguments:\n"
                for i, arg in enumerate(self.arguments):
                    if i < len(self.arguments) - 1:
                        result += f"{prefix}    ├── {arg.print_tree(prefix + '    │   ')}"
                    else:
                        result += f"{prefix}    └── {arg.print_tree(prefix + '        ')}"
            else:
                result += f"{prefix}└── arguments: []\n"
                
            return result

        def __repr__(self):
            return self.print_tree()
        
    class Class:
        def __init__(self, name, fields, parent=None):
            self.name = name
            self.fields = fields  # List of variable assignments
            self.parent = parent  
            Datatypes.add_type(name, self)
            keywords[name] = name

        def print_tree(self, prefix=""):
            result = f"{prefix}Class\n"
            result += f"{prefix}├── name: {self.name}\n"

            if self.parent:
                result += f"{prefix}└── parent: {self.parent}\n"  

            if self.fields:
                result += f"{prefix}└── fields:\n"
                for i, field in enumerate(self.fields):
                    if i < len(self.fields) - 1:
                        result += f"{prefix}    ├── {field.print_tree(prefix + '    │   ')}"
                    else:
                        result += f"{prefix}    └── {field.print_tree(prefix + '        ')}"
            else:
                result += f"{prefix}└── fields: []\n"

            return result

        def __repr__(self):
            return self.print_tree()

    class Union:
        def __init__(self, name, fields):
            self.name = name
            self.fields = fields  # List of variable assignments
            Datatypes.add_type(name, self)

        def print_tree(self, prefix=""):
            result = f"{prefix}Union\n"
            result += f"{prefix}├── name: {self.name}\n"

            if self.fields:
                result += f"{prefix}└── fields:\n"
                for i, field in enumerate(self.fields):
                    if i < len(self.fields) - 1:
                        result += f"{prefix}    ├── {field.print_tree(prefix + '    │   ')}"
                    else:
                        result += f"{prefix}    └── {field.print_tree(prefix + '        ')}"
            else:
                result += f"{prefix}└── fields: []\n"

            return result

        def __repr__(self):
            return self.print_tree()

    class Break:
        def print_tree(self, prefix=""):
            return f"{prefix}Break\n"

        def __repr__(self):
            return self.print_tree()

    class Continue:
        def print_tree(self, prefix=""):
            return f"{prefix}Continue\n"

        def __repr__(self):
            return self.print_tree()

    class ArrayDeclaration:
        def __init__(self, base_type, name, dimensions, initialization=None):
            self.base_type = base_type  # The base type (U8, U64, etc.)
            self.name = name            # Array variable name
            self.dimensions = dimensions  # List of expression nodes representing dimensions
            self.initialization = initialization  # Optional initialization expression

        def print_tree(self, prefix=""):
            result = f"{prefix}ArrayDeclaration\n"
            result += f"{prefix}├── base_type: {self.base_type}\n"
            result += f"{prefix}├── name: {self.name}\n"
            
            result += f"{prefix}├── dimensions:\n"
            for i, dim in enumerate(self.dimensions):
                if dim:
                    if i < len(self.dimensions) - 1:
                        result += f"{prefix}│   ├── {dim.print_tree(prefix + '│   │   ')}"
                    else:
                        result += f"{prefix}│   └── {dim.print_tree(prefix + '│       ')}"
                else:
                    result += f"{prefix}│   ├── dynamic[]\n"
            
            if self.initialization:
                result += f"{prefix}└── initialization: {self.initialization.print_tree(prefix + '    ')}"
            
            return result

        def __repr__(self):
            return self.print_tree()
        
    class ArrayInitialization:
        def __init__(self, elements):
            self.elements = elements  # List of expressions or nested ArrayInitializations

        def print_tree(self, prefix=""):
            result = f"{prefix}ArrayInitialization\n"
            
            if self.elements:
                for i, element in enumerate(self.elements):
                    if i < len(self.elements) - 1:
                        result += f"{prefix}├── {element.print_tree(prefix + '│   ')}"
                    else:
                        result += f"{prefix}└── {element.print_tree(prefix + '    ')}"
            else:
                result += f"{prefix}└── [empty]\n"
                
            return result

        def __repr__(self):
            return self.print_tree()
        
    class Pointer:
        def __init__(self, variable_name: str):
            self.variable_name = variable_name

        def print_tree(self, prefix=""):
            result = f"{prefix}Pointer\n"
            result += f"{prefix}└── variable_name: {self.variable_name}\n"
            return result

        def __repr__(self):
            return self.print_tree()

    class Reference:
        def __init__(self, variable_name):
            self.variable_name = variable_name
            
        def print_tree(self, prefix=""):
            result = f"{prefix}Reference\n"
            result += f"{prefix}└── variable_name: {self.variable_name}\n"
            return result
            
        def __repr__(self):
            return self.print_tree()