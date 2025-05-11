from typing import List, Union, Optional, Dict, Any, Tuple
from enum import Enum, auto
from lexer import *

class Variable:
    def __init__(self, name: str, value: Any):
        self.name: str = name
        self.value: Any = value

class NodeType(Enum):
    FUNCTION_DEFINITION = auto()
    VARIABLE_DECLARATION = auto()
    VARIABLE_ASSIGNMENT = auto()
    STRUCT_FIELD_ASSIGNMENT = auto()  # Add this for struct field assignments
    RETURN = auto()
    LITERAL = auto()
    UNARY_OP = auto()
    BINARY_OP = auto()
    FUNCTION_CALL = auto()
    BLOCK = auto()
    IF_STATEMENT = auto()
    WHILE_LOOP = auto()
    FOR_LOOP = auto()
    ARRAY_ACCESS = auto()
    STRUCT_ACCESS = auto()
    REFERENCE = auto()
    COMMENT = auto()
    BREAK = auto()
    CONTINUE = auto()
    CLASS = auto()
    UNION = auto()
    ARRAY_DECLARATION = auto()
    ARRAY_INITIALIZATION = auto()
    EXTERN = auto()

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
        def __init__(self, node_type: NodeType, value: Any = None, 
                    left: Optional['ASTNode.ExpressionNode'] = None, 
                    right: Optional['ASTNode.ExpressionNode'] = None, 
                    op: Optional[str] = None):
            self.node_type: NodeType = node_type
            self.value: Any = value  # For literals and identifiers
            self.left: Optional['ASTNode.ExpressionNode'] = left    # For binary and unary ops
            self.right: Optional['ASTNode.ExpressionNode'] = right  # For binary ops
            self.op: Optional[str] = op        # Operator for binary/unary ops

        def print_tree(self, prefix: str = "") -> str:
            result = f"{prefix}ExpressionNode ({self.node_type}"
            if self.op:
                result += f": {self.op}"
            if self.value is not None:
                result += f": {self.value}"
            result += ")\n"

            return result

    class Block:
        def __init__(self, nodes: List[Any]):
            self.nodes: List[Any] = nodes

        def print_tree(self, prefix: str = "") -> str:
            result = f"{prefix}Block\n"
            for node in self.nodes:
                # Check if the node has a print_tree method (i.e., it's a valid AST node)
                if hasattr(node, 'print_tree'):
                    result += node.print_tree(prefix + "    ")
                else:
                    result += f"{prefix}    {str(node)}\n"  # For non-structured nodes (simple types)
            return result

        def __repr__(self) -> str:
            return self.print_tree()

    class VariableDeclaration:
        def __init__(self, var_type: str, name: str, value: Optional[Any] = None, 
                    is_user_typed: bool = False, is_pointer: bool = False):
            self.var_type: str = var_type
            self.name: str = name
            self.value: Optional[Any] = value
            self.is_user_typed: bool = is_user_typed
            self.is_pointer: bool = is_pointer

        def print_tree(self, prefix: str = "") -> str:
            result = f"{prefix}VariableDeclaration\n"
            result += f"{prefix}├── var_type: {self.var_type}\n"
            result += f"{prefix}├── name: {self.name}\n"
            if self.value:
                if isinstance(self.value, str):
                    result += f"{prefix}└── value: {self.value + '\n'}"
                else:
                    result += f"{prefix}└── value: {self.value.print_tree(prefix + '    ')}"
            if self.is_user_typed:
                result += f"{prefix}└── user_typed: {self.is_user_typed}"
            if self.is_pointer:
                result += f"{prefix}└── is_pointer: {self.is_pointer}"
            return result

        def __repr__(self) -> str:
            return self.print_tree()

    class VariableAssignment:
        def __init__(self, name: str, value: Optional[Any] = None):
            self.name: str = name
            self.value: Optional[Any] = value

        def print_tree(self, prefix: str = "") -> str:
            result = f"{prefix}VariableAssignment\n"
            result += f"{prefix}├── name: {self.name}\n"
            if self.value:
                result += f"{prefix}└── value: {self.value.print_tree(prefix + '    ')}"
            return result

        def __repr__(self) -> str:
            return self.print_tree()

    class Return:
        def __init__(self, expression: Any):
            self.expression: Any = expression 

        def print_tree(self, prefix: str = "") -> str:
            return f"{prefix}Return\n{prefix}└── expression: {self.expression}\n"

        def __repr__(self) -> str:
            return self.print_tree()

    class FunctionDefinition:
        def __init__(self, name: str, return_type: str, body: 'ASTNode.Block', 
                    parameters: List['ASTNode.VariableDeclaration']):
            self.name: str = name
            self.return_type: str = return_type
            self.parameters: List['ASTNode.VariableDeclaration'] = parameters or []
            self.body: 'ASTNode.Block' = body 

        def print_tree(self, prefix: str = "") -> str:
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

        def __repr__(self) -> str:
            return self.print_tree() 

    class IfStatement:
        def __init__(self, condition: 'ASTNode.ExpressionNode', 
                    if_body: 'ASTNode.Block', 
                    else_body: Optional['ASTNode.Block'] = None):
            self.condition: 'ASTNode.ExpressionNode' = condition 
            self.if_body: 'ASTNode.Block' = if_body      
            self.else_body: Optional['ASTNode.Block'] = else_body  # Optional Block

        def print_tree(self, prefix: str = "") -> str:
            result = f"{prefix}IfStatement\n"
            result += f"{prefix}├── condition: {self.condition.print_tree(prefix + '│   ')}"
            result += f"{prefix}├── if_body: {self.if_body.print_tree(prefix + '│   ')}"

            if self.else_body:
                result += f"{prefix}├── else_body: {self.else_body.print_tree(prefix + '│   ')}"

            return result

        def __repr__(self) -> str:
            return self.print_tree()

    class WhileLoop:
        def __init__(self, condition: 'ASTNode.ExpressionNode', body: 'ASTNode.Block'):
            self.condition: 'ASTNode.ExpressionNode' = condition
            self.body: 'ASTNode.Block' = body

        def print_tree(self, prefix: str = "") -> str:
            result = f"{prefix}WhileLoop\n"
            result += f"{prefix}├── condition: {self.condition.print_tree(prefix + '│   ')}"
            result += f"{prefix}└── body: {self.body.print_tree(prefix + '    ')}"
            return result

        def __repr__(self) -> str:
            return self.print_tree()
        
    class ForLoop:
        def __init__(self, initialization: Optional['ASTNode.VariableDeclaration'], 
                    condition: Optional['ASTNode.ExpressionNode'], 
                    update: Optional['ASTNode.VariableAssignment'], 
                    body: 'ASTNode.Block'):
            self.initialization: Optional['ASTNode.VariableDeclaration'] = initialization 
            self.condition: Optional['ASTNode.ExpressionNode'] = condition           
            self.update: Optional['ASTNode.VariableAssignment'] = update  # TODO! didn't implemented ++ or -- feature so it's just variable assignment now 
            self.body: 'ASTNode.Block' = body                     
        
        def print_tree(self, prefix: str = "") -> str:
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
        
        def __repr__(self) -> str:
            return self.print_tree()
        
    class Comment:
        def __init__(self, text: str, is_inline: bool = False):
            self.text: str = text
            self.is_inline: bool = is_inline
            self.node_type: str = "Comment"

        def print_tree(self, prefix: str = "") -> str:
            type_info = "Inline" if self.is_inline else "Block"
            return f"{prefix}{self.node_type} ({type_info}): {self.text}\n"

        def __repr__(self) -> str:
            return self.print_tree()
        
    class FunctionCall:
        def __init__(self, name: str, arguments: List['ASTNode.ExpressionNode'], has_parentheses: bool = False):
            self.name: str = name
            self.arguments: List['ASTNode.ExpressionNode'] = arguments  # List of expressions
            self.has_parentheses: bool = has_parentheses  # Flag to indicate if parentheses were used

        def print_tree(self, prefix: str = "") -> str:
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

        def __repr__(self) -> str:
            return self.print_tree()
        
    class Class:
        def __init__(self, name: str, fields: List['ASTNode.VariableDeclaration'], parent: Optional[str] = None):
            self.name: str = name
            self.fields: List['ASTNode.VariableDeclaration'] = fields  # List of variable assignments
            self.parent: Optional[str] = parent  
            Datatypes.add_type(name, self)
            keywords[name] = name

        def print_tree(self, prefix: str = "") -> str:
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

        def __repr__(self) -> str:
            return self.name


    class Union:
        def __init__(self, name: str, fields: List['ASTNode.VariableDeclaration']):
            self.name: str = name
            self.fields: List['ASTNode.VariableDeclaration'] = fields  # List of variable assignments
            Datatypes.add_type(name, self)

        def print_tree(self, prefix: str = "") -> str:
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

        def __repr__(self) -> str:
            return self.print_tree()

    class Break:
        def print_tree(self, prefix: str = "") -> str:
            return f"{prefix}Break\n"

        def __repr__(self) -> str:
            return self.print_tree()

    class Continue:
        def print_tree(self, prefix: str = "") -> str:
            return f"{prefix}Continue\n"

        def __repr__(self) -> str:
            return self.print_tree()

    class ArrayDeclaration:
        def __init__(self, base_type: str, name: str, 
                    dimensions: List[Optional['ASTNode.ExpressionNode']], 
                    initialization: Optional['ASTNode.ArrayInitialization'] = None):
            self.base_type: str = base_type  # The base type (U8, U64, etc.)
            self.name: str = name            # Array variable name
            self.dimensions: List[Optional['ASTNode.ExpressionNode']] = dimensions  # List of expression nodes representing dimensions
            self.initialization: Optional['ASTNode.ArrayInitialization'] = initialization  # Optional initialization expression

        def print_tree(self, prefix: str = "") -> str:
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

        def __repr__(self) -> str:
            return self.print_tree()
        
    class ArrayInitialization:
        def __init__(self, elements: List[Union['ASTNode.ExpressionNode', 'ASTNode.ArrayInitialization']]):
            self.elements: List[Union['ASTNode.ExpressionNode', 'ASTNode.ArrayInitialization']] = elements  # List of expressions or nested ArrayInitializations

        def print_tree(self, prefix: str = "") -> str:
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

        def __repr__(self) -> str:
            return self.print_tree()
        
    class Pointer:
        def __init__(self, variable_name: str):
            self.variable_name: str = variable_name

        def print_tree(self, prefix: str = "") -> str:
            result = f"{prefix}Pointer\n"
            result += f"{prefix}└── variable_name: {self.variable_name}\n"
            return result

        def __repr__(self) -> str:
            return self.print_tree()

    class Reference:
        def __init__(self, variable_name: str):
            self.variable_name: str = variable_name
            
        def print_tree(self, prefix: str = "") -> str:
            result = f"{prefix}Reference\n"
            result += f"{prefix}└── variable_name: {self.variable_name}\n"
            return result
            
        def __repr__(self) -> str:
            return self.print_tree()
            
    class StructFieldAssignment:
        def __init__(self, struct_name, field_name, value):
            self.struct_name = struct_name
            self.field_name = field_name
            self.value = value

        def print_tree(self, prefix: str = "") -> str:
            result = f"{prefix}StructFieldAssignment\n"
            result += f"{prefix}├── struct_name: {self.struct_name}\n"
            result += f"{prefix}├── field_name: {self.field_name}\n"
            if hasattr(self.value, 'print_tree'):
                result += f"{prefix}└── value:\n"
                result += self.value.print_tree(prefix + "    ")
            else:
                result += f"{prefix}└── value: {self.value}\n"
            return result

        def __repr__(self) -> str:
            return self.print_tree()

    class VariableIncrement:
        def __init__(self, name: str):
            self.name: str = name

        def print_tree(self, prefix: str = "") -> str:
            return f"{prefix}VariableIncrement\n{prefix}└── name: {self.name}\n"

        def __repr__(self) -> str:
            return self.print_tree()

    class VariableDecrement:
        def __init__(self, name: str):
            self.name: str = name

        def print_tree(self, prefix: str = "") -> str:
            return f"{prefix}VariableDecrement\n{prefix}└── name: {self.name}\n"

        def __repr__(self) -> str:
            return self.print_tree()



 
        
    class Extern:
        class ExternType(Enum):
            FUNCTION = auto()
            VARIABLE = auto()
            STRUCT = auto()
            UNKNOWN = auto()   
        def __init__(self, declaration: Any):
            """
            Initialize an extern declaration that can wrap any declaration type
            
            Args:
                declaration: The declaration being marked as extern (function, variable, etc.)
            """
            self.declaration = declaration
            
            # Determine and store the type of the declaration
            if isinstance(declaration, ASTNode.FunctionDefinition):
                self.extern_type = self.ExternType.FUNCTION
            elif isinstance(declaration, ASTNode.VariableDeclaration):
                self.extern_type = self.ExternType.VARIABLE
            elif isinstance(declaration, ASTNode.Class):
                self.extern_type = self.ExternType.STRUCT
            else:
                self.extern_type = self.ExternType.UNKNOWN

        def print_tree(self, prefix: str = "") -> str:
            result = f"{prefix}Extern ({self.extern_type.name.lower()})\n"
            result += f"{prefix}└── declaration: {self.declaration.print_tree(prefix + '    ')}"
            return result

        def __repr__(self) -> str:
            return self.print_tree()