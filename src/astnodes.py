from typing import List, Union
from enum import Enum, auto
class Variable:
    def __init__(self, name, value):
        self.name = name
        self.value = value

class NodeType(Enum):
    LITERAL = auto()
    IDENTIFIER = auto()
    BINARY_OP = auto()
    UNARY_OP = auto()
    ASSIGNMENT = auto()
    CALL = auto()
class ASTNode:
    class ExpressionNode:
        def __init__(self, node_type, value=None, left=None, right=None, op=None):
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
        def __init__(self, var_type, name, value=None):
            self.var_type = var_type
            self.name = name
            self.value = value

        def print_tree(self, prefix=""):
            result = f"{prefix}VariableDeclaration\n"
            result += f"{prefix}├── var_type: {self.var_type}\n"
            result += f"{prefix}├── name: {self.name}\n"
            if self.value:
                if isinstance(self.value, str):
                    result += f"{prefix}└── value: {self.value + "\n"}"
                else:
                    result += f"{prefix}└── value: {self.value.print_tree(prefix + '    ')}"
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
        def __init__(self, var_name):
            self.var_name = var_name

        def print_tree(self, prefix=""):
            return f"{prefix}Return\n{prefix}└── var_name: {self.var_name}\n"

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
                        result += param.print_tree(prefix + "│   ")
            if self.body:
                result += f"{prefix}└── body: {self.body}\n"
                
            else:
                result += "No Function body"
            return result

        def __repr__(self):
            return self.print_tree() 