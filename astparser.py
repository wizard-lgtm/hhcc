from typing import List
from lexer import *
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


from typing import List, Union

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
        def __init__(self, name, return_type, parameters: List["ASTNode.VariableDeclaration"] = None, body: List[Union["ASTNode.ExpressionNode", "ASTNode.VariableDeclaration", "ASTNode.Return"]] = None):
            self.name = name
            self.return_type = return_type
            self.parameters = parameters or []
            self.body = body or []

        def print_tree(self, prefix=""):
            result = f"{prefix}FunctionDefinition\n"
            result += f"{prefix}├── name: {self.name}\n"
            result += f"{prefix}├── return_type: {self.return_type}\n"
            if self.parameters:
                result += f"{prefix}├── parameters:\n"
                for param in self.parameters:
                    result += param.print_tree(prefix + "│   ")
            if self.body:
                result += f"{prefix}└── body:\n"
                for stmt in self.body:
                    result += stmt.print_tree(prefix + "    ")
            return result

        def __repr__(self):
            return self.print_tree()    


class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.index = 0
    tokens: List[Token]
    index: int

    def current_token(self):
        if self.index < len(self.tokens):
            return self.tokens[self.index]
        else:
            return None
    
    def next_token(self):
        self.index += 1
        return self.current_token()

    def peek_token(self, offset=1):
        """Look ahead at next tokens without consuming them"""
        if self.index + offset < len(self.tokens):
            return self.tokens[self.index + offset]
        return None
    
    def check_semicolon(self):
        if self.current_token().value == separators["SEMICOLON"]:
            raise Exception(f"Syntax Error: Expected semicolon at line {self.current_token().line}")
        
    def parse_expression(self):
        start_token = self.current_token()
        
        # Handle literals and identifiers
        if start_token._type in [TokenType.LITERAL, TokenType.IDENTIFIER]:
            node = ASTNode.ExpressionNode(
                NodeType.LITERAL if start_token._type == TokenType.LITERAL else NodeType.IDENTIFIER, 
                value=start_token.value
            )
            self.next_token()  # Consume the token
            
            # Check for binary operations
            next_token = self.current_token()
            if next_token and next_token.value in operators.values():
                # Parse binary operation
                op_token = next_token
                self.next_token()  # Consume operator
                
                # Parse right side of the operation
                right_expr = self.parse_expression()
                
                node = ASTNode.ExpressionNode(
                    NodeType.BINARY_OP,
                    left=node,
                    right=right_expr,
                    op=op_token.value
                )
            
            return node
        
        raise Exception(f"Unexpected token in expression: {start_token}")

    def variable_declaration(self):
        print("Variable declaration")
        var_type = self.current_token()
        
        # Move to variable name
        var_name = self.next_token()
        if not var_name or var_name._type != TokenType.LITERAL:
            raise Exception(f"Syntax Error: Expected variable name at line {var_name.line}")
        
        node = ASTNode.VariableDeclaration(var_type.value, var_name.value, None)
        
        # Move to next token to check assignment or semicolon
        next_token = self.next_token()
        
        # Check for assignment
        if next_token and next_token.value == operators["ASSIGN"]:
            print("Assignment detected")
            # Move to value and parse expression
            self.next_token()  # Move past '='
            node.value = self.parse_expression()
            
            # Expect semicolon
            next_token = self.current_token()
        
        # Check semicolon
        if not next_token or next_token.value != separators["SEMICOLON"]:
            print(next_token)
            raise Exception(f"Syntax Error: Expected semicolon at line {next_token.line}")
        
        print(node)
        return node

    def parse(self):
        nodes = []
        print("AST Parsing Started")
        while self.index < len(self.tokens):
            current_token = self.current_token()
            if current_token is None:
                break
            
            if current_token._type == TokenType.KEYWORD and current_token.value in Datatypes.all_types():
                node = self.variable_declaration()
                nodes.append(node)
            else:
                # Important: Move to next token if not a variable declaration
                # to prevent infinite loop
                self.next_token()
        
        return nodes
        
        