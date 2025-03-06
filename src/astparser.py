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
            else:
                result += "No Function body"
            return result

        def __repr__(self):
            return self.print_tree()    


class Parser:
    def __init__(self, tokens: List[Token], code):
        self.tokens = tokens
        self.index = 0
        self.code = code
    code: str
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
            self.syntax_error("Excepted variable name", var_name)
        
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
            raise Exception(f"Syntax Error: Expected semicolon at line {next_token.line}")
        
        return node
    
    def variable_assignment(self):
        print("Variable assignment")
        
        # First token should be the variable name
        var_name = self.current_token()
        if not var_name or var_name._type != TokenType.LITERAL:
            raise Exception(f"Syntax Error: Expected variable name at line {var_name.line}")
        
        # Move to assignment operator
        next_token = self.next_token()
        if not next_token or next_token.value != operators["ASSIGN"]:
            raise Exception(f"Syntax Error: Expected assignment operator at line {next_token.line}")
        
        # Move to value and parse expression
        self.next_token()  # Move past '='
        value = self.parse_expression()
        
        # Create variable assignment node
        node = ASTNode.VariableAssignment(var_name.value, value)
        
        # Expect semicolon
        next_token = self.current_token()
        if not next_token or next_token.value != separators["SEMICOLON"]:
            self.syntax_error("Excepted semicolon ", next_token)
        
        return node
    def parse_blocks(self) -> List[ASTNode]:
        if self.current_token().value != separators["LBRACE"]:
            self.syntax_error("Expected '{'", self.current_token())
        self.next_token()
        statements = []

        while self.current_token() and self.current_token().value != separators["RBRACE"]:
            statement = self.parse_statements()
            if statement:
                statements.append(statement)
        # Check for '}'
        if not self.current_token() or self.current_token().value != separators["RBRACE"]:
            self.syntax_error("Expected '}'", self.current_token())
        pass

        # Skip the '}'
        self.next_token()

        return statements
    def parse_return_statement(self) -> ASTNode:
        # Parse variable name
        var_name = None
        if self.current_token().value == keywords["RETURN"]:
            self.next_token()  # Skip the 'return' keyword
            if self.current_token() and self.current_token().value != separators["SEMICOLON"]:
                var_name = self.current_token().value
                self.next_token()  # Skip the variable name
            
            # Check for semicolon
            if not self.current_token() or self.current_token().value != separators["SEMICOLON"]:
                self.syntax_error("Expected ';'", self.current_token())
            
            self.next_token()  # Skip the semicolon
        return ASTNode.Return(var_name)
    def function_declaration(self):
        print("Function declaration parsing")

        return_type = self.current_token()
        if not return_type or return_type._type != TokenType.KEYWORD:
            self.syntax_error("Excepted return type", return_type)
        if not (return_type.value in Datatypes.all_types()):
            self.syntax_error("Invalid return type", return_type)
        
        next_token = self.next_token()
        if not next_token or next_token._type != TokenType.LITERAL:
            self.syntax_error("Excepted function name", return_type)
        func_name = next_token.value

        # Move to opening parenthesis '('
        next_token = self.next_token()
        if not next_token or next_token.value != separators["LPAREN"]:
            self.syntax_error("Excepted '('", return_type)

        parameters = []
        peek_token = self.peek_token()
        if peek_token and peek_token.value == separators["RPAREN"]:
            self.next_token()  # Consume the ')'
        while True:
            param_type = self.next_token()
            if not param_type or param_type._type != TokenType.KEYWORD:
                self.syntax_error("Expected parameter type", param_type)
            if not (param_type.value in Datatypes.all_types()):
                self.syntax_error("Invalid parameter type", param_type)
            
            param_name = self.next_token()
            if not param_name or param_name._type != TokenType.LITERAL:
                self.syntax_error("Expected parameter name", param_name)
            
            # Create parameter variable declaration
            param = ASTNode.VariableDeclaration(param_type.value, param_name.value)
            parameters.append(param)
            
            next_token = self.next_token()
            if next_token.value == separators["RPAREN"]:
                break
            elif next_token.value != separators["COMMA"]:
                self.syntax_error("Expected ',' or ')'", next_token)
            

        # check semicolon or body brace
        next_token = self.next_token()
        if next_token._type != TokenType.SEPARATOR or (next_token.value != separators["SEMICOLON"] and next_token.value != separators["LBRACE"]):
            print(next_token)
            self.syntax_error("Excepted ';' or '{'", next_token)

        # Parse function body   
        function_body = List[ASTNode]
        # Idk how. What's even a body, a set of AST nodes?  
        
        function_node = ASTNode.FunctionDefinition(func_name, return_type.value, parameters, function_body)
        return function_node

    def parse(self):
        nodes = []
        print("AST Parsing Started")
        while self.index < len(self.tokens):
            current_token = self.current_token()
            if current_token is None:
                break
            
            if current_token._type == TokenType.KEYWORD:
                print(current_token)
                next_token = self.peek_token()
                if next_token._type == TokenType.LITERAL and self.peek_token(2).value == separators["LPAREN"]:
                    node = self.function_declaration()
                    nodes.append(node)
                if current_token.value in Datatypes.all_types():
                    node = self.variable_declaration()
                    nodes.append(node)
                if current_token.value == keywords["RETURN"]:
                    node = self.parse_return_statement()
                    nodes.append(node)

            
            elif current_token._type == TokenType.LITERAL:
                next_token = self.peek_token()
                if next_token and next_token.value == operators["ASSIGN"]:
                    node = self.variable_assignment()
                    nodes.append(node)
                else:
                    self.next_token()
            else:
                self.next_token()
        
        return nodes


    def syntax_error(self, message, token):
        print(token)
        error_line = self.code.splitlines()[token.line - 1]
        caret_position = " " * (token.column) + "^"
        raise Exception(f"Syntax Error: {message} at line {token.line}, column {token.column}\n{error_line}\n{caret_position}")
