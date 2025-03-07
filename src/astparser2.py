from lexer import *
from typing import List 
from astnodes import *  
class ASTParser: 
    code: str
    index: int
    tokens: List[Token]
    nodes: List[ASTNode]
    def __init__(self, tokens: List[Token], code: str):
        self.code = code
        self.index = 0
        self.tokens = tokens
        self.nodes = []

    def current_token(self):
        if self.index < len(self.tokens):
            return self.tokens[self.index]
        else:
            return None
    
    def next_token(self):
        self.index += 1
        return self.current_token()
    
    def peek_token(self, offset=1):
        """look ahead at next tokens without consuming them"""
        if self.index + offset < len(self.tokens):
            return self.tokens[self.index + offset]
        return None
    
    def syntax_error(self, message, token):
        error_line = self.code.splitlines()[token.line - 1]
        caret_position = " " * (token.column) + "^"
        print(f"caused token: {token}")
        raise Exception(f"Syntax Error: {message} at line {token.line}, column {token.column}\n{error_line}\n{caret_position}")

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
        
        # Check semicolon
        self.check_semicolon()
        
        return node
    
    def variable_assignment(self):
        print("Variable assignment")
        
        # First token should be the variable name
        var_name = self.current_token()
        if not var_name or var_name._type != TokenType.LITERAL:
            self.syntax_error("Excepted variable name", var_name)
        
        # Move to assignment operator
        next_token = self.next_token()
        if not next_token or next_token.value != operators["ASSIGN"]:
            self.syntax_error("Excepted assigmnet operator", next_token)
        
        # Move to value and parse expression
        self.next_token()  # Move past '='
        value = self.parse_expression()
        
        # Create variable assignment node
        node = ASTNode.VariableAssignment(var_name.value, value)
        
        # Expect semicolon
        self.check_semicolon()

        return node

    def return_statement(self) -> ASTNode:
        # Parse variable name
        print("Return Statement")
        var_name = None
        if self.current_token().value == keywords["RETURN"]:
            self.next_token()  # Skip the 'return' keyword
            if self.current_token() and self.current_token().value != separators["SEMICOLON"]:
                var_name = self.current_token().value
                self.next_token()  # Skip the variable name
        
        # Check semicolon 
        self.check_semicolon()
        return ASTNode.Return(var_name)

    def check_semicolon(self):
        current_token = self.current_token()
        if not (current_token._type == TokenType.SEPARATOR and current_token.value == separators["SEMICOLON"]):
            self.syntax_error("Excepted semicolon", current_token)
        self.next_token() # Consume semicolon

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
        
        self.syntax_error("Unexpected token", start_token)
        
    def parse_block(self):
        pass

    def parse_statement(self, inside_block: bool) -> ASTNode:
        current_token = self.current_token()
        if not current_token:
            return None

        if current_token._type == TokenType.KEYWORD:
            if current_token.value in Datatypes.all_types():
                return self.variable_declaration()
            if current_token.value == keywords["RETURN"]:
                return self.return_statement()
        
        elif current_token._type == TokenType.LITERAL:
            next_token = self.peek_token()
            if next_token and next_token.value == operators["ASSIGN"]:
                return self.variable_assignment()

        return None  # Explicitly return None if no statement was parsed


    

    def parse(self):
        
        while self.index < len(self.tokens): 
           node = self.parse_statement(inside_block=False)            
           self.nodes.append(node)

        return self.nodes
