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
        if token is not None:
            try:
                error_line = self.code.splitlines()[token.line - 1]
            except IndexError:
                error_line = "[ERROR: Line out of range]"
            
            # Ensure caret aligns with the column, even for lines shorter than the column number
            caret_position = " " * (min(token.column, len(error_line))) + "^"
            
            # More informative token printing (repr() gives more detailed information)
            print(f"Caused token: {repr(token)}")
            
            raise Exception(f"Syntax Error: {message} at line {token.line}, column {token.column}\n"
                            f"{error_line}\n"
                            f"{caret_position}")
        else:
            # Provide more context when token is None
            print("Syntax Error: Token is None, something went wrong during parsing. Please check the input.")


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
            self.syntax_error("Expected variable name", var_name)
        
        # Move to assignment operator
        next_token = self.next_token()
        if not next_token or next_token._type != TokenType.OPERATOR:
            self.syntax_error("Expected assignment operator", next_token)
        
        # Handle different assignment operators
        op = next_token.value
        
        # Move to value and parse expression
        self.next_token()  # Move past assignment operator
        value = self.parse_expression()
        
        # Create variable assignment node
        # For compound assignments (+=, -=, etc.), create a binary operation expression
        if op in [operators["ADD_ASSIGN"], operators["SUBTRACT_ASSIGN"], operators["MULTIPLY_ASSIGN"], 
                operators["DIVIDE_ASSIGN"], operators["MODULO_ASSIGN"]]:
            # Map the compound operator to its basic operator
            basic_op_map = {
                operators["ADD_ASSIGN"]: "+",
                operators["SUBTRACT_ASSIGN"]: "-",
                operators["MULTIPLY_ASSIGN"]: "*",
                operators["DIVIDE_ASSIGN"]: "/",
                operators["MODULO_ASSIGN"]: "%"
            }
            
            # Create a binary operation for a += b equivalent to a = a + b
            compound_value = ASTNode.ExpressionNode(
                NodeType.BINARY_OP,
                left=ASTNode.ExpressionNode(NodeType.LITERAL, value=var_name.value),
                right=value,
                op=basic_op_map[op]
            )
            
            node = ASTNode.VariableAssignment(var_name.value, compound_value)
        else:
            # Simple assignment (=)
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
        if current_token == None:
            self.syntax_error("Excepted semicolon", self.peek_token(-1))
        if not (current_token._type == TokenType.SEPARATOR and current_token.value == separators["SEMICOLON"]):
            self.syntax_error("Excepted semicolon", current_token)
        self.next_token() # Consume semicolon

    def parse_expression(self):
        return self.parse_binary_expression()

    def parse_binary_expression(self, precedence=0):
        # Define operator precedence
        precedence_map = {
            '+': 1, '-': 1,  # Addition, subtraction
            '*': 2, '/': 2, '%': 2,  # Multiplication, division, modulo
            '<': 3, '>': 3, '<=': 3, '>=': 3,  # Comparison
            '==': 4, '!=': 4,  # Equality
            '&&': 5,  # Logical AND
            '||': 6,  # Logical OR
        }
        
        # Start with parsing a primary expression
        left = self.parse_primary_expression()
        
        # Continue parsing binary operators as long as they have higher precedence
        while True:
            current = self.current_token()
            if not current or current._type != TokenType.OPERATOR:
                break
                
            op = current.value
            if op not in precedence_map or precedence_map[op] < precedence:
                break
                
            # Consume the operator
            self.next_token()
            
            # Parse the right side with higher precedence
            right = self.parse_binary_expression(precedence_map[op] + 1)
            
            # Create a binary operation node
            left = ASTNode.ExpressionNode(
                NodeType.BINARY_OP,
                left=left,
                right=right,
                op=op
            )
            
        return left

    def parse_primary_expression(self):
        current = self.current_token()
        
        if not current:
            self.syntax_error("Unexpected end of input during expression parsing", self.peek_token(-1))
            
        # Handle parenthesized expressions
        if current._type == TokenType.SEPARATOR and current.value == separators["LPAREN"]:
            self.next_token()  # Consume the '('
            expr = self.parse_binary_expression()  # Parse the inner expression
            
            # Expect closing parenthesis
            current = self.current_token()
            if not current or current.value != separators["RPAREN"]:
                self.syntax_error("Expected closing parenthesis ')'", current)
            
            self.next_token()  # Consume the ')'
            return expr
            
        # Handle literals and identifiers
        elif current._type == TokenType.LITERAL:
            node = ASTNode.ExpressionNode(
                NodeType.LITERAL, 
                value=current.value
            )
            self.next_token()  # Consume the token
            return node
            
        # Handle unary operators
        elif current._type == TokenType.OPERATOR and current.value in ['+', '-', '!']:
            op = current.value
            self.next_token()  # Consume the operator
            operand = self.parse_primary_expression()
            return ASTNode.ExpressionNode(
                NodeType.UNARY_OP,
                left=operand,
                op=op
            )
            
        else:
            self.syntax_error("Unexpected token in expression", current)

    def block(self):
        opening_token = self.current_token()  # Remember where the block started for better error reporting
        self.next_token()  # Consume the opening brace
        nodes = []
        
        while self.current_token() and self.current_token().value != separators["RBRACE"]:
            try:
                node = self.parse_statement(inside_block=True)
                if node:
                    nodes.append(node)
            except Exception as e:
                # Simple error recovery: skip to the next semicolon or right brace
                print(f"Error in block: {e}")
                while (self.current_token() and 
                    self.current_token().value not in [separators["SEMICOLON"], separators["RBRACE"]]):
                    self.next_token()
                if self.current_token() and self.current_token().value == separators["SEMICOLON"]:
                    self.next_token()  # Skip the semicolon
        
        # Check for rbrace
        print(self.current_token())
        if not self.current_token() or self.current_token().value != separators["RBRACE"]:
            self.syntax_error(f"Expected `}}` to close block that started at line {opening_token.line}", self.current_token() or self.tokens[-1])
            # No recovery possible here, the block is incomplete
            return ASTNode.Block(nodes)
        
        self.next_token()  # Consume the right brace
        return ASTNode.Block(nodes)
    
    def function_declaration(self):
        func_name = str()
        func_return_type = None
        parameters = []
        body = ASTNode.Block 

        func_return_type  = self.current_token().value

        # Get name

        next_token = self.next_token()
        if next_token._type != TokenType.LITERAL:
            self.syntax_error("Expected function name", next_token)
        func_name = next_token.value
        
        # Parse parameters

        # Move to opening parenthesis '('
        next_token = self.next_token()
        if not next_token or next_token.value != separators["LPAREN"]:
            self.syntax_error("Excepted '('", next_token)

        peek_token = self.peek_token()
        if peek_token and peek_token.value == separators["RPAREN"]:
            self.next_token()  # Consume the ')'
        else:
            while True:
                param_type = self.next_token()
                if not param_type or param_type._type != TokenType.KEYWORD:
                    self.syntax_error("Expected parameter type", param_type)
                if not (param_type.value in Datatypes.all_types()):
                    self.syntax_error("Invalid parameter type", param_type)
                
                param_name = self.next_token()
                if not param_name or param_name._type != TokenType.LITERAL:
                    self.syntax_error("Expected parameter name", param_name)

                # Check for default value
                default_value = None
                peek_token = self.peek_token()
                if peek_token and peek_token.value == operators["ASSIGN"]:
                    self.next_token()  # Consume '='
                    default_value_token = self.next_token()
                    if not default_value_token or default_value_token._type != TokenType.LITERAL:
                        self.syntax_error("Expected default value", default_value_token)
                    default_value = default_value_token.value                    
                
                # Create parameter variable declaration
                param = ASTNode.VariableDeclaration(param_type.value, param_name.value, default_value)
                parameters.append(param)
                
                next_token = self.next_token()
                if next_token.value == separators["RPAREN"]:
                    break
                elif next_token.value != separators["COMMA"]:
                    self.syntax_error("Expected ',' or ')'", next_token)

        # Parse body (block)
        next_token = self.next_token()
        if next_token is None or next_token.value not in [separators["SEMICOLON"], separators["LBRACE"]]:
            self.syntax_error("Expected ';' or '{'", next_token)
        body = self.block()

        return ASTNode.FunctionDefinition(func_name, func_return_type, body, parameters)

    def if_statement(self):
        # Parse condition
        next_token = self.next_token()
        if not (next_token._type == TokenType.SEPARATOR and next_token.value == separators["LPAREN"]):
            self.syntax_error("Expected expression", next_token)
        
        condition = self.parse_expression()
        
        # Check for closing parenthesis
        next_token = self.current_token()
        if not (next_token._type == TokenType.SEPARATOR and next_token.value == separators["LBRACE"]):
            self.syntax_error("Expected closing parenthesis", next_token)

        block = self.block()
        else_block = None
        
        # Check for optional else
        next_token = self.current_token()
        if next_token and next_token._type == TokenType.KEYWORD and next_token.value == keywords["ELSE"]:
            self.next_token() # consume else
        
            # Check if it's an else-if or a simple else
            peek_token = self.current_token()
            if peek_token and peek_token._type == TokenType.KEYWORD and peek_token.value == keywords["IF"]:
                # Handle else-if by recursively calling if_statement
                self.next_token()  # Consume the 'if' token
                else_block = [self.if_statement()]
            else:
                # Parse simple else block
                next_token = self.current_token()
                if not (next_token._type == TokenType.SEPARATOR and next_token.value == separators["LBRACE"]):
                    self.syntax_error("Expected '{' to start else block", next_token)
                
                else_block = self.block()
        
        return ASTNode.IfStatement(condition, block, else_block)
    
    def while_loop(self):
        self.next_token()
        condition = self.parse_expression()
        body = self.block()
        return ASTNode.WhileLoop(condition, body)

    def for_loop(self):
        # Check lparen
        next_token = self.next_token()
        if not (next_token._type == TokenType.SEPARATOR and next_token.value == separators["LPAREN"]):
            self.syntax_error("Expected '(' ", next_token)
        
        # parse init  
        self.next_token()  
        init = None
        if self.current_token()._type == TokenType.KEYWORD:
            init = self.variable_declaration()
        else:
            init = self.variable_assignment()
        
        # parse condition
        condition = self.parse_expression()
        
        # check for semicolon 
        if not self.current_token() or self.current_token().value != separators["SEMICOLON"]:
            self.syntax_error("Expected ';' after for loop condition", self.current_token())
        self.next_token()  # consume  
        
        # parse update
        update = None
        if self.current_token()._type == TokenType.LITERAL:
            # assigment 
            update_var = self.current_token().value
            self.next_token()  
            
            if self.current_token()._type == TokenType.OPERATOR:
                op = self.current_token().value
                # Handle compound assignment operators (+=, -=, etc.)
                if op in [operators["ADD_ASSIGN"], operators["SUBTRACT_ASSIGN"], operators["MULTIPLY_ASSIGN"], 
                        operators["DIVIDE_ASSIGN"], operators["MODULO_ASSIGN"]]:
                    self.next_token()  
                    value = self.parse_expression()
                    update = ASTNode.VariableAssignment(update_var, value)
                # Handle increment/decrement (++, --)
                elif op in [operators["increment"], operators["decrement"]]:
                    self.next_token()  # Move past operator
                    # Create a simple increment/decrement expression
                    value = ASTNode.ExpressionNode(
                        NodeType.BINARY_OP,
                        left=ASTNode.ExpressionNode(NodeType.LITERAL, value=update_var),
                        right=ASTNode.ExpressionNode(NodeType.LITERAL, value="1"),
                        op="+" if op == operators["increment"] else "-"
                    )
                    update = ASTNode.VariableAssignment(update_var, value)
        
        #  check for rparen
        if not self.current_token() or self.current_token().value != separators["RPAREN"]:
            self.syntax_error("Expected ')' after for loop update", self.current_token())
        self.next_token()  # Consume right parenthesis
        
        # parse body 
        body = self.block()
        
        return ASTNode.ForLoop(init, condition, update, body)

    def parse_statement(self, inside_block: bool = False) -> ASTNode:
        current_token = self.current_token()
        if not current_token:
            return None

        if current_token._type == TokenType.KEYWORD:
            next_token = self.peek_token()
            if current_token.value in Datatypes.all_types():
                if next_token._type == TokenType.LITERAL and self.peek_token(2).value == separators["LPAREN"]:
                    return self.function_declaration()
                else: 
                    return self.variable_declaration()
            if current_token.value == keywords["RETURN"]:
                return self.return_statement()
            if current_token.value == keywords["IF"]:
                return self.if_statement()
            if current_token.value == keywords["WHILE"]:
                return self.while_loop()
            if current_token.value == keywords["FOR"]:
                return self.for_loop()


        if current_token._type == TokenType.SEPARATOR:
            if current_token.value == separators["LBRACE"]:
                return self.block()

        elif current_token._type == TokenType.LITERAL:
            next_token = self.peek_token()
            if next_token and next_token.value == operators["ASSIGN"]:
                return self.variable_assignment()
        
        if current_token._type == TokenType.COMMENT:
            return self.comment()

        self.syntax_error("Unexpected statement", current_token)


    

    def parse(self):
        
        while self.index < len(self.tokens): 
            node = self.parse_statement(inside_block=False)
            if node:
                self.nodes.append(node)

        return self.nodes
