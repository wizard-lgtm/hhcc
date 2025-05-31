from lexer import *
from typing import List 
from astnodes import *  
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from compiler import Compiler  # Only for type hints
class ASTParser: 
    code: str
    index: int
    tokens: List[Token]
    nodes: List[ASTNode]
    def __init__(self, code: str, compiler: "Compiler"):
        self.code = code
        self.index = 0
        self.tokens = []
        self.nodes = []
        self.compiler = compiler

    def load_tokens(self, tokens: List[Token]):
        self.tokens = tokens

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


    def variable_declaration(self, user_typed=False):
 
        var_type = self.current_token()
        is_pointer = False
    
        peek_token = self.peek_token()
        # Check the next token is a pointer
        if peek_token and peek_token._type == TokenType.OPERATOR and peek_token.value == operators["POINTER"]:
            self.next_token()
            is_pointer = True
        
        
        # Move to variable name
        var_name = self.next_token()
        if not (var_name or var_name._type != TokenType.LITERAL):
            self.syntax_error("Expected variable name", var_name)


        # Check if this is an array declaration by looking ahead
        peek_token = self.peek_token()
        if peek_token and peek_token.value == separators["LBRACKET"]:
            # Array declaration - let's consume the variable name first
            self.next_token()  # Consume variable name
            return self.array_declaration(var_type.value, var_name.value, user_typed)
        
        # Regular variable declaration continues...
        node = ASTNode.VariableDeclaration(var_type.value, var_name.value, None, user_typed, is_pointer)
        
        # Move to next token to check assignment or semicolon
        next_token = self.next_token()
        
        # Check for assignment
        if next_token and next_token.value == operators["ASSIGN"]:

            # Move to value and parse expression
            self.next_token()  # Move past '='
            node.value = self.parse_expression()
        
        # Check semicolon
        self.check_semicolon()
        
        return node
    def variable_assignment(self):

        
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


        expression = None
        if self.current_token().value == keywords["RETURN"]:
            self.next_token()  # Skip the 'return' keyword
            if self.current_token() and self.current_token().value != separators["SEMICOLON"]:
                expression = self.parse_expression()
        
        # Check semicolon 
        self.check_semicolon()
        return ASTNode.Return(expression)

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
            '|': 3, '&': 3, '^': 3,  # Bitwise OR, AND, XOR
            '<<': 3, '>>': 3,        # Bitwise shifts
            '<': 4, '>': 4, '<=': 4, '>=': 4,  # Comparison
            '==': 5, '!=': 5,  # Equality
            '&&': 6,  # Logical AND
            '||': 7,  # Logical OR
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
            
        # Handle references (& operator)
        if current._type == TokenType.OPERATOR and current.value == '&':
            self.next_token()  # Consume the '&'
            operand = self.parse_primary_expression()
        
            return ASTNode.ExpressionNode(
                NodeType.REFERENCE,
                left=operand,
                op='&'
            )

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
            
            # Check for function calls, array indexing, or struct access
            while self.current_token():
                # Handle function calls with parentheses
                if self.current_token()._type == TokenType.SEPARATOR and self.current_token().value == separators["LPAREN"]:
                    args = []
                    self.next_token()  # Consume the '('
                    
                    # Parse arguments if there are any
                    if self.current_token() and self.current_token().value != separators["RPAREN"]:
                        while True:
                            args.append(self.parse_expression())
                            
                            # Check for comma or closing parenthesis
                            if not self.current_token():
                                self.syntax_error("Unexpected end of input during function call", self.peek_token(-1))
                            
                            if self.current_token().value == separators["RPAREN"]:
                                break  # Exit the loop
                            elif self.current_token().value != separators["COMMA"]:
                                self.syntax_error("Expected ',' or ')' in function arguments", self.current_token())
                            
                            self.next_token()  # Consume the comma
                    
                    # Expect closing parenthesis
                    if not self.current_token() or self.current_token().value != separators["RPAREN"]:
                        self.syntax_error("Expected closing parenthesis ')'", self.current_token())
                    
                    self.next_token()  # Consume the ')'
                    
                    # Create a function call expression node
                    node = ASTNode.ExpressionNode(
                        NodeType.FUNCTION_CALL,
                        left=node,  # Function name
                        right=args,  # Arguments list
                        op="()"  # Use () to denote function call
                    )
                    
                # Handle array indexing with brackets
                elif self.current_token()._type == TokenType.SEPARATOR and self.current_token().value == separators["LBRACKET"]:
                    self.next_token()  # Consume the '['
                    index_expr = self.parse_expression()  # Parse the index expression
                    
                    # Expect closing bracket
                    if not self.current_token() or self.current_token().value != separators["RBRACKET"]:
                        self.syntax_error("Expected closing bracket ']'", self.current_token())
                    
                    self.next_token()  # Consume the ']'
                    
                    # Create an array access node
                    node = ASTNode.ExpressionNode(
                        NodeType.ARRAY_ACCESS,
                        left=node,  # The array identifier
                        right=index_expr,  # The index expression
                        op="[]"  # Use a special operator to denote array access
                    )
                
                # Handle struct field access with dot
                elif self.current_token()._type == TokenType.SEPARATOR and self.current_token().value == separators["DOT"]:
                    self.next_token()  # Consume the '.'
                    
                    # Expect field name
                    if not self.current_token() or self.current_token()._type != TokenType.LITERAL:
                        self.syntax_error("Expected field name after '.'", self.current_token())
                    
                    field_name = self.current_token().value
                    self.next_token()  # Consume the field name
                    
                    # Create a struct access node
                    node = ASTNode.ExpressionNode(
                        NodeType.STRUCT_ACCESS,
                        left=node,  # The struct identifier
                        right=ASTNode.ExpressionNode(NodeType.LITERAL, value=field_name),  # The field name
                        op="."  # Use dot to denote struct access
                    )
                else:
                    break  # Exit loop if no more function calls, array indexing, or struct access
            
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
                self.syntax_error(f"Error in block: {e}\n", self.current_token())
                while (self.current_token() and 
                    self.current_token().value not in [separators["SEMICOLON"], separators["RBRACE"]]):
                    self.next_token()
                if self.current_token() and self.current_token().value == separators["SEMICOLON"]:
                    self.next_token()  # Skip the semicolon
        
        # Check for rbrace
   
        if not self.current_token() or self.current_token().value != separators["RBRACE"]:
            self.syntax_error(f"Expected `}}` to close block that started at line {opening_token.line}", self.current_token() or self.tokens[-1])
            # No recovery possible here, the block is incomplete
            return ASTNode.Block(nodes)
        
        self.next_token()  # Consume the right brace
    
        return ASTNode.Block(nodes)
    def function_declaration(self): 
        func_name = "" 
        func_return_type = None 
        parameters = [] 
        body = None
        has_variadic_args = False 
        
        # Return type 
        func_return_type = self.current_token().value 
        
        # Get function name 
        next_token = self.next_token() 
        if next_token._type != TokenType.LITERAL: 
            self.syntax_error("Expected function name", next_token) 
        func_name = next_token.value 
        
        # Move to '(' 
        next_token = self.next_token() 
        if not next_token or next_token.value != separators["LPAREN"]: 
            self.syntax_error("Expected '('", next_token) 
        
        while True: 
            peek_token = self.peek_token() 
            
            # Handle variadic: "..." 
            if peek_token and peek_token._type == TokenType.OPERATOR and peek_token.value == separators["THREEDOTS"]: 
                self.next_token() # Consume '...' 
                has_variadic_args = True 
                next_token = self.next_token() 
                if not next_token or next_token.value != separators["RPAREN"]: 
                    self.syntax_error("Expected ')' after '...'", next_token) 
                break 
            
            # End of parameter list 
            if peek_token and peek_token.value == separators["RPAREN"]: 
                self.next_token() # Consume ')' 
                break 
            
            # Parse typed parameter 
            param_type = self.next_token() 
            if not param_type or param_type._type != TokenType.KEYWORD: 
                self.syntax_error("Expected parameter type", param_type) 
            if not (param_type.value in Datatypes.all_types()): 
                self.syntax_error("Invalid parameter type", param_type) 
            
            # Pointer check 
            is_pointer = False 
            peek_token = self.peek_token() 
            if peek_token and peek_token._type == TokenType.OPERATOR and peek_token.value == operators["POINTER"]: 
                self.next_token() # Consume '*' 
                is_pointer = True 
            
            # Parameter name 
            param_name = self.next_token() 
            if not param_name or param_name._type != TokenType.LITERAL: 
                self.syntax_error("Expected parameter name", param_name) 
            
            # Default value check 
            default_value = None 
            peek_token = self.peek_token() 
            if peek_token and peek_token.value == operators["ASSIGN"]: 
                self.next_token() # Consume '=' 
                default_value_token = self.next_token() 
                if not default_value_token or default_value_token._type != TokenType.LITERAL: 
                    self.syntax_error("Expected default value", default_value_token) 
                default_value = default_value_token.value 
            
            # Append parameter 
            param = ASTNode.VariableDeclaration( 
                param_type.value, 
                param_name.value, 
                default_value, 
                False, 
                is_pointer 
            ) 
            parameters.append(param) 
            
            # Comma or end of parameters 
            next_token = self.next_token() 
            if next_token.value == separators["RPAREN"]: 
                break 
            elif next_token.value != separators["COMMA"]: 
                self.syntax_error("Expected ',' or ')'", next_token) 
        
        # Function body or declaration 
        next_token = self.peek_token()  # Just peek to check what's next
        if next_token is None:
            self.syntax_error("Expected ';' or '{'", next_token)
        
        if next_token.value == separators["LBRACE"]:
            self.next_token()  # Consume '{'
            body = self.block() 
        elif next_token.value == separators["SEMICOLON"]:
            self.next_token()  # Consume ';'
            self.next_token()  # Consume ';'
            body = None
        else:
            self.syntax_error("Expected ';' or '{'", next_token)
        
        return ASTNode.FunctionDefinition(func_name, func_return_type, body, parameters, has_variadic_args)
        
    def if_statement(self):

        # Parse condition
        next_token = self.next_token()
        if not (next_token._type == TokenType.SEPARATOR and next_token.value == separators["LPAREN"]):
            self.syntax_error("Expected '(' after if", next_token)
        
        self.next_token()  # Consume the opening parenthesis
        condition = self.parse_expression()
        
        # Check for closing parenthesis
        if not self.current_token() or self.current_token().value != separators["RPAREN"]:
            self.syntax_error("Expected closing parenthesis ')'", self.current_token())
        
        self.next_token()  # Consume the closing parenthesis
        
        # Parse the if block
        if not self.current_token() or self.current_token().value != separators["LBRACE"]:
            self.syntax_error("Expected '{' to start if block", self.current_token())
        
        block = self.block()
        else_block = None
        
        # Check for optional else
        if self.current_token() and self.current_token()._type == TokenType.KEYWORD and self.current_token().value == keywords["ELSE"]:
            self.next_token()  # consume else
        
            # Check if it's an else-if or a simple else
            if self.current_token() and self.current_token()._type == TokenType.KEYWORD and self.current_token().value == keywords["IF"]:
                # Handle else-if by recursively calling if_statement
                else_block = self.if_statement()
            else:
                # Parse simple else block
                if not self.current_token() or self.current_token().value != separators["LBRACE"]:
                    self.syntax_error("Expected '{' to start else block", self.current_token())
                
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

    def comment(self):
        value = self.current_token().value
        # Determine if it's an inline comment (e.g., starts with //)
        is_inline = value.strip().startswith("//")
        node = ASTNode.Comment(text=value, is_inline=is_inline)
        self.next_token()
        return node

    def function_call(self):
        # Get function name
        func_name = self.current_token().value
        
        # Check if there are parentheses for arguments
        has_parentheses = False
        args = []
        
        next_token = self.peek_token()
        if next_token and next_token.value == separators["LPAREN"]:
            has_parentheses = True
            self.next_token()  # Consume function name and move to '('
            
            # Check if there are any arguments
            peek_token = self.peek_token()
            if peek_token and peek_token.value != separators["RPAREN"]:
                self.next_token()  # Consume the '('
                
                # Parse arguments until we hit the closing parenthesis
                while True:
                    arg = self.parse_expression()
                    args.append(arg)
                    
                    # Check for comma or closing parenthesis
                    if not self.current_token():
                        self.syntax_error("Unexpected end of input during argument parsing", self.peek_token(-1))
                        
                    if self.current_token().value == separators["RPAREN"]:
                        break  # Exit the loop
                    elif self.current_token().value != separators["COMMA"]:
                        self.syntax_error("Expected ',' or ')' in arguments", self.current_token())
                    
                    self.next_token()  # Consume the comma
            else:
                self.next_token()  # Consume the '('
            
            # Check for closing parenthesis
            if not self.current_token() or self.current_token().value != separators["RPAREN"]:
                self.syntax_error("Expected ')' to close function call", self.current_token())
            
            self.next_token()  # Consume the ')'
        else:
            # No parentheses, just consume the function name
            self.next_token()
        
        # Check for semicolon
        self.check_semicolon()
        
        return ASTNode.FunctionCall(func_name, args, has_parentheses)   

    def class_declaration(self):
        class_name = None
        parent_class = None
        fields = []
        
        # Parse class name
        next_token = self.next_token()
        if not (next_token._type == TokenType.LITERAL):
            self.syntax_error("Expected class name", next_token)
        class_name = next_token.value

        # Check for inheritance (colon followed by parent class name)
        peek_token = self.peek_token()
        if peek_token and peek_token._type == TokenType.SEPARATOR and peek_token.value == separators["COLON"]:
            self.next_token()  # Consume the colon
            
            # Get parent class name
            parent_token = self.next_token()
            if not parent_token or parent_token._type != TokenType.LITERAL:
                self.syntax_error("Expected parent class name after ':'", parent_token)
            parent_class = parent_token.value

        # Parse block start (expect '{')
        next_token = self.next_token()
        if not (next_token._type == TokenType.SEPARATOR and next_token.value == separators["LBRACE"]):
            self.syntax_error("Expected '{'", next_token)

        # Consume the opening brace
        self.next_token()
        
        # Parse fields until we hit the closing brace
        while self.current_token() and self.current_token().value != separators["RBRACE"]:
            # Parse field declaration
            var = self.variable_declaration()
            fields.append(var)
        
        # Check if we found the closing brace
        if not self.current_token() or self.current_token().value != separators["RBRACE"]:
            self.syntax_error("Expected '}' to close class declaration", self.current_token() or self.tokens[-1])
        
        # Consume the closing brace
        self.next_token()
        
        # Check for semicolon
        self.check_semicolon()

        # Modify your ASTNode.Class constructor to accept parent_class
        return ASTNode.Class(class_name, fields, parent_class)

    def union_declaration(self):
        name = None
        fields = []

        # Parse class name
        next_token = self.next_token()
        if not (next_token._type == TokenType.LITERAL):
            self.syntax_error("Expected union name", next_token)
        name = next_token.value

        # Parse block start (expect '{')
        next_token = self.next_token()
        if not (next_token._type == TokenType.SEPARATOR and next_token.value == separators["LBRACE"]):
            self.syntax_error("Expected '{'", next_token)

        self.next_token()

        # Parse fields until we hit the closing brace
        while self.current_token() and self.current_token().value != separators["RBRACE"]:
            # Parse field declaration
            if self.current_token()._type == TokenType.KEYWORD and self.current_token().value in Datatypes.all_types():
                var = self.variable_declaration()
                fields.append(var)
            else:
                self.syntax_error("Expected field declaration", self.current_token())
        
        # Check if we found the closing brace
        if not self.current_token() or self.current_token().value != separators["RBRACE"]:
            self.syntax_error("Expected '}' to close class declaration", self.current_token() or self.tokens[-1])
        
        # Consume the closing brace
        self.next_token()
        
        # Check for semicolon
        self.check_semicolon()

        return ASTNode.Union(name, fields)

    def break_statement(self):
        # consume the break keyword
        self.next_token()
        
        # semicolon
        self.check_semicolon()
        
        return ASTNode.Break()

    def continue_statement(self):
        # soncume the continue keyword
        self.next_token()
        
        # semicolon
        self.check_semicolon()
        
        return ASTNode.Continue()
    
    def array_declaration(self, base_type, name, user_typed=False):
        dimensions = []
        
        # Parse all dimensions
        while self.peek_token() and self.peek_token().value == separators["LBRACKET"]:
            self.next_token()  # Consume '['
            
            # Check if this dimension is specified or empty
            if self.peek_token() and self.peek_token().value == separators["RBRACKET"]:
                # Empty dimension like arr[]
                dimensions.append(None)
                self.next_token()  # Consume ']'
            else:
                # Parse dimension expression
                self.next_token()  # Move past '['
                dimension_expr = self.parse_expression()
                dimensions.append(dimension_expr)
                
                # Expect closing bracket
                if not self.current_token() or self.current_token().value != separators["RBRACKET"]:
                    self.syntax_error("Expected ']'", self.current_token())
                
                self.next_token()  # Consume ']'
        
        # Check for initialization
        initialization = None
        if self.current_token() and self.current_token().value == operators["ASSIGN"]:
            self.next_token()  # Consume '='
            
            # Parse array initialization
            if self.current_token() and self.current_token().value == separators["LBRACE"]:
                initialization = self.parse_array_initialization()
            else:
                self.syntax_error("Expected '{' for array initialization", self.current_token())
        
        # Check semicolon
        self.check_semicolon()
        
        return ASTNode.ArrayDeclaration(base_type, name, dimensions, initialization)
    

    def array_declaration(self, base_type, name, user_typed=False):
        dimensions = []
        
        # Parse all dimensions
        while self.current_token() and self.current_token().value == separators["LBRACKET"]:
            self.next_token()  # Move past '['
            
            # Check if this dimension is specified or empty
            if self.current_token() and self.current_token().value == separators["RBRACKET"]:
                # Empty dimension like arr[]
                dimensions.append(None)
            else:
                # Parse dimension expression
                dimension_expr = self.parse_expression()
                dimensions.append(dimension_expr)
                
                # Expect closing bracket
                if not self.current_token() or self.current_token().value != separators["RBRACKET"]:
                    self.syntax_error("Expected ']'", self.current_token())
            
            self.next_token()  # Consume ']'
        
        # Check for initialization
        initialization = None
        if self.current_token() and self.current_token().value == operators["ASSIGN"]:
            self.next_token()  # Consume '='
            
            # Parse array initialization
            if self.current_token() and self.current_token().value == separators["LBRACE"]:
                initialization = self.parse_array_initialization()
            else:
                self.syntax_error("Expected '{' for array initialization", self.current_token())
        
        # Check semicolon
        self.check_semicolon()
        
        return ASTNode.ArrayDeclaration(base_type, name, dimensions, initialization)

    
    def parse_array_initialization(self):

        
        if not self.current_token() or self.current_token().value != separators["LBRACE"]:
            self.syntax_error("Expected '{' to start array initialization", self.current_token())
        
        self.next_token()  # Consume '{'
        
        elements = []
        
        # Empty initialization case like { }
        if self.current_token() and self.current_token().value == separators["RBRACE"]:
            self.next_token()  # Consume '}'
            return ASTNode.ArrayInitialization(elements)
        
        # Parse elements separated by commas
        while True:
            # Check for nested array initialization
            if self.current_token() and self.current_token().value == separators["LBRACE"]:
                elements.append(self.parse_array_initialization())
            else:
                # Parse regular expression element
                elements.append(self.parse_expression())
            
            # Check for comma or closing brace
            if not self.current_token():
                self.syntax_error("Unexpected end of input during array initialization", self.peek_token(-1))
                
            if self.current_token().value == separators["RBRACE"]:
                self.next_token()  # Consume '}'
                break  # End of initialization
            elif self.current_token().value != separators["COMMA"]:
                self.syntax_error("Expected ',' or '}' in array initialization", self.current_token())
            else:
                self.next_token()  # Consume comma
        
        return ASTNode.ArrayInitialization(elements)

    def parse_statement(self, inside_block: bool = False) -> ASTNode:
        current_token = self.current_token()
        if not current_token:
            return None
        
        # Handle extern declarations
        if current_token._type == TokenType.KEYWORD and current_token.value == keywords["EXTERN"]:
            return self.parse_extern_declaration()
        
        if current_token._type == TokenType.KEYWORD:
            next_token = self.peek_token()
            
            # Handle function declarations and definitions
            if current_token.value in Datatypes.all_types():
                # Check for pointer type first
                if next_token._type == TokenType.OPERATOR and next_token.value == operators["POINTER"]:
                    # Look ahead one more token to see the variable name
                    peek_token_2 = self.peek_token(2)
                    if peek_token_2 and peek_token_2._type == TokenType.LITERAL:
                        # Check if it's a function (look for opening parenthesis after variable name)
                        peek_token_3 = self.peek_token(3)
                        if peek_token_3 and peek_token_3.value == separators["LPAREN"]:
                            return self.function_declaration()
                        else:
                            return self.variable_declaration()
                elif next_token._type == TokenType.LITERAL:
                    # Check if it's potentially a function
                    peek_token_2 = self.peek_token(2)
                    if peek_token_2 and peek_token_2.value == separators["LPAREN"]:
                        return self.function_declaration()
                    else:
                        return self.variable_declaration()
            
            # Add inline assembly supportAdd commentMore actions
            if current_token.value == keywords.get("ASM", "asm"):
                self.next_token()  # Consume 'asm'
                return self.inline_assembly()

            if current_token.value == keywords["RETURN"]:
                return self.return_statement()
            if current_token.value == keywords["IF"]:
                return self.if_statement()
            if current_token.value == keywords["WHILE"]:
                return self.while_loop()
            if current_token.value == keywords["FOR"]:
                return self.for_loop()
            if current_token.value == keywords["CLASS"]:
                return self.class_declaration()
            if current_token.value == keywords["UNION"]:
                return self.union_declaration()
            if current_token.value == keywords["BREAK"]:
                return self.break_statement()
            if current_token.value == keywords["CONTINUE"]:
                return self.continue_statement()
        
        if current_token._type == TokenType.SEPARATOR:
            if current_token.value == separators["LBRACE"]:
                return self.block()
        
        elif current_token._type == TokenType.LITERAL:
            next_token = self.peek_token()
            if current_token.value in Datatypes.user_defined_types:
                return self.variable_declaration(True)
            elif next_token and next_token._type == TokenType.OPERATOR and next_token.value in assignment_operators.values(): 
                return self.variable_declaration()
            elif (next_token and next_token._type == TokenType.SEPARATOR and next_token.value == separators["LPAREN"]) or \
                (next_token and next_token._type == TokenType.SEPARATOR and next_token.value == separators["SEMICOLON"]):
                # Function call with parentheses or without parentheses
                return self.function_call()
            elif next_token and next_token._type == TokenType.SEPARATOR and next_token.value == separators["DOT"]:
                # Struct field assignment: t.a = 1;
                return self.struct_field_assignment()
        
        if current_token._type == TokenType.COMMENT:
            return self.comment()
        
        elif self.current_token()._type == TokenType.LITERAL and self.peek_token(1).value == operators["increment"]:
            return self.variable_increment()
        elif self.current_token()._type == TokenType.LITERAL and self.peek_token(1).value == operators["decrement"]:
            return self.variable_decrement()
        
        self.syntax_error("Unexpected statement", current_token) 


                        
    def parse_extern_declaration(self) -> ASTNode:
        """
        Parse extern declarations for variables, functions, and struct forward declarations.
        Returns an Extern node that wraps the actual declaration.
        """
        self.next_token()  # Consume 'extern'
        current_token = self.current_token()
        if not current_token:
            self.syntax_error("Expected statement after 'extern'", current_token)
        
        # Handle extern struct declarations
        if current_token._type == TokenType.KEYWORD and current_token.value == keywords["CLASS"]:
            # Use the existing class declaration parser
            # For a forward declaration, this should handle the struct name and semicolon
            struct_decl = self.class_declaration()
            return ASTNode.Extern(struct_decl)
        
        # Handle extern variable declarations (with explicit type)
        elif current_token._type == TokenType.KEYWORD and current_token.value in Datatypes.all_types():
            # Check if it's potentially a function
            next_token = self.peek_token()
            if next_token and next_token._type == TokenType.LITERAL:
                peek_token_2 = self.peek_token(2)
                if peek_token_2 and peek_token_2.value == separators["LPAREN"]:
                    # It's a function declaration with explicit return type
                    func_decl = self.function_declaration()
                    return ASTNode.Extern(func_decl)
                else:
                    # It's a variable declaration
                    var_decl = self.variable_declaration(user_typed=True)
                    return ASTNode.Extern(var_decl)
        
        # Handle extern function declarations without explicit return type
        elif current_token._type == TokenType.LITERAL:
            # Check if this looks like a function (identifier followed by parenthesis)
            next_token = self.peek_token()
            if next_token and next_token.value == separators["LPAREN"]:
                self.index -= 1 # Move back to the function name token
                func_decl = self.function_declaration()
                return ASTNode.Extern(func_decl)
        
        # If we reach here, it's an error
        self.syntax_error("Expected type, 'struct', or function name after 'extern'", self.current_token())
    

    def variable_decrement(self):
        node =  ASTNode.VariableDecrement(self.current_token().value)
        self.next_token()  # Consume the variable name
        self.next_token()  # Consume the variable name
        self.check_semicolon()
        return node

    def struct_field_assignment(self):
        # Start with the struct name (already consumed in parse_statement)
        struct_name = self.current_token().value
        
        # Consume the struct name and move to the dot
        next_token = self.next_token()
        if not next_token or next_token.value != separators["DOT"]:
            self.syntax_error("Expected '.' for struct field access", next_token)
        
        # Get the field name
        field_name = self.next_token()
        if not field_name or field_name._type != TokenType.LITERAL:
            self.syntax_error("Expected field name after '.'", field_name)
        
        # Move to assignment operator
        next_token = self.next_token()
        if not next_token or next_token._type != TokenType.OPERATOR or next_token.value not in assignment_operators.values():
            self.syntax_error("Expected assignment operator", next_token)
        
        # Parse the value expression
        self.next_token()  # Move past assignment operator
        value = self.parse_expression()
        
        # Check semicolon
        self.check_semicolon()
        
        # Since we might not have NodeType.STRUCT_FIELD_ASSIGNMENT defined yet,
        # we can use the existing VariableAssignment with a composite name
        # or handle it specially in the code generator later
        access_expr = ASTNode.ExpressionNode(
            NodeType.STRUCT_ACCESS,
            left=ASTNode.ExpressionNode(NodeType.LITERAL, value=struct_name),
            right=ASTNode.ExpressionNode(NodeType.LITERAL, value=field_name.value),
            op="."
        )
        
        return ASTNode.VariableAssignment(f"{struct_name}.{field_name.value}", value)
        

    def inline_assembly(self):
        """
        Parse GCC-style inline assembly statements
        Syntax: asm [volatile] ( assembly_template : output_operands : input_operands : clobbers );
        """
        # 'asm' keyword should already be consumed by parse_statement
        is_volatile = False
        
        # Check for optional 'volatile' keyword
        if (self.current_token()._type == TokenType.KEYWORD and 
            self.current_token().value == keywords.get("VOLATILE")):
            is_volatile = True
            self.next_token()
        
        # Expect opening parenthesis
        if (not self.current_token() or 
            self.current_token().value != separators["LPAREN"]):
            self.syntax_error("Expected '(' after 'asm'", self.current_token())
        
        self.next_token()  # Consume '('
        
        # Parse assembly template (required)
        assembly_code = self.parse_assembly_template()
        
        # Initialize constraint lists
        output_constraints = []
        input_constraints = []
        clobber_list = []
        
        # Check for output constraints (optional)
        if (self.current_token() and 
            self.current_token().value == separators["COLON"]):
            self.next_token()  # Consume ':'
            output_constraints = self.parse_constraint_list()
            
            # Check for input constraints (optional)
            if (self.current_token() and 
                self.current_token().value == separators["COLON"]):
                self.next_token()  # Consume ':'
                input_constraints = self.parse_constraint_list()
                
                # Check for clobber list (optional)
                if (self.current_token() and 
                    self.current_token().value == separators["COLON"]):
                    self.next_token()  # Consume ':'
                    clobber_list = self.parse_clobber_list()
        
        # Expect closing parenthesis
        if (not self.current_token() or 
            self.current_token().value != separators["RPAREN"]):
            self.syntax_error("Expected ')' to close inline assembly", self.current_token())
        
        self.next_token()  # Consume ')'
        
        # Check for semicolon
        self.check_semicolon()
        
        return ASTNode.InlineAsm(
            assembly_code=assembly_code,
            output_constraints=output_constraints,
            input_constraints=input_constraints,
            clobber_list=clobber_list,
            is_volatile=is_volatile
        )        

    def parse_assembly_template(self):
        """
        Parse the assembly template string, which can be:
        - A single string literal
        - Multiple concatenated string literals
        """
        assembly_parts = []
        
        while (self.current_token() and 
            self.current_token()._type == TokenType.LITERAL):
            # Remove quotes from string literal
            string_value = self.current_token().value.strip('"\'')
            assembly_parts.append(string_value)
            self.next_token()
        
        if not assembly_parts:
            self.syntax_error("Expected assembly template string", self.current_token())
        
        # Join multiple string literals with newlines (common practice)
        return '\n'.join(assembly_parts)
        
    def parse_constraint_list(self):
        """
        Parse constraint list for input/output operands
        Format: "constraint" (variable), "constraint" (variable), ...
        """
        constraints = []
        
        # Handle empty constraint list
        if (self.current_token() and 
            self.current_token().value == separators["COLON"]):
            return constraints
        
        if (self.current_token() and 
            self.current_token().value == separators["RPAREN"]):
            return constraints
        
        while True:
            # Parse constraint string
            if (not self.current_token() or 
                self.current_token()._type != TokenType.STRING):
                break
            
            constraint_str = self.current_token().value.strip('"\'')
            self.next_token()
            
            # Expect opening parenthesis
            if (not self.current_token() or 
                self.current_token().value != separators["LPAREN"]):
                self.syntax_error("Expected '(' after constraint", self.current_token())
            
            self.next_token()  # Consume '('
            
            # Parse variable expression (could be simple variable or more complex)
            if (not self.current_token() or 
                self.current_token()._type != TokenType.LITERAL):
                self.syntax_error("Expected variable name in constraint", self.current_token())
            
            variable_name = self.current_token().value
            self.next_token()
            
            # Expect closing parenthesis
            if (not self.current_token() or 
                self.current_token().value != separators["RPAREN"]):
                self.syntax_error("Expected ')' after variable name", self.current_token())
            
            self.next_token()  # Consume ')'
            
            # Store constraint with variable
            constraints.append({
                'constraint': constraint_str,
                'variable': variable_name
            })
            
            # Check for comma (more constraints)
            if (self.current_token() and 
                self.current_token().value == separators["COMMA"]):
                self.next_token()  # Consume ','
                continue
            else:
                break
        
        return constraints
    
        
    def parse_clobber_list(self):
        """
        Parse clobber list (registers that are modified)
        Format: "register", "register", ...
        """
        clobbers = []
        
        # Handle empty clobber list
        if (self.current_token() and 
            self.current_token().value == separators["RPAREN"]):
            return clobbers
        
        while True:
            # Parse clobber string
            if (not self.current_token() or 
                self.current_token()._type != TokenType.STRING):
                break
            
            clobber_str = self.current_token().value.strip('"\'')
            clobbers.append(clobber_str)
            self.next_token()
            
            # Check for comma (more clobbers)
            if (self.current_token() and 
                self.current_token().value == separators["COMMA"]):
                self.next_token()  # Consume ','
                continue
            else:
                break

        return clobbers


    def parse(self):
        
        while self.index < len(self.tokens): 
            node = self.parse_statement(inside_block=False)
            if node:
                self.nodes.append(node)

        return self.nodes
