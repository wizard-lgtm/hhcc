# Holy HolyC Compiler
# https://github.com/wizard-lgtm/hhcc
# Licensed under MIT

class TokenType:
    KEYWORD = "KEYWORD"
    IDENTIFIER = "IDENTIFIER"
    SEPARATOR = "SEPERATOR"
    OPERATOR = "OPERATOR"
    LITERAL = "LITERAL"
    COMMENT = "COMMENT"
    WHITESPACE = "WHITESPACE"
    DIRECTIVES = "DIRECTIVES"

# https://holyc-lang.com/docs/language-spec/learn-directives
directives = {
    "INCLUDE": "#include",   # Used to include code from another file or one of the libraries. For your code use a relative path and the #include "name.HC" syntax for a library use #include <name> (presently this is not used).
    "DEFINE": "#define",     # Defines a symbolic constant. These can only be strings or numerical expressions. Function macros are not supported in HolyC or TempleOS. This makes them much simpler. When referring to these in your code the value will get pasted in when the preprocessor sees the name of the #define.
    "UNDEF": "#undef",       # Undefines a symbolic constant that has been defined with #define if it exists.
    "IFDEF": "#ifdef",       # Conditionally compiles code if the symbolic constant exists. This does not test the value, merely ensuring its existence. Must be terminated with #endif. The following checks the existence of SOME_VARIABLE if it exists FOO will be defined and "Hello!" will be printed to stdout.
    "IFNDEF": "#ifndef",     # The opposite of #ifdef conditionally compiles the code if the symbolic constant does not exist. In this example "hello" will be printed if the symbolic constant MAX_LEN is not defined.
    "ELIFDEF": "#elifdef",   # For chaining conditions from #ifdef
    "ELSE": "#else",         # Used in conjunction with an #ifdef #ifndef or #elifdef to conditionally compile code if the condition is false. In the following example "foo" will be printed if SOME_VARIABLE is defined, else it will print "bar" if it is not.
    "ERROR": "#error"        # If hit it will immediately terminate compilation and print an error to stderr. Only accepts a string
}

class OperatorType:
    ARITHMETIC = "ARITHMETIC"
    BITWISE = "BITWISE"
    LOGICAL = "LOGICAL"
    COMPARISON = "COMPARISON"
    ASSIGNMENT = "ASSIGNMENT"
    INCREMENT = "INCREMENT"

# https://holyc-lang.com/docs/language-spec/learn-variables
operators = {
    # Arithmetic Operators
    "ADD": "+",
    "SUBTRACT": "-",
    "DIVIDE": "/",
    "MULTIPLY": "*",
    "MODULO": "%",

    # Bitwise Operators
    "BITWISE_NOT": "~",
    "BITWISE_XOR": "^",
    "BITWISE_AND": "&",
    "BITWISE_OR": "|",
    "SHIFT_LEFT": "<<",
    "SHIFT_RIGHT": ">>",

    # Logical Operators
    "LOGICAL_NOT": "!",
    "LOGICAL_AND": "&&",
    "LOGICAL_OR": "||",

    # Comparison Operators
    "LESS_THAN": "<",
    "GREATER_THAN": ">",
    "GREATER_OR_EQUAL": ">=",
    "LESS_OR_EQUAL": "<=",
    "EQUAL": "==",
    "NOT_EQUAL": "!=",

    # Assignment Operators
    "ASSIGN": "=",
    "SHIFT_LEFT_ASSIGN": "<<=",
    "SHIFT_RIGHT_ASSIGN": ">>=",
    "BITWISE_AND_ASSIGN": "&=",
    "BITWISE_OR_ASSIGN": "|=",
    "DIVIDE_ASSIGN": "/=",
    "MULTIPLY_ASSIGN": "*=",
    "ADD_ASSIGN": "+=",
    "SUBTRACT_ASSIGN": "-=",
    "MODULO_ASSIGN": "%=",

    # Increment/Decrement Operators
    "INCREMENT": "++",
    "DECREMENT": "--"
}

keywords = {
    "F64": "F64",        # 64bit floating point type. 8bytes wide.
    "U64": "U64",        # Unsigned 64bit Integer type. 8bytes wide.
    "I64": "I64",        # Signed 64bit Integer type. 8bytes wide.
    "U32": "U32",        # Unsigned 32bit Integer type. 4bytes wide.
    "I32": "I32",        # Signed 32bit Integer type. 4bytes wide.
    "U16": "U16",        # Unsigned 16bit Integer type. 2bytes wide.
    "I16": "I16",        # Signed 16bit Integer type. 2bytes wide.
    "U8": "U8",          # Unsigned 8bit Integer type. 1byte wide.
    "BOOL": "Bool",      # Signed 8bit Integer type. 1byte wide. This should either be 1 or 0.
    "U0": "U0",          # void type. Has no size.
    "IF": "if",
    "ELSE": "else",
    "SWITCH": "switch",
    "DO": "do",
    "WHILE": "while",
    "FOR": "for",
    "BREAK": "break",
    "CONTINUE": "continue",
    "GOTO": "goto",
    "RETURN": "return",
    "CLASS": "class",
    "UNION": "union",
    "ASM": "asm",
    "PUBLIC": "public",
    "EXTERN": "extern"
}

separators = {
    "COMMA": ",",           # separates arguments, elements in a list, etc.
    "DOT": ".",             # access member, method, or property
    "COLON": ":",           # for label, or to separate keys in a map
    "SEMICOLON": ";",       # statement termination
    "LPAREN": "(",          # left parenthesis
    "RPAREN": ")",          # right parenthesis
    "LBRACKET": "[",        # left square bracket
    "RBRACKET": "]",        # right square bracket
    "LBRACE": "{",          # left curly brace
    "RBRACE": "}",          # right curly brace
    "ARROW": "->",          # member or function pointer access
}

class Token: 
    def __init__(self, _type, value, line, column):
        self._type = _type
        self.value = value 
        self.line = line 
        self.column = column 

    def print(self):
        print(f'TOKEN: Type: {self._type}, value: {self.value}, line: {self.line}, column: {self.column}')
    _type: TokenType
    value: str
    line: int
    column: int

def tokenize(source_code:str): 
    tokens = []
    line = 1
    column = 0
    cursor = 0 

    while cursor < len(source_code):
        
        current_chr = source_code[cursor]

        # Skip whitespace (spaces, tabs)
        if current_chr.isspace():
            if current_chr == '\n':
                line += 1
                column = 0
            else:
                column += 1
            cursor += 1
            continue

        # Keywords
        if current_chr.isalnum():
            keyword_start = cursor
            while cursor < len(source_code) and source_code[cursor].isalnum() and source_code[cursor] != ' ':
                cursor += 1
            # parse keyword value
            keyword_val = source_code[keyword_start: cursor]
            if keyword_val in keywords:
                token = Token(TokenType.KEYWORD, keyword_val, line, column)
                tokens.append(token)  
            else:
                token = Token(TokenType.LITERAL, keyword_val, line, column)   
                tokens.append(token)       

        # Numerical
        elif current_chr.isdigit():
            start = cursor
            while cursor < len(source_code) and source_code[cursor].isdigit():
                cursor += 1
            # parse numeric value
            val = source_code[start: cursor]
            token = Token(TokenType.LITERAL, val, line, column)
            tokens.append(token)

        
        # Handle operators
        for operator_name, operator_value in operators.items():
            if source_code[cursor:cursor + len(operator_value)] == operator_value:
                token = Token(TokenType.OPERATOR, operator_name, line, column)
                tokens.append(token)
                cursor += len(operator_value)  # Move the cursor by the length of the operator
                column += len(operator_value)  # Adjust column number based on the operator length
                break
        # Handle separators
        for separator_name, separator_value in separators.items():
            if source_code[cursor:cursor + len(separator_value)] == separator_value:
                token = Token(TokenType.SEPARATOR, separator_name, line, column)
                tokens.append(token)
                cursor += len(separator_value)  # Move the cursor by the length of the separator
                column += len(separator_value)  # Adjust column number based on the separator length
                break
        else:
            cursor+=1
            column+=1

        # Operators
    return tokens
def main():
    code = r"""
U8 a = 5;
U64 b = 10;
U8 c = a + b;
"""
    tokens = tokenize(code)  
    for token in tokens:
        token.print()
main()