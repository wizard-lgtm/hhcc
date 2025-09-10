from typing import TYPE_CHECKING

from llvmlite import ir

if TYPE_CHECKING:
    from compiler import Compiler  # Only for type hints
class TokenType:
    KEYWORD = "KEYWORD"
    SEPARATOR = "SEPERATOR"
    OPERATOR = "OPERATOR"
    LITERAL = "LITERAL"
    COMMENT = "COMMENT"
    WHITESPACE = "WHITESPACE"
    DIRECTIVE = "DIRECTIVE"

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

    # increment/decrement operators
    "increment": "++",
    "decrement": "--",
    "POINTER": "*",
    "REFERANCE": "&"
}

assignment_operators = {
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

}

class Datatypes:
    U8 = "U8"
    U16 = "U16"
    U32 = "U32"
    U64 = "U64"
    I8 = "I8"
    I16 = "I16"
    I32 = "I32"
    I64 = "I64"
    BOOL = "Bool"    # Boolean type, 1 byte wide (should be 0 or 1).
    U0 = "U0"        # Void type, has no size.
    F64 = "F64"
    F32 = "F32"
    
    user_defined_types = {}
    type_info = {
        BOOL: {"type": ir.IntType(1), "signed": False},
        U8: {"type": ir.IntType(8), "signed": False},
        I8: {"type": ir.IntType(8), "signed": True},
        U16: {"type": ir.IntType(16), "signed": False},
        U32: {"type": ir.IntType(32), "signed": False},
        U64: {"type": ir.IntType(64), "signed": False},
        I16: {"type": ir.IntType(16), "signed": True},
        I32: {"type": ir.IntType(32), "signed": True},
        I64: {"type": ir.IntType(64), "signed": True},
        U0: {"type": ir.VoidType(), "signed": False},
        F32: {"type": ir.FloatType(), "signed": True},
        F64: {"type": ir.DoubleType(), "signed": True}
    }

    @classmethod
    def all_types(cls):
        return [
            cls.F64, cls.U64, cls.I64, cls.U32, cls.I32,
            cls.U16, cls.I16, cls.U8, cls.I8, cls.BOOL, cls.U0
        ] + list(cls.user_defined_types.keys())

    @classmethod
    def get_type_from_string(cls, type_name: str):
        return getattr(cls, type_name, cls.user_defined_types.get(type_name, None))

    @classmethod
    def add_type(cls, name, type_def):
        cls.user_defined_types[name] = type_def


    @classmethod
    def get_llvm_type(cls, type_name: str):
        if type_name in cls.type_info:
            return cls.type_info[type_name]["type"]
        
        # Handle pointer types
        elif type_name.endswith('*'):
            base_type = cls.get_llvm_type(type_name[:-1])
            return ir.PointerType(base_type)
        
        # Handle user-defined struct/class types
        elif type_name in cls.user_defined_types:
            return cls.user_defined_types[type_name].get_llvm_type()
        
        else:
            raise Exception(f"Unknown type '{type_name}'")
            
    @classmethod
    def get_type(cls, name: str):
        return cls.user_defined_types.get(name, None)
        
    @classmethod
    def is_signed_type(cls, type_name: str) -> bool:
        """Determine if a type is signed."""
        if not type_name:
            return False  # Default to unsigned if type_name is None or empty
        if isinstance(type_name, str) and type_name.endswith('*'):
            return False  # Pointers are never signed
        return type_name.startswith("I") or type_name.startswith("F")

    @classmethod
    def is_integer_type(cls, type_name: str) -> bool:
        """Determine if a type is an integer type."""
        if not type_name:
            return True  # Default to integer if type_name is None or empty
        if type_name.endswith('*'):
            return False  # Pointers are not integer types
        return type_name.startswith("I") or type_name.startswith("U") or type_name == cls.BOOL

    @classmethod
    def is_float_type(cls, type_name: str) -> bool:
        """Determine if a type is a floating point type."""
        if not type_name:
            return False  # Default to not float if type_name is None or empty
        return type_name.startswith("F")
    @classmethod
    def is_float_type(cls, type_name: str) -> bool:
        return type_name.startswith("F")

    @classmethod
    def to_llvm_type(cls, type_name: str, pointer_level: int = 0) -> ir.Type:
        """Convert the type name + pointer level to the corresponding llvmlite type."""

        # Handle user-defined types
        if type_name in cls.user_defined_types:
            user_type = cls.user_defined_types[type_name]
            llvm_type = user_type.get_llvm_type() if hasattr(user_type, 'get_llvm_type') else user_type
        else:
            # Primitive and built-in types
            if type_name == cls.U8:
                llvm_type = ir.IntType(8)
            elif type_name == cls.U16:
                llvm_type = ir.IntType(16)
            elif type_name == cls.U32:
                llvm_type = ir.IntType(32)
            elif type_name == cls.U64:
                llvm_type = ir.IntType(64)
            elif type_name == cls.I8:
                llvm_type = ir.IntType(8)
            elif type_name == cls.I16:
                llvm_type = ir.IntType(16)
            elif type_name == cls.I32:
                llvm_type = ir.IntType(32)
            elif type_name == cls.I64:
                llvm_type = ir.IntType(64)
            elif type_name == cls.BOOL:
                llvm_type = ir.IntType(1)
            elif type_name == cls.U0:
                llvm_type = ir.VoidType()
            elif type_name == cls.F64:
                llvm_type = ir.DoubleType()
            elif type_name == cls.F32:
                llvm_type = ir.FloatType()
            else:
                raise ValueError(f"Unknown type: {type_name}")

        # Apply pointer levels
        for _ in range(pointer_level):
            llvm_type = llvm_type.as_pointer()

        return llvm_type



keywords = {
    "F64": "F64",        # 64bit floating point type. 8bytes wide.
    "U64": "U64",        # Unsigned 64bit Integer type. 8bytes wide.
    "I64": "I64",        # Signed 64bit Integer type. 8bytes wide.
    "U32": "U32",        # Unsigned 32bit Integer type. 4bytes wide.
    "I32": "I32",        # Signed 32bit Integer type. 4bytes wide.
    "U16": "U16",        # Unsigned 16bit Integer type. 2bytes wide.
    "I16": "I16",        # Signed 16bit Integer type. 2bytes wide.
    "U8": "U8",          # Unsigned 8bit Integer type. 1byte wide.
    "I8": "I8",          # Signed 8bit Integer type. 1byte wide.
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
    "EXTERN": "extern",
    "TRUE": "true",
    "FALSE": "false",
    "VOLATILE": "volatile", #  # Used to indicate that a variable may be changed by external factors, such as hardware or other threads.
    "ENUM": "enum",  # Used to define a set of named integer constants.
    "CONST": "const",
    "STATIC": "static",
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
    "SCOPE": "::",          # scope resolution operator, used to access members of a class or namespace
    "THREEDOTS": "...",        # ellipsis for variadic functions
}

class Token: 
    def __init__(self, _type, value, line, column):
        self._type = _type
        self.value = value 
        self.line = line 
        self.column = column 
    
    def __repr__(self):
        return f"Token(type={self._type}, value={repr(self.value)}, line={self.line}, column={self.column})"

    def print(self):
        print(f'TOKEN: Type: {self._type}, value: {self.value}, line: {self.line}, column: {self.column}')
    _type: TokenType
    value: str
    line: int
    column: int


class Lexer:
    def __init__(self, compiler: "Compiler"):
        self.compiler = compiler
        self.source_code = self.compiler.src 

        self.type_info = {
            Datatypes.BOOL: {"type": ir.IntType(1), "signed": False},
            Datatypes.U8: {"type": ir.IntType(8), "signed": False},
            Datatypes.U16: {"type": ir.IntType(16), "signed": False},
            Datatypes.U32: {"type": ir.IntType(32), "signed": False},
            Datatypes.U64: {"type": ir.IntType(64), "signed": False},
            Datatypes.I8: {"type": ir.IntType(8), "signed": True},
            Datatypes.I16: {"type": ir.IntType(16), "signed": True},
            Datatypes.I32: {"type": ir.IntType(32), "signed": True},
            Datatypes.I64: {"type": ir.IntType(64), "signed": True},
            Datatypes.U0: {"type": ir.VoidType(), "signed": False},
            Datatypes.F32: {"type": ir.FloatType(), "signed": True},
            Datatypes.F64: {"type": ir.DoubleType(), "signed": True}
        }

        self.type_map = {k: v["type"] for k, v in self.type_info.items()}

    def _handle_string_literal(self, cursor, line, column):
        """Handle regular string literals like "hello world" """
        start = cursor
        cursor += 1  # Skip opening quote
        
        value = ""
        while cursor < len(self.source_code):
            current_char = self.source_code[cursor]
            
            if current_char == '"':
                # End of string
                cursor += 1  # Skip closing quote
                break
            elif current_char == '\\' and cursor + 1 < len(self.source_code):
                # Handle escape sequences
                cursor += 1
                next_char = self.source_code[cursor]
                if next_char == 'n':
                    value += '\n'
                elif next_char == 't':
                    value += '\t'
                elif next_char == 'r':
                    value += '\r'
                elif next_char == '\\':
                    value += '\\'
                elif next_char == '"':
                    value += '"'
                elif next_char == '0':
                    value += '\0'
                else:
                    # Unknown escape sequence, just add both characters
                    value += '\\' + next_char
                cursor += 1
            else:
                value += current_char
                cursor += 1
        
        # Return the string content without quotes, cursor position, and token
        token = Token(TokenType.LITERAL, f'"{value}"', line, column)
        return token, cursor

    def _handle_char_literal(self, cursor, line, column):
        """Handle character literals like 'p' or '\n' """
        start = cursor
        cursor += 1  # Skip opening quote
        
        value = ""
        if cursor < len(self.source_code):
            current_char = self.source_code[cursor]
            
            if current_char == '\\' and cursor + 1 < len(self.source_code):
                # Handle escape sequences
                cursor += 1
                next_char = self.source_code[cursor]
                if next_char == 'n':
                    value += '\n'
                elif next_char == 't':
                    value += '\t'
                elif next_char == 'r':
                    value += '\r'
                elif next_char == '\\':
                    value += '\\'
                elif next_char == "'":
                    value += "'"
                elif next_char == '0':
                    value += '\0'
                else:
                    # Unknown escape sequence, just add both characters
                    value += '\\' + next_char
                cursor += 1
            else:
                # Regular character
                value += current_char
                cursor += 1
        
        # Expect closing quote
        if cursor < len(self.source_code) and self.source_code[cursor] == "'":
            cursor += 1  # Skip closing quote
        
        # Return the character literal with quotes, cursor position, and token
        token = Token(TokenType.LITERAL, f"'{value}'", line, column)
        return token, cursor

    def _handle_raw_string_literal(self, cursor, line, column):
        """Handle raw string literals like R"(content)" """
        start = cursor
        cursor += 2  # Skip 'R"'
        
        # Find the opening delimiter
        delimiter_start = cursor
        if cursor < len(self.source_code) and self.source_code[cursor] == '(':
            cursor += 1  # Skip '('
            
            # Find the content until the closing pattern
            value = ""
            while cursor < len(self.source_code):
                if (cursor + 1 < len(self.source_code) and 
                    self.source_code[cursor:cursor + 2] == ')"'):
                    cursor += 2  # Skip ')"'
                    break
                else:
                    value += self.source_code[cursor]
                    cursor += 1
            
            # Return the raw string content, cursor position, and token
            token = Token(TokenType.LITERAL, f'R"({value})"', line, column)
            return token, cursor
        else:
            # Malformed raw string, treat as regular tokens
            return None, start

    def tokenize(self):
        source_code = self.source_code
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
    
            # Handle Character Literals
            elif current_chr == "'":
                token, new_cursor = self._handle_char_literal(cursor, line, column)
                tokens.append(token)
                column += new_cursor - cursor
                cursor = new_cursor
                continue

            # Handle Raw String Literals (R"(...)")
            if (cursor + 1 < len(source_code) and 
                source_code[cursor:cursor + 2] == 'R"'):
                token, new_cursor = self._handle_raw_string_literal(cursor, line, column)
                if token:
                    tokens.append(token)
                    column += new_cursor - cursor
                    cursor = new_cursor
                    continue
                # If raw string parsing failed, fall through to normal processing

            # Handle Regular String Literals
            elif current_chr == '"':
                token, new_cursor = self._handle_string_literal(cursor, line, column)
                tokens.append(token)
                column += new_cursor - cursor
                cursor = new_cursor
                continue

            elif current_chr == '#':
                start = cursor
                while cursor < len(source_code) and source_code[cursor] not in [' ', '\n', '"', '<']:
                    cursor += 1
                directive_name = source_code[start:cursor]
                token = Token(TokenType.DIRECTIVE, directive_name, line, column)
                tokens.append(token)
                column += cursor - start

                # şimdi string veya <> yakala
                if cursor < len(source_code) and source_code[cursor] in ['"', '<']:
                    if source_code[cursor] == '"':
                        string_token, new_cursor = self._handle_string_literal(cursor, line, column)
                        tokens.append(string_token)
                        column += new_cursor - cursor
                        cursor = new_cursor
                    else:
                        # #include <stdio.h> tarzı
                        start = cursor
                        cursor += 1
                        while cursor < len(source_code) and source_code[cursor] != '>':
                            cursor += 1
                        cursor += 1
                        path = source_code[start:cursor]
                        tokens.append(Token(TokenType.LITERAL, path, line, column))
                        column += cursor - start
                continue


            # Identifiers and Keywords
            elif current_chr.isalpha() or current_chr == "_":  # Start of an identifier/keyword
                start = cursor
                while cursor < len(source_code) and (source_code[cursor].isalnum() or source_code[cursor] == "_"):
                    cursor += 1

                value = source_code[start:cursor]

                if value in keywords.values():
                    # handling for boolean literals
                    if value in [keywords["TRUE"], keywords["FALSE"]]:
                        token = Token(TokenType.LITERAL, value, line, column)
                    else:
                        token = Token(TokenType.KEYWORD, value, line, column)
                else:
                    token = Token(TokenType.LITERAL, value, line, column)

                tokens.append(token)
                column += cursor - start
                continue  # Avoid incrementing cursor again

            # Numerical Literals (including floating point and hexadecimal)
            elif current_chr.isdigit() or (current_chr == '.' and cursor + 1 < len(source_code) and source_code[cursor + 1].isdigit()):
                start = cursor
                has_decimal_point = False
                is_hex = False
                
                # Check for hexadecimal (0x or 0X)
                if (current_chr == '0' and cursor + 1 < len(source_code) and 
                    (source_code[cursor + 1] == 'x' or source_code[cursor + 1] == 'X')):
                    is_hex = True
                    cursor += 2  # Skip '0x' or '0X'
                    
                    # Consume hexadecimal digits
                    while cursor < len(source_code) and (source_code[cursor].isdigit() or 
                                                    source_code[cursor].lower() in 'abcdef'):
                        cursor += 1
                else:
                    # Regular decimal number
                    # First part: digits before decimal point
                    while cursor < len(source_code) and source_code[cursor].isdigit():
                        cursor += 1
                    
                    # Handle decimal point if present
                    if cursor < len(source_code) and source_code[cursor] == '.':
                        has_decimal_point = True
                        cursor += 1  # Consume the decimal point
                        
                        # Digits after decimal point
                        while cursor < len(source_code) and source_code[cursor].isdigit():
                            cursor += 1

                    # Handle scientific notation (e.g. 1.23e+10) - only for decimal numbers
                    if not is_hex and cursor < len(source_code) and (source_code[cursor] == 'e' or source_code[cursor] == 'E'):
                        cursor += 1  # Consume the 'e' or 'E'
                        
                        # Optional + or - sign
                        if cursor < len(source_code) and (source_code[cursor] == '+' or source_code[cursor] == '-'):
                            cursor += 1
                        
                        # Exponent digits
                        if cursor < len(source_code) and source_code[cursor].isdigit():
                            while cursor < len(source_code) and source_code[cursor].isdigit():
                                cursor += 1
                        else:
                            # Malformed scientific notation, roll back to before the 'e'/'E'
                            cursor = start
                            while cursor < len(source_code) and source_code[cursor] != 'e' and source_code[cursor] != 'E':
                                cursor += 1

                value = source_code[start:cursor]
                token = Token(TokenType.LITERAL, value, line, column)
                tokens.append(token)
                column += cursor - start
                continue  # Avoid incrementing cursor again

            # Handle Comments
            elif cursor + 1 < len(source_code) and source_code[cursor:cursor + 2] == "//":
                start = cursor
                cursor += 2  # Skip past the //
                
                # Continue until we hit a newline or end of file
                while cursor < len(source_code) and source_code[cursor] != '\n':
                    cursor += 1
                
                value = source_code[start:cursor]
                token = Token(TokenType.COMMENT, value, line, column)
                tokens.append(token)
                column += cursor - start
                continue  # Avoid incrementing cursor again

            # Handle multi-character operators and separators first (longest match first)
            else:
                # Check for scope resolution operator :: first
                if (cursor + 1 < len(source_code) and 
                    source_code[cursor:cursor + 2] == separators["SCOPE"]):
                    token = Token(TokenType.SEPARATOR, separators["SCOPE"], line, column)
                    tokens.append(token)
                    cursor += 2
                    column += 2
                    continue
                
                # Handle ellipsis (before other operators)
                elif source_code[cursor:cursor + 3] == separators["THREEDOTS"]:
                    token = Token(TokenType.SEPARATOR, separators["THREEDOTS"], line, column)
                    tokens.append(token)
                    cursor += 3
                    column += 3
                    continue

                # Handle other multi-character operators
                operator_found = False
                for operator_name, operator_value in sorted(operators.items(), key=lambda x: -len(x[1])):  # Match longest first
                    if source_code[cursor:cursor + len(operator_value)] == operator_value:
                        token = Token(TokenType.OPERATOR, operator_value, line, column)
                        tokens.append(token)
                        cursor += len(operator_value)
                        column += len(operator_value)
                        operator_found = True
                        break
                
                if not operator_found:
                    # Handle single-character separators
                    separator_found = False
                    for separator_name, separator_value in separators.items():
                        if (len(separator_value) == 1 and  # Only single-character separators here
                            source_code[cursor:cursor + len(separator_value)] == separator_value):
                            token = Token(TokenType.SEPARATOR, separator_value, line, column)
                            tokens.append(token)
                            cursor += len(separator_value)
                            column += len(separator_value)
                            separator_found = True
                            break
                    
                    if not separator_found:
                        # Unknown character, skip it
                        cursor += 1
                        column += 1

        return tokens