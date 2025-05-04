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
    def __init__(self):
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

    @classmethod
    def all_types(cls):
        return [
            cls.F64, cls.U64, cls.I64, cls.U32, cls.I32,
            cls.U16, cls.I16, cls.U8, cls.BOOL, cls.U0
        ] + list(cls.user_defined_types.keys())

    @classmethod
    def get_type_from_string(cls, type_name: str):
        return getattr(cls, type_name, cls.user_defined_types.get(type_name, None))

    @classmethod
    def add_type(cls, name, type_def):
        cls.user_defined_types[name] = type_def


    @classmethod
    def get_llvm_type(self, type_name: str):
        if type_name in self.type_info:
            return self.type_info[type_name]["type"]
        # Handle pointer and other types as before
        elif type_name.endswith('*'):
            base_type = self.get_llvm_type(type_name[:-1])
            return ir.PointerType(base_type)
        else:
            print("Non-Primitive types not implemented. Returning a generic type (U8*)")
            return ir.PointerType(ir.IntType(8))


    @classmethod
    def is_signed_type(cls, type_name: str) -> bool:
        """Determine if a type is signed."""
        if not type_name:
            return False  # Default to unsigned if type_name is None or empty
        if type_name.endswith('*'):
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
    def to_llvm_type(cls, type_name: str) -> ir.Type:
        """Convert the type name to the corresponding llvmlite type."""
        # First check if the type is a user-defined type
        if type_name in cls.user_defined_types:
            user_type = cls.user_defined_types[type_name]
            
            # If it's a wrapper like ClassTypeInfo, extract the LLVM type
            if hasattr(user_type, 'get_llvm_type'):
                return user_type.get_llvm_type()
            
            # Otherwise, assume it's already an LLVM type
            return user_type

        # Primitive and built-in types
        if type_name == cls.U8:
            return ir.IntType(8)
        elif type_name == cls.U16:
            return ir.IntType(16)
        elif type_name == cls.U32:
            return ir.IntType(32)
        elif type_name == cls.U64:
            return ir.IntType(64)
        elif type_name == cls.I8:
            return ir.IntType(8)
        elif type_name == cls.I16:
            return ir.IntType(16)
        elif type_name == cls.I32:
            return ir.IntType(32)
        elif type_name == cls.I64:
            return ir.IntType(64)
        elif type_name == cls.BOOL:
            return ir.IntType(1)  # Boolean is typically 1 bit in LLVM IR
        elif type_name == cls.U0:
            return ir.VoidType()  # Void type has no size
        elif type_name == cls.F64:
            return ir.DoubleType()  # Double precision float
        elif type_name == cls.F32:
            return ir.FloatType()  # Single precision float
        else:
            raise ValueError(f"Unknown type: {type_name}")



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
    "EXTERN": "extern",
    "TRUE": "true",
    "FALSE": "false"
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

            # Handle Directives (they start with #)
            if current_chr == '#':
                start = cursor
                # Find the directive name
                directive_start = cursor
                while cursor < len(source_code) and source_code[cursor] != ' ' and source_code[cursor] != '\n':
                    cursor += 1
                
                directive_name = source_code[directive_start:cursor]
                

                # Skip whitespace after directive name
                while cursor < len(source_code) and source_code[cursor].isspace() and source_code[cursor] != '\n':
                    cursor += 1
                
                # Capture the rest of the line as part of the directive
                directive_start = cursor
                while cursor < len(source_code) and source_code[cursor] != '\n':
                    cursor += 1
                
                directive_value = source_code[start:cursor]
                token = Token(TokenType.DIRECTIVE, directive_value, line, column)
                tokens.append(token)
                column += cursor - start
                continue


            # Identifiers and Keywords
            if current_chr.isalpha() or current_chr == "_":  # Start of an identifier/keyword
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

            # Numerical Literals
            elif current_chr.isdigit():
                start = cursor
                while cursor < len(source_code) and source_code[cursor].isdigit():
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


            # Handle Operators
            for operator_name, operator_value in sorted(operators.items(), key=lambda x: -len(x[1])):  # Match longest first
                if source_code[cursor:cursor + len(operator_value)] == operator_value:
                    token = Token(TokenType.OPERATOR, operator_value, line, column)
                    tokens.append(token)
                    cursor += len(operator_value)
                    column += len(operator_value)
                    break
            else:
                # Handle Separators
                for separator_name, separator_value in separators.items():
                    if source_code[cursor:cursor + len(separator_value)] == separator_value:
                        token = Token(TokenType.SEPARATOR, separator_value, line, column)
                        tokens.append(token)
                        cursor += len(separator_value)
                        column += len(separator_value)
                        break
                else:
                    cursor += 1
                    column += 1

        return tokens