-- Holy HolyC Compiler
-- https://github.com/wizard-lgtm/hhcc
-- Licensed under MIT

local TokenType = {
    KEYWORD = 1,
    IDENTIFIER = 2,
    SEPARATOR = 3,
    OPERATOR = 4,
    LITERAL = 5,
    COMMENT = 6,
    WHITESPACE = 7,
    DIRECTIVES = 8
}
-- https://holyc-lang.com/docs/language-spec/learn-directives
local directives = {
    INCLUDE = "#include",   -- Used to include code from another file or one of the libraries. For your code use a relative path and the #include "name.HC" syntax for a library use #include <name> (presently this is not used).
    DEFINE = "#define",     -- Defines a symbolic constant. These can only be strings or numerical expressions. Function macros are not supported in HolyC or TempleOS. This makes them much simpler. When referring to these in your code the value will get pasted in when the preprocessor sees the name of the #define.
    UNDEF = "#undef",       -- Undefines a symbolic constant that has been defined with #define if it exists.
    IFDEF = "#ifdef",       -- Conditionally compiles code if the symbolic constant exists. This does not test the value, merely ensuring its existence. Must be terminated with #endif. The following checks the existence of SOME_VARIABLE if it exists FOO will be defined and "Hello!" will be printed to stdout.
    IFNDEF = "#ifndef",     -- The opposite of #ifdef conditionally compiles the code if the symbolic constant does not exist. In this example "hello" will be printed if the symbolic constant MAX_LEN is not defined.
    ELIFDEF = "#elifdef",   -- For chaining conditions from #ifdef
    ELSE = "#else",         -- Used in conjunction with an #ifdef #ifndef or #elifdef to conditionally compile code if the condition is false. In the following example "foo" will be printed if SOME_VARIABLE is defined, else it will print "bar" if it is not.
    ERROR = "#error"        -- If hit it will immediately terminate compilation and print an error to stderr. Only accepts a string
}

local OperatorType = {
    ARITHMETIC = 1,
    BITWISE = 2,
    LOGICAL = 3,
    COMPARISON = 4,
    ASSIGNMENT = 5,
    INCREMENT = 6
}
-- https://holyc-lang.com/docs/language-spec/learn-variables
local operators = {
    -- Arithmetic Operators
    { type = OperatorType.ARITHMETIC, name = "add", value = "+" },
    { type = OperatorType.ARITHMETIC, name = "subtract", value = "-" },
    { type = OperatorType.ARITHMETIC, name = "divide", value = "/" },
    { type = OperatorType.ARITHMETIC, name = "multiply", value = "*" },
    { type = OperatorType.ARITHMETIC, name = "modulo", value = "%" },

    -- Bitwise Operators
    { type = OperatorType.BITWISE, name = "bitwise NOT", value = "~" },
    { type = OperatorType.BITWISE, name = "bitwise XOR", value = "^" },
    { type = OperatorType.BITWISE, name = "bitwise AND", value = "&" },
    { type = OperatorType.BITWISE, name = "bitwise OR", value = "|" },
    { type = OperatorType.BITWISE, name = "shift left", value = "<<" },
    { type = OperatorType.BITWISE, name = "shift right", value = ">>" },

    -- Logical Operators
    { type = OperatorType.LOGICAL, name = "logical NOT", value = "!" },
    { type = OperatorType.LOGICAL, name = "logical AND", value = "&&" },
    { type = OperatorType.LOGICAL, name = "logical OR", value = "||" },

    -- Comparison Operators
    { type = OperatorType.COMPARISON, name = "less than", value = "<" },
    { type = OperatorType.COMPARISON, name = "greater than", value = ">" },
    { type = OperatorType.COMPARISON, name = "greater or equal to", value = ">=" },
    { type = OperatorType.COMPARISON, name = "less than or equal to", value = "<=" },
    { type = OperatorType.COMPARISON, name = "equal", value = "==" },
    { type = OperatorType.COMPARISON, name = "not equal", value = "!=" },

    -- Assignment Operators
    { type = OperatorType.ASSIGNMENT, name = "assign", value = "=" },
    { type = OperatorType.ASSIGNMENT, name = "shift left and assign", value = "<<=" },
    { type = OperatorType.ASSIGNMENT, name = "shift right and assign", value = ">>=" },
    { type = OperatorType.ASSIGNMENT, name = "bitwise AND and assign", value = "&=" },
    { type = OperatorType.ASSIGNMENT, name = "bitwise OR and assign", value = "|=" },
    { type = OperatorType.ASSIGNMENT, name = "divide and assign", value = "/=" },
    { type = OperatorType.ASSIGNMENT, name = "multiply and assign", value = "*=" },
    { type = OperatorType.ASSIGNMENT, name = "add and assign", value = "+=" },
    { type = OperatorType.ASSIGNMENT, name = "subtract and assign", value = "-=" },
    { type = OperatorType.ASSIGNMENT, name = "modulo and assign", value = "%=" },

    -- Increment/Decrement Operators
    { type = OperatorType.INCREMENT, name = "increment", value = "++" },
    { type = OperatorType.INCREMENT, name = "decrement", value = "--" }
}

local keywords = {
    F64 = "F64",        -- 64bit floating point type. 8bytes wide.
    U64 = "U64",        -- Unsigned 64bit Integer type. 8bytes wide.
    I64 = "I64",        -- Signed 64bit Integer type. 8bytes wide.
    U32 = "U32",        -- Unsigned 32bit Integer type. 4bytes wide.
    I32 = "I32",        -- Signed 32bit Integer type. 4bytes wide.
    U16 = "U16",        -- Unsigned 16bit Integer type. 2bytes wide.
    I16 = "I16",        -- Signed 16bit Integer type. 2bytes wide.
    U8 = "U8",          -- Unsigned 8bit Integer type. 1byte wide.
    BOOL = "Bool",      -- Signed 8bit Integer type. 1byte wide. This should either be 1 or 0.
    U0 = "U0",          -- void type. Has no size.
    IF = "if",
    ELSE = "else",
    SWITCH = "switch",
    DO = "do",
    WHILE = "while",
    FOR = "for",
    BREAK = "break",
    CONTINUE = "continue",
    GOTO = "goto",
    RETURN = "return",
    CLASS = "class",
    UNION = "union",
    ASM = "asm",
    PUBLIC = "public",
    EXTERN = "extern"
}

local seperators = {
    COMMA = ",",           -- separates arguments, elements in a list, etc.
    DOT = ".",             -- access member, method, or property
    COLON = ":",           -- for label, or to separate keys in a map
    SEMICOLON = ";",       -- statement termination
    LPAREN = "(",          -- left parenthesis
    RPAREN = ")",          -- right parenthesis
    LBRACKET = "[",        -- left square bracket
    RBRACKET = "]",        -- right square bracket
    LBRACE = "{",          -- left curly brace
    RBRACE = "}",          -- right curly brace
    ARROW = "->",          -- member or function pointer access
}
