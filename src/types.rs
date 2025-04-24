use std::collections::HashMap;
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum TokenType {
    Keyword,
    DataType,
    Separator,
    Operator,
    Literal,
    Comment,
    Whitespace,
    Directive,
    Identifier,
}


// Token structure
#[derive(Debug, Clone)]
pub struct Token {
    pub token_type: TokenType,
    pub value: String,
    pub line: usize,
    pub column: usize,
}

impl Token {
    pub fn new(token_type: TokenType, value: String, line: usize, column: usize) -> Self {
        Self {
            token_type,
            value,
            line,
            column,
        }
    }

    pub fn print(&self) {
        println!(
            "TOKEN: Type: {:?}, value: {}, line: {}, column: {}",
            self.token_type, self.value, self.line, self.column
        );
    }

    // Helper methods to check token type
    pub fn is_data_type(&self) -> bool {
        self.token_type == TokenType::DataType
    }
    
    pub fn is_keyword(&self) -> bool {
        self.token_type == TokenType::Keyword
    }
    
    pub fn is_separator(&self) -> bool {
        self.token_type == TokenType::Separator
    }
}


#[derive(Debug, Eq, PartialEq, Hash)]
pub enum Directive {
    Include,
    Define,
    Undef,
    IfDef,
    IfNDef,
    ElIfDef,
    Else,
    Error,
}

impl fmt::Display for Directive {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Directive::Include => "#include",
            Directive::Define => "#define",
            Directive::Undef => "#undef",
            Directive::IfDef => "#ifdef",
            Directive::IfNDef => "#ifndef",
            Directive::ElIfDef => "#elifdef",
            Directive::Else => "#else",
            Directive::Error => "#error",
        };
        write!(f, "{}", s)
    }
}

pub struct Directives;

impl Directives {
    pub fn get_all() -> HashMap<Directive, &'static str> {
        use Directive::*;

        let mut directives = HashMap::new();
        directives.insert(Include, "#include");
        directives.insert(Define, "#define");
        directives.insert(Undef, "#undef");
        directives.insert(IfDef, "#ifdef");
        directives.insert(IfNDef, "#ifndef");
        directives.insert(ElIfDef, "#elifdef");
        directives.insert(Else, "#else");
        directives.insert(Error, "#error");
        directives
    }
}


#[derive(Debug, Eq, PartialEq, Hash, Clone, Copy)]
pub enum Operator {
    // Arithmetic
    Add,
    Subtract,
    Divide,
    Multiply,
    Modulo,
    // Bitwise
    BitwiseNot,
    BitwiseXor,
    BitwiseAnd,
    BitwiseOr,
    ShiftLeft,
    ShiftRight,
    // Logical
    LogicalNot,
    LogicalAnd,
    LogicalOr,
    // Comparison
    LessThan,
    GreaterThan,
    GreaterOrEqual,
    LessOrEqual,
    Equal,
    NotEqual,
    // Assignment
    Assign,
    ShiftLeftAssign,
    ShiftRightAssign,
    BitwiseAndAssign,
    BitwiseOrAssign,
    DivideAssign,
    MultiplyAssign,
    AddAssign,
    SubtractAssign,
    ModuloAssign,
    // Increment/Decrement
    Increment,
    Decrement,
    
    // Special
    Pointer,
    Reference,
}

impl fmt::Display for Operator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let symbol = match self {
            Operator::Add => "+",
            Operator::Subtract => "-",
            Operator::Divide => "/",
            Operator::Multiply => "*",
            Operator::Modulo => "%",
            Operator::BitwiseNot => "~",
            Operator::BitwiseXor => "^",
            Operator::BitwiseAnd => "&",
            Operator::BitwiseOr => "|",
            Operator::ShiftLeft => "<<",
            Operator::ShiftRight => ">>",
            Operator::LogicalNot => "!",
            Operator::LogicalAnd => "&&",
            Operator::LogicalOr => "||",
            Operator::LessThan => "<",
            Operator::GreaterThan => ">",
            Operator::GreaterOrEqual => ">=",
            Operator::LessOrEqual => "<=",
            Operator::Equal => "==",
            Operator::NotEqual => "!=",
            Operator::Assign => "=",
            Operator::ShiftLeftAssign => "<<=",
            Operator::ShiftRightAssign => ">>=",
            Operator::BitwiseAndAssign => "&=",
            Operator::BitwiseOrAssign => "|=",
            Operator::DivideAssign => "/=",
            Operator::MultiplyAssign => "*=",
            Operator::AddAssign => "+=",
            Operator::SubtractAssign => "-=",
            Operator::ModuloAssign => "%=",
            Operator::Increment => "++",
            Operator::Decrement => "--",
            Operator::Pointer => "*",
            Operator::Reference => "&",
        };
        write!(f, "{}", symbol)
    }
}

impl Operator {
    // Helper to get the string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            Operator::Add => "+",
            Operator::Subtract => "-",
            Operator::Divide => "/",
            Operator::Multiply => "*",
            Operator::Modulo => "%",
            Operator::BitwiseNot => "~",
            Operator::BitwiseXor => "^",
            Operator::BitwiseAnd => "&",
            Operator::BitwiseOr => "|",
            Operator::ShiftLeft => "<<",
            Operator::ShiftRight => ">>",
            Operator::LogicalNot => "!",
            Operator::LogicalAnd => "&&",
            Operator::LogicalOr => "||",
            Operator::LessThan => "<",
            Operator::GreaterThan => ">",
            Operator::GreaterOrEqual => ">=",
            Operator::LessOrEqual => "<=",
            Operator::Equal => "==",
            Operator::NotEqual => "!=",
            Operator::Assign => "=",
            Operator::ShiftLeftAssign => "<<=",
            Operator::ShiftRightAssign => ">>=",
            Operator::BitwiseAndAssign => "&=",
            Operator::BitwiseOrAssign => "|=",
            Operator::DivideAssign => "/=",
            Operator::MultiplyAssign => "*=",
            Operator::AddAssign => "+=",
            Operator::SubtractAssign => "-=",
            Operator::ModuloAssign => "%=",
            Operator::Increment => "++",
            Operator::Decrement => "--",
            Operator::Pointer => "*",
            Operator::Reference => "&",
        }
    }
}

pub struct Operators;
impl Operators {
    pub fn get_all() -> HashMap<Operator, &'static str> {
        use Operator::*;
        let mut operators = HashMap::new();
        let all_operators = vec![
            // Arithmetic
            Add, Subtract, Divide, Multiply, Modulo,
            // Bitwise
            BitwiseNot, BitwiseXor, BitwiseAnd, BitwiseOr, ShiftLeft, ShiftRight,
            // Logical
            LogicalNot, LogicalAnd, LogicalOr,
            // Comparison
            LessThan, GreaterThan, GreaterOrEqual, LessOrEqual, Equal, NotEqual,
            // Assignment
            Assign, ShiftLeftAssign, ShiftRightAssign, BitwiseAndAssign, BitwiseOrAssign,
            DivideAssign, MultiplyAssign, AddAssign, SubtractAssign, ModuloAssign,
            // Increment/Decrement
            Increment, Decrement,
            // Special
            Pointer, Reference,
        ];
        for op in all_operators {
            operators.insert(op, op.as_str());
        }
        operators
    }
    
    pub fn get_assignment_operators() -> HashMap<Operator, &'static str> {
        use Operator::*;
        let mut assignment_operators = HashMap::new();
        let list = vec![
            Assign, ShiftLeftAssign, ShiftRightAssign,
            BitwiseAndAssign, BitwiseOrAssign, DivideAssign,
            MultiplyAssign, AddAssign, SubtractAssign, ModuloAssign,
        ];
        for op in list {
            assignment_operators.insert(op, op.as_str());
        }
        assignment_operators
    }
}



// Adding the data types from Python to Rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataType {
    F64,  // 64bit floating point type. 8bytes wide.
    U64,  // Unsigned 64bit Integer type. 8bytes wide.
    I64,  // Signed 64bit Integer type. 8bytes wide.
    U32,  // Unsigned 32bit Integer type. 4bytes wide.
    I32,  // Signed 32bit Integer type. 4bytes wide.
    U16,  // Unsigned 16bit Integer type. 2bytes wide.
    I16,  // Signed 16bit Integer type. 2bytes wide.
    U8,   // Unsigned 8bit Integer type. 1byte wide.
    I8,   // Signed 8bit Integer type
    Bool, // Boolean type, 1 byte wide (should be 0 or 1).
    U0,   // Void type, has no size.
    F32,  // 32bit floating point
}

impl fmt::Display for DataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            DataType::F64 => "F64",
            DataType::U64 => "U64",
            DataType::I64 => "I64",
            DataType::U32 => "U32",
            DataType::I32 => "I32",
            DataType::U16 => "U16",
            DataType::I16 => "I16",
            DataType::U8 => "U8",
            DataType::I8 => "I8",
            DataType::Bool => "Bool",
            DataType::U0 => "U0",
            DataType::F32 => "F32",
        };
        write!(f, "{}", s)
    }
}

#[derive(Debug, Clone)]
pub struct TypeInfo {
    pub is_signed: bool,
    pub size_in_bits: u32,
    pub is_float: bool,
    pub is_void: bool,
}

pub struct DataTypes;

impl DataTypes {
    pub fn get_all() -> HashMap<DataType, String> {
        use DataType::*;
        let mut types = HashMap::new();
        types.insert(F64, "F64".to_string());
        types.insert(U64, "U64".to_string());
        types.insert(I64, "I64".to_string());
        types.insert(U32, "U32".to_string());
        types.insert(I32, "I32".to_string());
        types.insert(U16, "U16".to_string());
        types.insert(I16, "I16".to_string());
        types.insert(U8, "U8".to_string());
        types.insert(I8, "I8".to_string());
        types.insert(Bool, "Bool".to_string());
        types.insert(U0, "U0".to_string());
        types.insert(F32, "F32".to_string());
        types
    }

    pub fn get_type_info(data_type: &DataType) -> TypeInfo {
        match data_type {
            DataType::F64 => TypeInfo { is_signed: true, size_in_bits: 64, is_float: true, is_void: false },
            DataType::U64 => TypeInfo { is_signed: false, size_in_bits: 64, is_float: false, is_void: false },
            DataType::I64 => TypeInfo { is_signed: true, size_in_bits: 64, is_float: false, is_void: false },
            DataType::U32 => TypeInfo { is_signed: false, size_in_bits: 32, is_float: false, is_void: false },
            DataType::I32 => TypeInfo { is_signed: true, size_in_bits: 32, is_float: false, is_void: false },
            DataType::U16 => TypeInfo { is_signed: false, size_in_bits: 16, is_float: false, is_void: false },
            DataType::I16 => TypeInfo { is_signed: true, size_in_bits: 16, is_float: false, is_void: false },
            DataType::U8 => TypeInfo { is_signed: false, size_in_bits: 8, is_float: false, is_void: false },
            DataType::I8 => TypeInfo { is_signed: true, size_in_bits: 8, is_float: false, is_void: false },
            DataType::Bool => TypeInfo { is_signed: false, size_in_bits: 8, is_float: false, is_void: false },
            DataType::U0 => TypeInfo { is_signed: false, size_in_bits: 0, is_float: false, is_void: true },
            DataType::F32 => TypeInfo { is_signed: true, size_in_bits: 32, is_float: true, is_void: false },
        }
    }

    pub fn is_signed_type(data_type: &DataType) -> bool {
        Self::get_type_info(data_type).is_signed
    }

    pub fn is_integer_type(data_type: &DataType) -> bool {
        let info = Self::get_type_info(data_type);
        !info.is_float && !info.is_void
    }

    pub fn is_float_type(data_type: &DataType) -> bool {
        Self::get_type_info(data_type).is_float
    }

    pub fn from_string(type_name: &str) -> Option<DataType> {
        match type_name {
            "F64" => Some(DataType::F64),
            "U64" => Some(DataType::U64),
            "I64" => Some(DataType::I64),
            "U32" => Some(DataType::U32),
            "I32" => Some(DataType::I32),
            "U16" => Some(DataType::U16),
            "I16" => Some(DataType::I16),
            "U8" => Some(DataType::U8),
            "I8" => Some(DataType::I8),
            "Bool" => Some(DataType::Bool),
            "U0" => Some(DataType::U0),
            "F32" => Some(DataType::F32),
            _ => None,
        }
    }
}

// Add keywords from Python to Rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Keyword {
    // Data types
    F64,
    U64,
    I64,
    U32,
    I32,
    U16,
    I16,
    U8,
    Bool,
    U0,
    // Control flow
    If,
    Else,
    Switch,
    Do,
    While,
    For,
    Break,
    Continue,
    Goto,
    Return,
    // Object oriented
    Class,
    Union,
    // Other
    Asm,
    Public,
    Extern,
    // Constants
    True,
    False,
}

impl fmt::Display for Keyword {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Keyword::F64 => "F64",
            Keyword::U64 => "U64",
            Keyword::I64 => "I64",
            Keyword::U32 => "U32",
            Keyword::I32 => "I32",
            Keyword::U16 => "U16",
            Keyword::I16 => "I16",
            Keyword::U8 => "U8",
            Keyword::Bool => "Bool",
            Keyword::U0 => "U0",
            Keyword::If => "if",
            Keyword::Else => "else",
            Keyword::Switch => "switch",
            Keyword::Do => "do",
            Keyword::While => "while",
            Keyword::For => "for",
            Keyword::Break => "break",
            Keyword::Continue => "continue",
            Keyword::Goto => "goto",
            Keyword::Return => "return",
            Keyword::Class => "class",
            Keyword::Union => "union",
            Keyword::Asm => "asm",
            Keyword::Public => "public",
            Keyword::Extern => "extern",
            Keyword::True => "true",
            Keyword::False => "false",
        };
        write!(f, "{}", s)
    }
}

pub struct Keywords;

impl Keywords {
    pub fn get_all() -> HashMap<Keyword, &'static str> {
        use Keyword::*;
        let mut keywords = HashMap::new();
        keywords.insert(F64, "F64");
        keywords.insert(U64, "U64");
        keywords.insert(I64, "I64");
        keywords.insert(U32, "U32");
        keywords.insert(I32, "I32");
        keywords.insert(U16, "U16");
        keywords.insert(I16, "I16");
        keywords.insert(U8, "U8");
        keywords.insert(Bool, "Bool");
        keywords.insert(U0, "U0");
        keywords.insert(If, "if");
        keywords.insert(Else, "else");
        keywords.insert(Switch, "switch");
        keywords.insert(Do, "do");
        keywords.insert(While, "while");
        keywords.insert(For, "for");
        keywords.insert(Break, "break");
        keywords.insert(Continue, "continue");
        keywords.insert(Goto, "goto");
        keywords.insert(Return, "return");
        keywords.insert(Class, "class");
        keywords.insert(Union, "union");
        keywords.insert(Asm, "asm");
        keywords.insert(Public, "public");
        keywords.insert(Extern, "extern");
        keywords.insert(True, "true");
        keywords.insert(False, "false");
        keywords
    }

    pub fn from_string(keyword: &str) -> Option<Keyword> {
        match keyword {
            "F64" => Some(Keyword::F64),
            "U64" => Some(Keyword::U64),
            "I64" => Some(Keyword::I64),
            "U32" => Some(Keyword::U32),
            "I32" => Some(Keyword::I32),
            "U16" => Some(Keyword::U16),
            "I16" => Some(Keyword::I16),
            "U8" => Some(Keyword::U8),
            "Bool" => Some(Keyword::Bool),
            "U0" => Some(Keyword::U0),
            "if" => Some(Keyword::If),
            "else" => Some(Keyword::Else),
            "switch" => Some(Keyword::Switch),
            "do" => Some(Keyword::Do),
            "while" => Some(Keyword::While),
            "for" => Some(Keyword::For),
            "break" => Some(Keyword::Break),
            "continue" => Some(Keyword::Continue),
            "goto" => Some(Keyword::Goto),
            "return" => Some(Keyword::Return),
            "class" => Some(Keyword::Class),
            "union" => Some(Keyword::Union),
            "asm" => Some(Keyword::Asm),
            "public" => Some(Keyword::Public),
            "extern" => Some(Keyword::Extern),
            "true" => Some(Keyword::True),
            "false" => Some(Keyword::False),
            _ => None,
        }
    }
}

// Add separators from Python to Rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Separator {
    Comma,       // ,
    Dot,         // .
    Colon,       // :
    Semicolon,   // ;
    LParen,      // (
    RParen,      // )
    LBracket,    // [
    RBracket,    // ]
    LBrace,      // {
    RBrace,      // }
    Arrow,       // ->
}

impl fmt::Display for Separator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Separator::Comma => ",",
            Separator::Dot => ".",
            Separator::Colon => ":",
            Separator::Semicolon => ";",
            Separator::LParen => "(",
            Separator::RParen => ")",
            Separator::LBracket => "[",
            Separator::RBracket => "]",
            Separator::LBrace => "{",
            Separator::RBrace => "}",
            Separator::Arrow => "->",
        };
        write!(f, "{}", s)
    }
}

pub struct Separators;

impl Separators {
    pub fn get_all() -> HashMap<Separator, &'static str> {
        use Separator::*;
        let mut separators = HashMap::new();
        separators.insert(Comma, ",");
        separators.insert(Dot, ".");
        separators.insert(Colon, ":");
        separators.insert(Semicolon, ";");
        separators.insert(LParen, "(");
        separators.insert(RParen, ")");
        separators.insert(LBracket, "[");
        separators.insert(RBracket, "]");
        separators.insert(LBrace, "{");
        separators.insert(RBrace, "}");
        separators.insert(Arrow, "->");
        separators
    }

    pub fn from_string(separator: &str) -> Option<Separator> {
        match separator {
            "," => Some(Separator::Comma),
            "." => Some(Separator::Dot),
            ":" => Some(Separator::Colon),
            ";" => Some(Separator::Semicolon),
            "(" => Some(Separator::LParen),
            ")" => Some(Separator::RParen),
            "[" => Some(Separator::LBracket),
            "]" => Some(Separator::RBracket),
            "{" => Some(Separator::LBrace),
            "}" => Some(Separator::RBrace),
            "->" => Some(Separator::Arrow),
            _ => None,
        }
    }
}

// Update TokenType to include DataType for completeness
