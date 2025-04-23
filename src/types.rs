use std::collections::HashMap;
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum TokenType {
    Keyword,
    Separator,
    Operator,
    Literal,
    Comment,
    Whitespace,
    Directive,
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