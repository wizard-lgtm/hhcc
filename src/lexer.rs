use std::collections::HashMap;
use std::fmt;
use std::error::Error;

pub use crate::types::TokenType;

// Assuming the types module has been created from our previous implementation
use crate::types::{
    DataType, DataTypes, Keyword, Keywords, 
    Operator, Operators, Separator, Separators,
    Token
};

// Custom error type for lexer errors
#[derive(Debug)]
pub enum LexerError {
    UnrecognizedToken(char, usize, usize),
    UnterminatedString(usize, usize),
    InvalidDirective(String, usize, usize),
}

impl fmt::Display for LexerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LexerError::UnrecognizedToken(c, line, col) => 
                write!(f, "Unrecognized token '{}' at line {}, column {}", c, line, col),
            LexerError::UnterminatedString(line, col) => 
                write!(f, "Unterminated string at line {}, column {}", line, col),
            LexerError::InvalidDirective(dir, line, col) => 
                write!(f, "Invalid directive '{}' at line {}, column {}", dir, line, col),
        }
    }
}

impl Error for LexerError {}

// Source location tracking
#[derive(Debug, Clone, Copy)]
pub struct SourceLocation {
    pub line: usize,
    pub column: usize,
    pub index: usize,
}

impl SourceLocation {
    pub fn new(line: usize, column: usize, index: usize) -> Self {
        Self { line, column, index }
    }
}

pub struct Lexer {
    source_code: String,
    cursor: usize,
    line: usize,
    column: usize,
    keywords: HashMap<String, Keyword>,
    operators: Vec<(Operator, String)>,  // Sorted by length for greedy matching
    separators: HashMap<String, Separator>,
    data_types: HashMap<String, DataType>,
}

impl Lexer {
    pub fn new(source_code: String) -> Self {
        // Initialize with empty maps
        let mut lexer = Self {
            source_code,
            cursor: 0,
            line: 1,
            column: 0,
            keywords: HashMap::new(),
            operators: Vec::new(),
            separators: HashMap::new(),
            data_types: HashMap::new(),
        };
        
        // Prepare lookup tables
        lexer.initialize_tables();
        
        lexer
    }
    
    fn initialize_tables(&mut self) {
        // Initialize keywords lookup (string -> enum)
        for (keyword, keyword_str) in Keywords::get_all() {
            self.keywords.insert(keyword_str.to_string(), keyword);
        }
        
        // Initialize operators lookup (sorted by length for greedy matching)
        for (op, op_str) in Operators::get_all() {
            self.operators.push((op, op_str.to_string()));
        }
        // Sort by length in descending order
        self.operators.sort_by(|a, b| b.1.len().cmp(&a.1.len()));
        
        // Initialize separators lookup
        for (sep, sep_str) in Separators::get_all() {
            self.separators.insert(sep_str.to_string(), sep);
        }
        
        // Initialize data types lookup
        for (type_val, type_str) in DataTypes::get_all() {
            self.data_types.insert(type_str.to_string(), type_val); // Changed to store with to_string()
        }
    }
        
    // Get current position as a SourceLocation
    fn current_location(&self) -> SourceLocation {
        SourceLocation::new(self.line, self.column, self.cursor)
    }
    
    // Check if we've reached the end of the input
    fn is_at_end(&self) -> bool {
        self.cursor >= self.source_code.len()
    }
    
    // Peek at the current character without advancing
    fn peek(&self) -> Option<char> {
        if self.is_at_end() {
            None
        } else {
            Some(self.source_code.as_bytes()[self.cursor] as char)
        }
    }
    
    // Peek at the next character without advancing
    fn peek_next(&self) -> Option<char> {
        if self.cursor + 1 >= self.source_code.len() {
            None
        } else {
            Some(self.source_code.as_bytes()[self.cursor + 1] as char)
        }
    }
    
    // Advance the cursor and return the character
    fn advance(&mut self) -> Option<char> {
        if self.is_at_end() {
            return None;
        }
        
        let c = self.source_code.as_bytes()[self.cursor] as char;
        self.cursor += 1;
        self.column += 1;
        
        Some(c)
    }
    
    // Match and consume a string if it matches the current position
    fn match_string(&mut self, s: &str) -> bool {
        if self.cursor + s.len() > self.source_code.len() {
            return false;
        }
        
        if &self.source_code[self.cursor..self.cursor + s.len()] == s {
            for _ in 0..s.len() {
                self.advance();
            }
            return true;
        }
        
        false
    }
    
    pub fn tokenize(&mut self) -> Result<Vec<Token>, LexerError> {
        let mut tokens = Vec::new();
        
        while !self.is_at_end() {
            // Store starting position for this token
            let start_pos = self.current_location();
            
            match self.peek() {
                // Skip whitespace
                Some(c) if c.is_whitespace() => {
                    if c == '\n' {
                        self.line += 1;
                        self.column = 0;
                    }
                    self.advance();
                }
                
                // Handle directives
                Some('#') => {
                    self.tokenize_directive(&mut tokens, start_pos)?;
                }
                
                // Handle identifiers, keywords, and types
                Some(c) if c.is_alphabetic() || c == '_' => {
                    self.tokenize_identifier(&mut tokens, start_pos);
                }
                
                // Handle numeric literals
                Some(c) if c.is_digit(10) => {
                    self.tokenize_number(&mut tokens, start_pos);
                }
                
                // Handle string literals
                Some('"') => {
                    self.tokenize_string(&mut tokens, start_pos)?;
                }
                
                // Handle comments
                Some('/') if self.peek_next() == Some('/') => {
                    self.tokenize_comment(&mut tokens, start_pos);
                }
                
                // Handle operators and separators
                Some(_) => {
                    // Try matching operators (longest first)
                    let mut matched = false;
                    
                    let operators = self.operators.clone();

                    // Try operators
                    for (_op, op_str) in operators {
                        if self.match_string(op_str.as_str()) {
                            tokens.push(Token::new(
                                TokenType::Operator, 
                                op_str.clone(), 
                                start_pos.line, 
                                start_pos.column
                            ));
                            matched = true;
                            break;
                        }
                    }
                    
                    let separators = self.separators.clone();

                    // Try separators if no operator matched
                    if !matched {
                        for (sep_str, _sep) in separators {
                            if self.match_string(sep_str.as_str()) {
                                tokens.push(Token::new(
                                    TokenType::Separator, 
                                    sep_str.clone(), 
                                    start_pos.line, 
                                    start_pos.column
                                ));
                                matched = true;
                                break;
                            }
                        }
                    }
                    
                    // If nothing matched, report an error
                    if !matched {
                        if let Some(c) = self.advance() {
                            return Err(LexerError::UnrecognizedToken(
                                c, start_pos.line, start_pos.column
                            ));
                        }
                    }
                }
                
                None => break, // End of input
            }
        }
        
        Ok(tokens)
    }
    
    fn tokenize_directive(&mut self, tokens: &mut Vec<Token>, start_pos: SourceLocation) -> Result<(), LexerError> {
        // Consume the '#'
        self.advance();
        
        // Read the directive name
        let mut directive_name = String::new();
        while let Some(c) = self.peek() {
            if c.is_alphanumeric() {
                directive_name.push(c);
                self.advance();
            } else {
                break;
            }
        }
        
        // Skip whitespace after directive name
        while let Some(c) = self.peek() {
            if c.is_whitespace() && c != '\n' {
                self.advance();
            } else {
                break;
            }
        }
        
        // Read the directive value until end of line
        let mut directive_value = String::new();
        while let Some(c) = self.peek() {
            if c != '\n' {
                directive_value.push(c);
                self.advance();
            } else {
                break;
            }
        }
        
        // Combine into a single directive token
        let full_directive = format!("#{}{}", directive_name, directive_value);
        tokens.push(Token::new(
            TokenType::Directive,
            full_directive,
            start_pos.line,
            start_pos.column
        ));
        
        Ok(())
    }
        
    fn tokenize_identifier(&mut self, tokens: &mut Vec<Token>, start_pos: SourceLocation) {
        let mut identifier = String::new();
        
        // Read all alphanumeric characters or underscores
        while let Some(c) = self.peek() {
            if c.is_alphanumeric() || c == '_' {
                identifier.push(c);
                self.advance();
            } else {
                break;
            }
        }
        
        // First check if it's a data type
        if self.data_types.contains_key(&identifier) {
            tokens.push(Token::new(
                TokenType::DataType,
                identifier,
                start_pos.line,
                start_pos.column
            ));
        }
        // Then check if it's a keyword
        else if let Some(&keyword) = self.keywords.get(&identifier) {
            if keyword == Keyword::True || keyword == Keyword::False {
                tokens.push(Token::new(
                    TokenType::Literal,
                    identifier,
                    start_pos.line,
                    start_pos.column
                ));
            } else {
                tokens.push(Token::new(
                    TokenType::Keyword,
                    identifier,
                    start_pos.line,
                    start_pos.column
                ));
            }
        }
        // It's a regular identifier
        else {
            tokens.push(Token::new(
                TokenType::Identifier,
                identifier,
                start_pos.line,
                start_pos.column
            ));
        }
    }
    
    fn tokenize_number(&mut self, tokens: &mut Vec<Token>, start_pos: SourceLocation) {
        let mut number = String::new();
        let mut has_decimal = false;
        
        // Read digits and potentially one decimal point
        while let Some(c) = self.peek() {
            if c.is_digit(10) {
                number.push(c);
                self.advance();
            } else if c == '.' && !has_decimal {
                number.push(c);
                has_decimal = true;
                self.advance();
            } else {
                break;
            }
        }
        
        tokens.push(Token::new(
            TokenType::Literal,
            number,
            start_pos.line,
            start_pos.column
        ));
    }
    
    fn tokenize_string(&mut self, tokens: &mut Vec<Token>, start_pos: SourceLocation) -> Result<(), LexerError> {
        // Consume the opening quote
        self.advance();
        
        let mut string_content = String::new();
        
        // Read until closing quote or end of input
        while let Some(c) = self.peek() {
            if c == '"' {
                // Consume the closing quote
                self.advance();
                
                // Create the token including quotes
                tokens.push(Token::new(
                    TokenType::Literal,
                    format!("\"{}\"", string_content),
                    start_pos.line,
                    start_pos.column
                ));
                
                return Ok(());
            } else if c == '\n' {
                // Strings can't contain unescaped newlines
                return Err(LexerError::UnterminatedString(start_pos.line, start_pos.column));
            } else if c == '\\' {
                // Handle escape sequences
                self.advance(); // Consume the backslash
                
                if let Some(next_c) = self.peek() {
                    match next_c {
                        'n' => string_content.push('\n'),
                        't' => string_content.push('\t'),
                        'r' => string_content.push('\r'),
                        '\\' => string_content.push('\\'),
                        '"' => string_content.push('"'),
                        _ => string_content.push(next_c), // Just add the character
                    }
                    self.advance();
                }
            } else {
                string_content.push(c);
                self.advance();
            }
        }
        
        // If we get here, the string wasn't terminated
        Err(LexerError::UnterminatedString(start_pos.line, start_pos.column))
    }
    
    fn tokenize_comment(&mut self, tokens: &mut Vec<Token>, start_pos: SourceLocation) {
        // Consume the two forward slashes
        self.advance();
        self.advance();
        
        let mut comment = "//".to_string();
        
        // Read until end of line or end of input
        while let Some(c) = self.peek() {
            if c == '\n' {
                break;
            }
            comment.push(c);
            self.advance();
        }
        
        tokens.push(Token::new(
            TokenType::Comment,
            comment,
            start_pos.line,
            start_pos.column
        ));
    }
}

// Example usage
#[cfg(test)]
mod tests {
    use super::*;
    
 #[test]
fn debug_token_counts() {
    // Test the first example
    let source = "U32 main() { return 0; }".to_string();
    let mut lexer = Lexer::new(source);
    let tokens = lexer.tokenize().unwrap();
    
    println!("Basic test token count: {}", tokens.len());
    for (i, token) in tokens.iter().enumerate() {
        println!("Token {}: {:?} - '{}'", i, token.token_type, token.value);
    }
    
    // Expected tokens:
    // 1. U32 (DataType)
    // 2. main (Identifier)
    // 3. ( (Separator)
    // 4. ) (Separator)
    // 5. { (Separator)
    // 6. return (Keyword)
    // 7. 0 (Literal)
    // 8. ; (Separator)
    // 9. } (Separator)
    
    assert_eq!(tokens.len(), 9); // Changed from 8 to 9
    assert_eq!(tokens[0].token_type, TokenType::DataType);
    assert_eq!(tokens[1].token_type, TokenType::Identifier);
}

#[test]
fn check_data_type_registration() {
    // Create a lexer to check data type registration
    let source = "".to_string();
    let lexer = Lexer::new(source);
    
    // Print out what's in the data_types map
    println!("Data types registered:");
    for (key, _) in &lexer.data_types {
        println!("  - {}", key);
    }
    
    // Ensure U32 is in the map
    assert!(lexer.data_types.contains_key("U32"), "U32 should be in data_types map");
}
}