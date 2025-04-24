// src/main.rs
use hhcc::lexer::Lexer;

fn main() {
    let mut lexer = Lexer::new("U32 main() { return 0; }".to_string());
    let tokens = lexer.tokenize().unwrap();

    for token in tokens {
        println!("{:?}: {}", token.token_type, token.value);
    }
}
