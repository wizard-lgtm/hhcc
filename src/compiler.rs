use std::fs;
use std::path::{Path, PathBuf};
use std::error::Error;
use std::collections::HashMap;

use crate::lexer::Lexer;
use crate::types::Token;
// These will be uncommented when implemented
// use crate::astparser::{ASTParser, ASTNode};
// use crate::preprocessor::Preprocessor;
// use crate::target::Target;
// use crate::codegen::Codegen;

#[derive(Debug)]
pub struct Compiler {
    file: PathBuf,
    file_directory: PathBuf,
    working_directory: PathBuf,
    src: String,
    tokens: Vec<Token>,
    // astnodes: Vec<ASTNode>,
    debug: bool,
    dump_ast: bool,
    dump_tokens: bool,
    dump_defines: bool,
    dump_preprocessed: bool,
    dump_llvmir: bool,
    defines: HashMap<String, String>,
    triple: Option<String>,
    target: Option<String>, // Will be Target type once implemented
    output_file: Option<String>,
    version: String,
}

impl Compiler {
    pub fn new(
        file: &str,
        debug: bool,
        dump_ast: bool,
        dump_tokens: bool,
        dump_defines: bool,
        dump_preprocessed: bool,
        dump_llvmir: bool,
        triple: Option<&str>,
        target: Option<&str>,
        output_file: Option<&str>,
    ) -> Result<Self, Box<dyn Error>> {
        let file_path = Path::new(file).canonicalize()?;
        let file_directory = file_path.parent().unwrap_or(Path::new("")).to_path_buf();
        let working_directory = std::env::current_dir()?;
        
        let src = fs::read_to_string(&file_path)?;
        
        println!("hhcc compiler version: 0.0.4");
        
        if debug {
            println!("Compiling file: {}", file_path.display());
            println!("Working directory: {}", working_directory.display());
            println!("File directory: {}", file_directory.display());
        }
        
        // Create compiler instance
        let compiler = Compiler {
            file: file_path,
            file_directory,
            working_directory,
            src,
            tokens: Vec::new(),
            // astnodes: Vec::new(),
            debug,
            dump_ast,
            dump_tokens,
            dump_defines,
            dump_preprocessed,
            dump_llvmir,
            defines: HashMap::new(),
            triple: triple.map(String::from),
            target: target.map(String::from),
            output_file: output_file.map(String::from),
            version: String::from("0.0.4"),
        };
        
        if debug {
            println!("Selected Target: {:?}", compiler.target);
        }
        
        Ok(compiler)
    }
    
    pub fn compile(&mut self) -> Result<(), Box<dyn Error>> {
        // 1. Preprocessor
        if self.debug {
            println!("Running Preprocessor");
        }
        
        // Temporary placeholder for preprocessor
        let processed_code = self.src.clone();
        // let preprocessor = Preprocessor::new(self);
        // let processed_code = preprocessor.preprocess();
        
        if self.dump_preprocessed {
            println!("==== Preprocessed Code ====");
            println!("{}", processed_code);
            println!("============================");
        }
        
        if self.dump_defines {
            println!("======Defines==========");
            for (key, value) in &self.defines {
                println!("{}: {}", key, value);
            }
            println!("================");
        }
        
        // Update the source after preprocessing
        self.src = processed_code;
        
        // 2. Lexical Analysis
        let mut lexer = Lexer::new(self.src.clone());
        if self.debug {
            println!("==== Running Lexer ====");
        }
        
        // Adjust this line based on your actual Lexer implementation
        self.tokens = lexer.tokenize().unwrap();
        
        if self.dump_tokens {
            println!("==== Tokens ====");
            for token in &self.tokens {
                println!("{:?}", token);
            }
            println!("================");
        }
        
        // 3. AST Parsing - commented out until implemented
        if self.debug {
            println!("==== Running AST Parser ====");
        }
        
        /*
        let mut astparser = ASTParser::new(&self.src, self);
        astparser.load_tokens(&self.tokens);
        let nodes = astparser.parse();
        self.astnodes = nodes;
        
        if self.dump_ast {
            println!("==== AST Nodes ====");
            for node in &self.astnodes {
                println!("{:?}", node);
            }
            println!("==================");
        }
        */
        
        // 4. LLVM IR Generation - commented out until implemented
        if self.debug {
            println!("==== Running Codegen ====");
        }
        
        /*
        let codegen = Codegen::new(self);
        let llvmir = codegen.gen();
        
        // Output LLVM IR to file if specified
        if let Some(output_file) = &self.output_file {
            fs::write(output_file, llvmir.to_string())?;
            println!("LLVM IR written to {}", output_file);
        }
        
        if self.dump_llvmir {
            println!("==== Generated LLVM IR Code ====");
            println!("{}", llvmir);
            println!("==================");
        }
        */
        
        println!("Done!");
        Ok(())
    }
}