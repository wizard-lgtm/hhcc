// Holy HolyC Compiler
// https://github.com/wizard-lgtm/hhcc
// Licensed under MIT

use std::process;
use clap::{Arg, Command};
use hhcc::lexer::Lexer;
use hhcc::compiler::Compiler;

static VERSION: &str = "0.0.6";

fn main() {

    if std::env::args().len() <= 1 {
        println!("Holy Holy C Compiler version {}", VERSION);
        eprintln!("No input file provided.\nTry `hhcc <file.hc>` or `hhcc --help` for usage.");
        std::process::exit(1);
    }
    // Otherwise, run the compiler with CLI arguments
    let matches = Command::new("Holy HolyC Compiler")
        .version(VERSION)
        .author("wizard-lgtm")
        .about("compiler for HolyC language")
        .arg(
            Arg::new("file")
                .help("Source file to compile")
                .required(true)
                .index(1),
        )
        .arg(
            Arg::new("debug")
                .short('d')
                .long("debug")
                .help("Enable debugging output")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("dump-ast")
                .long("dump-ast")
                .short('a')
                .help("Dump the AST tree")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("dump-tokens")
                .long("dump-tokens")
                .short('t')
                .help("Dump the token list")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("dump-defines")
                .long("dump-defines")
                .short('f')
                .help("Dump defines")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("dump-preprocessed")
                .long("dump-preprocessed")
                .short('p')
                .help("Dump Preprocessed Code")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("dump-llvmir")
                .long("dump-llvmir")
                .short('l')
                .help("Dump LLVM IR")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("target")
                .long("target")
                .help("Target in the format <arch>-<os>-<abi>. Default is native target.")
                .value_name("TARGET"),
        )
        .arg(
            Arg::new("triple")
                .long("triple")
                .help("Target in the format <arch>-<os>-<abi>. Default is native target.")
                .value_name("TRIPLE"),
        )
        .arg(
            Arg::new("S")
                .short('S')
                .help("Compile only; do not assemble or link")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("emit-llvm")
                .long("emit-llvm")
                .help("Generate LLVM IR code")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("output")
                .short('o')
                .long("output")
                .help("Output file name")
                .value_name("FILE"),
        )
        .get_matches();

    let file = matches.get_one::<String>("file").unwrap();
    let debug = matches.get_flag("debug");
    let dump_ast = matches.get_flag("dump-ast");
    let dump_tokens = matches.get_flag("dump-tokens");
    let dump_defines = matches.get_flag("dump-defines");
    let dump_preprocessed = matches.get_flag("dump-preprocessed");
    let dump_llvmir = matches.get_flag("dump-llvmir") || 
                    (matches.get_flag("S") && matches.get_flag("emit-llvm"));
    
    let triple = matches.get_one::<String>("triple").map(|s| s.as_str());
    let target = matches.get_one::<String>("target").map(|s| s.as_str());
    
    // Determine output file
    let mut output_file = matches.get_one::<String>("output").map(|s| s.as_str());
    if matches.get_flag("S") && matches.get_flag("emit-llvm") && output_file.is_none() {
        output_file = Some("output.ll");
    }

    match Compiler::new(
        file,
        debug,
        dump_ast,
        dump_tokens,
        dump_defines,
        dump_preprocessed,
        dump_llvmir,
        triple,
        target,
        output_file,
    ) {
        Ok(mut compiler) => {
            if let Err(err) = compiler.compile() {
                eprintln!("Compilation error: {}", err);
                process::exit(1);
            }
        },
        Err(err) => {
            eprintln!("Error initializing compiler: {}", err);
            process::exit(1);
        }
    }
}

// Original test function
fn test_lexer() {
    let mut lexer = Lexer::new("U32 main() { return 0; }".to_string());
    let tokens = lexer.tokenize().unwrap();
    for token in tokens {
        println!("{:?}: {}", token.token_type, token.value);
    }
}