# Holy HolyC Compiler (hhcc) - Usage Guide

## Overview

Holy HolyC Compiler (hhcc) is a compiler for the HolyHolyC programming language. This guide provides information on how to use the compiler, its available options, and basic workflow.

## Basic Usage

```
hhcc [OPTIONS] <FILE>
```

Where `<FILE>` is the path to your HolyC source file.

## Command Line Options

| Option                | Short | Description                                                                |
| --------------------- | ----- | -------------------------------------------------------------------------- |
| `--debug`             | `-d`  | Enable debugging output                                                    |
| `--dump-ast`          | `-a`  | Dump the Abstract Syntax Tree (AST)                                        |
| `--dump-tokens`       | `-t`  | Dump the token list from lexical analysis                                  |
| `--dump-defines`      | `-f`  | Dump preprocessor defines                                                  |
| `--dump-preprocessed` | `-p`  | Dump preprocessed source code                                              |
| `--dump-llvmir`       | `-l`  | Dump generated LLVM IR code                                                |
| `--target TARGET`     |       | Specify target architecture, OS, and ABI in the format `<arch>-<os>-<abi>` |
| `--triple TRIPLE`     |       | Alternative way to specify target                                          |
| `-S`                  |       | Compile only; do not assemble or link                                      |
| `--emit-llvm`         |       | Generate LLVM IR code                                                      |
| `--output FILE`       | `-o`  | Specify output file name                                                   |
| `--help`              | `-h`  | Print help information                                                     |
| `--version`           | `-V`  | Print version information                                                  |

## Examples

### Basic compilation

```
hhcc source.hc
```

### Compile with debugging information

```
hhcc -d source.hc
```

### Generate LLVM IR and save to a file

```
hhcc -S --emit-llvm -o output.ll source.hc
```

### View tokens from lexical analysis

```
hhcc --dump-tokens source.hc
```

### View preprocessed code

```
hhcc --dump-preprocessed source.hc
```

### Specify target architecture

```
hhcc --target x86_64-unknown-linux-gnu source.hc
```

## Compilation Phases

The compiler operates in several phases:

1. **Preprocessing**: Processes include directives and macros
2. **Lexical Analysis**: Converts source code into tokens
3. **Parsing**: Builds an Abstract Syntax Tree (AST)
4. **Code Generation**: Generates LLVM IR from the AST

You can inspect the output of each phase using the corresponding dump options.

## Exit Codes

- `0`: Successful compilation
- `1`: Error occurred during compilation

## Project Structure

The Rust implementation is organized into several modules:

- `compiler.rs`: Main compiler implementation
- `lexer.rs`: Lexical analysis module
- `types.rs`: Common type definitions
- (To be implemented) `astparser.rs`: AST parsing module
- (To be implemented) `preprocessor.rs`: Preprocessor module
- (To be implemented) `target.rs`: Target architecture handling
- (To be implemented) `codegen.rs`: LLVM IR code generation module

## Dependencies

- clap: Command line argument parsing
