# Holy HolyC Compiler
# https://github.com/wizard-lgtm/hhcc
# Licensed under MIT
import argparse
import os
from lexer import *
from astparser2 import *
from preprocessor import Preprocessor
from target import Target
from codegen import Codegen

import platform 
class Compiler:
    file = str
    src = str
    tokens = list[Token]
    astnodes = list[ASTNode]
    debug = bool
    dump_ast = bool
    dump_tokens = bool
    dump_llvmir = bool
    preprocessor: Preprocessor
    lexer: Lexer
    astparser: ASTParser
    target: Target
    
    def __init__(self, file, debug=False, dump_ast=False, dump_tokens=False, dump_defines=False, dump_preprocessed=False, dump_llvmir=False, triple=None, target=None):
        self.version = "0.0.4"
        self.file = os.path.abspath(file)
        self.file_directory = os.path.dirname(self.file)
        self.working_directory = os.getcwd()
        self.debug = debug
        self.dump_ast = dump_ast
        self.dump_tokens = dump_tokens
        self.dump_defines = dump_defines
        self.dump_preprocessed = dump_preprocessed
        self.dump_llvmir = dump_llvmir
        self.defines = {} # TODO! -> LLVM IR generate step (constant defines)
        self.triple = triple
        
        print(f"hhcc compiler version: {self.version}")

        if self.debug:
            print(f"Compiling file: {self.file}")
            print(f"Working directory: {self.working_directory}")
            print(f"File directory: {self.file_directory}")
        
        with open(file, "r") as src:
            self.src = src.read()
        
        self.preprocessor = Preprocessor(self)
        self.astparser = ASTParser(self.src, self)

        # Set the target architecture, OS, and ABI

        self.target = target

        if self.debug: 
            print(f"Selected Target: {self.target}")
    
    def compile(self):
        # 1. Preprocessor
        if self.debug:
            print("Running Preprocessor")
        processed_code = self.preprocessor.preprocess()
        
        if self.dump_preprocessed:
            print("==== Preprocessed Code ====")
            print(processed_code)
            print("============================")
        
        if self.dump_defines:
            print("======Defines==========")
            for items in self.defines.items():
                print(items)
            print("================")
        # Update the source after preprocessing
        self.src = processed_code
        
        # 2. Lexical Analysis
        self.lexer = Lexer(self)
        if self.debug:
            print("==== Running Lexer ====")
        self.tokens = self.lexer.tokenize()
        if self.dump_tokens:
            print("==== Tokens ====")
            for token in self.tokens:
                token.print()
            print("================")
        
        # 3. AST Parsing
        if self.debug:
            print("==== Running AST Parser ====")
        self.astparser.load_tokens(self.tokens)
        nodes = self.astparser.parse()
        if self.dump_ast:
            print("==== AST Nodes ====")
            for node in nodes:
                print(node)
            print("==================")
        
        # 4. LLVM IR Generation
        if self.debug:
            print("==== Running Codegen ====")
        codegen = Codegen(self)
        llvmir = codegen.gen()
        if self.dump_llvmir:
            print("==== Generated LLVM IR Code ====")
            print(llvmir)
            print("==================")

        print("Done!")
        return nodes

def parse_args():
    parser = argparse.ArgumentParser(description="Holy HolyC Compiler")
    parser.add_argument("file", help="Source file to compile")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debugging output")
    parser.add_argument("-da", "--dump-ast", action="store_true", help="Dump the AST tree")
    parser.add_argument("-dt", "--dump-tokens", action="store_true", help="Dump the token list")
    parser.add_argument("-df", "--dump-defines", action="store_true", help="Dump defines")
    parser.add_argument("-dp", "--dump-preprocessed", action="store_true", help="Dump Preprocessed Code")
    parser.add_argument("-dl", "--dump-llvmir", action="store_true", help="Dump llvm_ir")
    parser.add_argument("--target", help="Target in the format <arch>-<os>-<abi>. Default is native target.", default=None)
    parser.add_argument("--triple", help="Target in the format <arch>-<os>-<abi>. Default is native target.", default=None)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Parse the target argument if provided
    target = None
    if args.target:
        try:
            # Try to create a Target using the string from the command line
            target = Target.from_string(args.target)
        except ValueError as e:
            print(f"Error parsing target: {e}")
            exit(1)

    # Create the compiler instance with the parsed arguments
    compiler = Compiler(args.file, debug=args.debug, dump_ast=args.dump_ast, dump_tokens=args.dump_tokens, dump_defines=args.dump_defines, dump_preprocessed=args.dump_preprocessed, dump_llvmir=args.dump_llvmir, triple=args.triple, target=target)
    compiler.compile()