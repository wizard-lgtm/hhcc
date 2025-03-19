# Holy HolyC Compiler
# https://github.com/wizard-lgtm/hhcc
# Licensed under MIT
import argparse
import os
from lexer import *
from astparser2 import *
from preprocessor import Preprocessor

class Compiler:
    file = str
    src = str
    tokens = list[Token]
    astnodes = list[ASTNode]
    debug = bool
    dump_ast = bool
    dump_tokens = bool
    preprocessor: Preprocessor
    lexer: Lexer
    astparser: ASTParser
    
    def __init__(self, file, debug=False, dump_ast=False, dump_tokens=False, dump_defines=False):
        self.file = os.path.abspath(file)
        self.file_directory = os.path.dirname(self.file)
        self.working_directory = os.getcwd()
        self.debug = debug
        self.dump_ast = dump_ast
        self.dump_tokens = dump_tokens
        self.dump_defines = dump_defines
        self.defines = {} # TODO! -> LLVM IR generate step (constant defines)
        
        if self.debug:
            print(f"Compiling file: {self.file}")
            print(f"Working directory: {self.working_directory}")
            print(f"File directory: {self.file_directory}")
        
        with open(file, "r") as src:
            self.src = src.read()
        
        self.preprocessor = Preprocessor(self)
        self.astparser = ASTParser(self.src, self)
    
    def compile(self):
        # 1. Preprocessor
        if self.debug:
            print("Running Preprocessor")
        processed_code = self.preprocessor.preprocess()
        
        if self.debug:
            print("==== Preprocessed Code ====")
            print(processed_code)
            print("============================")
        
        if self.debug or self.dump_defines:
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
        if self.debug or self.dump_tokens:
            print("==== Tokens ====")
            for token in self.tokens:
                token.print()
            print("================")
        
        # 3. AST Parsing
        if self.debug:
            print("==== Running AST Parser ====")
        self.astparser.load_tokens(self.tokens)
        nodes = self.astparser.parse()
        if self.debug or self.dump_ast:
            print("==== AST Nodes ====")
            for node in nodes:
                print(node)
            print("==================")
        
        # 4. LLVM IR Generation
        # TODO!
        
        return nodes

def parse_args():
    parser = argparse.ArgumentParser(description="Holy HolyC Compiler")
    parser.add_argument("file", help="Source file to compile")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debugging output")
    parser.add_argument("--dump-ast", action="store_true", help="Dump the AST tree")
    parser.add_argument("--dump-tokens", action="store_true", help="Dump the token list")
    parser.add_argument("--dump-defines", action="store_true", help="Dump defines")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    compiler = Compiler(args.file, debug=args.debug, dump_ast=args.dump_ast, dump_tokens=args.dump_tokens, dump_defines=args.dump_defines)
    compiler.compile()