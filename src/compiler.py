# Holy HolyC Compiler
# https://github.com/wizard-lgtm/hhcc
# Licensed under MIT

import argparse

import argparse
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


    def __init__(self, file, debug=False, dump_ast=False, dump_tokens=False):
        self.file = file
        self.debug = debug
        self.dump_ast = dump_ast
        self.dump_tokens = dump_tokens
        with open(file, "r") as src:
            self.src = src.read()
        self.preprocessor = Preprocessor(self)
        self.lexer = Lexer(self.src, self)
        self.astparser = ASTParser(self.src, self)

    def compile(self):
        # 1. Preprocessor
        # TODO!

        # 2. Lexical Analysis
        self.tokens = self.lexer.tokenize()
        if self.debug or self.dump_tokens:
            for token in self.tokens:
                token.print()

        # 3. AST Parsing
        self.astparser.load_tokens(self.tokens)
        nodes = self.astparser.parse()
        if self.debug or self.dump_ast:
            for node in nodes:
                print(node)

        # 4. LLVM IR Generation
        # TODO!

def parse_args():
    parser = argparse.ArgumentParser(description="Holy HolyC Compiler")
    parser.add_argument("file", help="Source file to compile")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debugging output")
    parser.add_argument("--dump-ast", action="store_true", help="Dump the AST tree")
    parser.add_argument("--dump-tokens", action="store_true", help="Dump the token list")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    Compiler(args.file, debug=args.debug, dump_ast=args.dump_ast, dump_tokens=args.dump_tokens).compile()
