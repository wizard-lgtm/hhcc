# Holy HolyC Compiler
# https://github.com/wizard-lgtm/hhcc
# Licensed under MIT

from typing import List
from lexer import *
from astparser2 import *
from preprocessor import *

class Compiler:
    file = str,
    src = str,
    tokens = list[Token],
    astnodes = list[ASTNode]
    debug = bool
    def __init__(self, file, debug=False):
        self.file =  file
        self.debug = debug
        with open(file, "r") as src:
            self.src = src.read()
    def compile(self):
        # 1. Preprocessor
        # TODO!

        # 2. Lexical Analysis
        tokens = Lexer(self.src).tokenize()
        if self.debug:
            for token in tokens:
                token.print()

        # 3. AST Parsing
        parser = ASTParser(tokens, self.src)
        nodes = parser.parse()
        if self.debug:
            for node in nodes:
                print(node)

        # 4. LLVM IR Generation
        # TODO!
                

    
if __name__ == "__main__":
    file = "./src/code/main.HC"
    Compiler(file, debug = True).compile()