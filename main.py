# Holy HolyC Compiler
# https://github.com/wizard-lgtm/hhcc
# Licensed under MIT

from typing import List
from lexer import *
from astparser import *
                
def main():
    code = r"""
U8 a;
U64 b = 10;
U8 c = a + b + 5;
return c;
"""
    tokens = Lexer(code).tokenize()
    for token in tokens:
        token.print()
    parser = Parser(tokens)
    parser.parse()
main()