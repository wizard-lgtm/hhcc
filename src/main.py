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
b = 4;
U8 c = a + b + 5;
U8 myfunc(I64 a, U64 b){
    U8 c = a + b;
    return c;
}

return c;
"""
    tokens = Lexer(code).tokenize()
    for token in tokens:
        token.print()
    parser = Parser(tokens, code)
    
    nodes = parser.parse()
    for node in nodes:
        print(node)
main()