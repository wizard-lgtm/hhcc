# Holy HolyC Compiler
# https://github.com/wizard-lgtm/hhcc
# Licensed under MIT

from typing import List
from lexer import *
from astparser2 import *
                
def main():
    code = r"""
U8 a;
U64 b = 10;
b = 4;
U8 c = a + b + 5;
U8 d = (9 + 10) * 5;
U8 myfunc(U8 a=5, U8 b){
    U8 c = a + 5;

    return c;
}

if(a == 5){

}
if (b == 5)
{
    U8 a;
}
if(b==4){}
if(b!=4){}
else{}
"""
    tokens = Lexer(code).tokenize()
    for token in tokens:
        token.print()
    parser = ASTParser(tokens, code)
    
    nodes = parser.parse()
    for node in nodes:
        print(node)

main()