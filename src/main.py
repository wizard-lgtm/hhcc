# Holy HolyC Compiler
# https://github.com/wizard-lgtm/hhcc
# Licensed under MIT

from typing import List
from lexer import *
from astparser2 import *
                
def main():
    code = r"""
U8 a = 1 == 1;
// Variable declarations with various types and initializations
U8 simple_var;
U16 init_var = 42;
U32 expr_var = 10 + 32;
U64 complex_expr = (5 * 10) + (20 / 4);

// Variable assignments with different operators
simple_var = 5;
init_var += 10;
complex_expr *= 2;
simple_var /= 2;
init_var %= 3;

// Function declaration with parameters and default values
U8 test_function(U8 param1 = 5, U16 param2, U32 param3 = 10) {
    // Local variable declarations
    U8 local_var = param1 + 10;
    U16 another_var;
    
    // Nested block
    {
        U32 nested_var = 42;
        another_var = nested_var / 2;
    }
    
    // If statement with comparison operators
    if (param1 > param2) {
        local_var = 100;
    } else if (param1 == param2) {
        local_var = 200;
    } else {
        local_var = 300;
    }
    
    // While loop
    while (local_var > 0) {
        local_var -= 1;
    }
    
    // For loop with initialization, condition, and update
    for (U8 i = 0; i < 10; i += 1) {
        another_var += i;
    }
    
    // For loop with decrement
    for (U8 j = 10; j > 0; j -= 1) {
        another_var *= 2;
    }

    // Return statement
    return local_var + another_var;
}

// Test complex expressions
U64 complex_calculation = 10 + 20 * 30 / 5 - 15;
U8 logical_test = (5 > 3) && (10 <= 15) || !(1 == 0);

// Nested if statements
if (simple_var > 0) {
    if (init_var > 10) {
        complex_expr = 1000;
    } else {
        complex_expr = 500;
    }
} else {
    complex_expr = 0;
}

// Empty function
U8 empty_func() {
    return;
}

// Function with all expression types
U8 expression_test() {
    // Unary operators
    U8 neg = -5;
    U8 pos = +10;
    U8 not_val = !0;
    
    // Binary operators of different precedence
    U8 arithmetic = 5 + 10 * 2 / 4 - 3 % 2;
    U8 comparison = (5 < 10) && (15 >= 10) || (5 != 3) && !(2 == 3);
    
    // Parenthesized expressions
    U8 parenthesized = ((5 + 3) * 2) / (1 + 1);
    
    return arithmetic + comparison + parenthesized;

    U8 a = "test";
}

// Nested loops
for (U8 outer = 0; outer < 5; outer += 1) {
    for (U8 inner = 0; inner < outer; inner += 1) {
        simple_var += outer * inner;
    }
    
    while (simple_var > 100) {
        simple_var -= 10;
    }
}

if(!x){}

// Function calls
myfunc(a, b, c);
myfunc;

class Person : Dog
{
  U8 name;
  I64 age;
};

union Example
{
  I32 age;
  U8 ch;
};

Person e;

a = e.say[3];

Bool boolean = true;

break;
continue;


// Arrays

// 1. Basic Initialization

U64 arr1[5] = {1, 2, 3, 4, 5};  // Explicit size, full initialization
U64 arr2[] = {1, 2, 3, 4, 5};   // Implicit size
U64 arr3[5] = {1, 2};           // Partial initialization, rest will be 0
U64 arr4[5] = {0};              // All elements initialized to 0

// 2. Multidimensional Arrays
U64 matrix1[2][2] = { {1, 2}, {3, 4} };  // Standard 2D array
U64 matrix2[2][2] = { 1, 2, 3, 4 };      // Flattened initialization
U64 matrix3[][2] = { {1, 2}, {3, 4} };   // Implicit first dimension

U8* a;


"""
    tokens = Lexer(code).tokenize()
    for token in tokens:
        token.print()
    parser = ASTParser(tokens, code)
    
    nodes = parser.parse()
    for node in nodes:
        print(node)

main()