// Test file for binary operations

U8 main() {
    // Arithmetic operations
    U8 a = 10 + 5;         // Addition: 15
    U8 b = 10 - 3;         // Subtraction: 7
    U8 c = 5 * 4;          // Multiplication: 20
    U8 d = 20 / 4;         // Division: 5
    U8 e = 10 % 3;         // Modulo: 1
    
    // Bitwise operations
    U8 f = 12 & 5;         // Bitwise AND: 4
    U8 g = 12 | 5;         // Bitwise OR: 13
    U8 h = 12 ^ 5;         // Bitwise XOR: 9
    U8 i = 5 << 2;         // Left shift: 20
    U8 j = 20 >> 2;        // Right shift: 5
    
    // Comparison operations (result should be 1 for true, 0 for false)
    Bool k = 5 == 5;        // Equal: 1
    Bool l = 5 != 3;        // Not equal: 1
    Bool m = 5 < 10;        // Less than: 1
    Bool n = 5 <= 5;        // Less than or equal: 1
    Bool o = 10 > 5;        // Greater than: 1
    Bool p = 10 >= 10;      // Greater than or equal: 1
    
    // Logical operations
    Bool q = (5 > 3) && (10 > 5);  // Logical AND: 1
    Bool r = (5 > 10) || (10 > 5); // Logical OR: 1
    
    // Testing precedence
    U8 s = 5 + 3 * 4;     // Should be 17 (multiplication has higher precedence)
    U8 t = (5 + 3) * 4;   // Should be 32 (parentheses override precedence)
    
    // Complex expression
    U8 result = a + b * c / d - e & f;
    U8 result = a + b * c / d - e;
    if (1==1){
        return 2;
    }
    return result;
}