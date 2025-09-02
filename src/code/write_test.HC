extern U0 stdout_write(U8* str, U64 len);

U0 print_str(U8* str){
    U64 len = 0;
    while(str[len] != '\0') {
        len++;
    }
    stdout_write(str, len);
}

U0 print_int(U64 val){
    U8 buf[32];
    I64 i = 0;
    if(val == 0){
        buf[i++] = '0';
    } else {
        while(val > 0){
            buf[i++] = '0' + (val % 10);
            val /= 10;
        }
    }
    // reverse string
    I64 j = 0;
    I64 k = i-1;
    while(j < k){
        U8 tmp = buf[j]; buf[j] = buf[k]; buf[k] = tmp;
        j++; k--;
    }
    stdout_write(buf, i);
}


U0 main(){
    U64 passed = 0;
    U64 failed = 0;

    // Test 1: AND
    if((5 & 3) == 1) passed++; else failed++;

    // Test 2: OR
    if((5 | 2) == 7) passed++; else failed++;

    // Test 3: XOR
    if((5 ^ 3) == 6) passed++; else failed++;

    // Test 4: NOT

    // Test 5: Left shift
    if((1 << 3) == 8) passed++; else failed++;

    // Test 6: Right shift
    if((8 >> 2) == 2) passed++; else failed++;

    print_str("TEsts completed.\n");

    // Print results
    print_str("Passed: "); print_int(passed); print_str("\n");
    print_str("Failed: "); print_int(failed); print_str("\n");
}
