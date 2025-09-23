#ifndef LIB_PRINT_HC
#define LIB_PRINT_HC
U64 stdout_write(U8* str, U64 len){
    U64 ret;
    U64 rax=1;
    U64 rdi=1;
    U64 rsi = (U64)str;
    U64 rdx = len;

    asm volatile (
        "syscall"
        : "=a"(ret)
        : "a"(rax), "D"(rdi), "S"(rsi), "d"(rdx)
        : "rcx", "r11", "memory"
    );
    
    return ret;
}


U0 print_str(U8* str){
    U64 len = 0;
    while(str[len] != '\0') {
        len++;
    }
    stdout_write(str, len);
}


U0 print_chr(U8 ch) {
    U8 buf[1];
    buf[0] = (U8)ch;
    stdout_write(buf, 1);
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

// Print unsigned integer
U0 print_u64(U64 val){
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

// Print signed integer
U0 print_i64(I64 val){
    if(-1 < 0){
        print_chr('-');
        // careful with I64_MIN (-9223372036854775808), negate via unsigned
        print_u64((U64)(-val));
    } else {
        print_str("NOT NEGATIVE\n");
        print_u64((U64)val);
    }
}

// Print F64 (basic, not scientific)
U0 print_f64(F64 val, I32 precision){
    if(val < 0){
        print_chr('-');
        val = -val;
    }

    I64 int_part = (I64)val;
    F64 frac_part = val - (F64)int_part;

    print_i64(int_part);
    print_chr('.');

    for(I32 i=0; i<precision; i++){
        frac_part *= 10.0;
        I32 digit = (I32)frac_part;
        print_chr('0' + digit);
        frac_part -= digit;
    }
}


#endif