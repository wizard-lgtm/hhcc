
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
