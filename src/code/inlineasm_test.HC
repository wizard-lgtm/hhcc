U64 write(U8* str, U64 len){
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


U64 main() {
    U8 str[] = "hello world\n";
    str[3] = 'p';       // change 'l' to 'p'
    U64 len = 12;       // correct length of "heppo world\n"
    write(&str, len);

    return 0;
}
