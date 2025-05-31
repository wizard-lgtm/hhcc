U64 write(U8* str, U64 len){
    
    U64 syscall_number = 1; 
    U64 rax, rdi, rsi, rdx;
    rax = syscall_number; // syscall number for write
    rdi = 1; // file descriptor 1 is stdout
    rsi = (U64)str; // pointer to the string
    rdx = len; // length of the string
    asm volatile(
        "syscall"
        : "=a"(rax) // output: rax will contain the return value
        : "D"(rdi), "S"(rsi), "d"(rdx) // inputs: rdi, rsi, rdx
        : "rcx", "r11", "memory" // clobbered registers
    );
    return rax; // return the number of bytes written
}