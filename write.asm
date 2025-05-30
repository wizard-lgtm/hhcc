section .text
global stdout_write

; void stdout_write(const char* str, unsigned long len)
stdout_write:
    ; rdi = str
    ; rsi = len

    mov     rax, 1      ; syscall: write
    mov     rdx, rsi    ; rdx = len
    mov     rsi, rdi    ; rsi = str
    mov     rdi, 1      ; rdi = stdout (fd 1)

    syscall
    ret