section .text
global stdout_write

; void stdout_write(const char* str, unsigned long len)
; rdi = str
; rsi = len
stdout_write:
    mov     rax, 1      ; syscall number: write
    mov     rdx, rsi    ; count = len
    mov     rsi, rdi    ; buf = str
    mov     rdi, 1      ; fd = stdout
    syscall
    ret
