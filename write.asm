section .text
global stdout_write

; void stdout_write(const char* str, unsigned long len)
; Arguments (Linux x86_64 calling convention):
; rdi = const char* str
; rsi = unsigned long len
stdout_write:
    push    rdi         ; Save rdi in case caller needs it preserved
    push    rsi         ; Save rsi for the same reason

    mov     rax, 1      ; syscall number for write
    mov     rdi, 1      ; file descriptor (stdout)
    ; rsi and rdx already have correct values
    ; rsi = buffer (str)
    ; rdx = length
    syscall

    pop     rsi         ; Restore original rsi
    pop     rdi         ; Restore original rdi
    ret
