.section __TEXT,__text,regular,pure_instructions
.globl _stdout_write
.align 2

; void stdout_write(const char* str, unsigned long len)
; x0 = str
; x1 = len

_stdout_write:
    mov     x2, x1            ; x2 = len
    mov     x1, x0            ; x1 = str
    mov     x0, #1            ; x0 = stdout (fd 1)

    ; syscall number 0x2000004 (macOS: write)
    movz    x16, #0x4         ; Load low 16 bits
    movk    x16, #0x2000, lsl #16 ; Load upper 16 bits

    svc     #0
    ret
