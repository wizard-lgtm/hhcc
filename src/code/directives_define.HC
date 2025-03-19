// 1. Object-like Macros (Constants)
#define PI 3.14159
#define BUFFER_SIZE 256
#define ENABLE_DEBUG 1

// 2. Function-like Macros (Inline Calculations)
#define SQUARE(x) ((x)*(x))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

// 3. Multi-line Macros
#define PRINT_DEBUG(msg) \
    continue;

// 4. Stringizing Macro (# operator)
#define STRINGIFY(x) #x

// 5. Token Pasting (## operator)
#define CONCAT(a, b) a##b

// 6. Variadic Macros (__VA_ARGS__)
#define LOG(format, ...) printf(format, __VA_ARGS__)

// 7. Undefining a Macro
#define TEMP 42
#undef TEMP  // TEMP is now undefined

// 8. Conditional Compilation
#define DEBUG
#ifdef DEBUG
    #define LOG_MESSAGE(msg) printf("LOG: %s\n", msg)
#else
    #define LOG_MESSAGE(msg)
#endif

// 9. Header Guards (For Header Files)
#ifndef MY_HEADER_H
#define MY_HEADER_H
// Header content here
#endif

// 10. Assembly Macros (HolyC Specific)
#define HALT() asm { HLT; }  // Halts the CPU

// 11. Defining Structs or Inline Code with Macros
#define DEFINE_STRUCT() struct { int x, y; }