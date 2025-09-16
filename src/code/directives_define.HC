#include "lib/print.HC"
// defines_test_io.HC
// Test suite for HolyC preprocessor defines using custom print functions

// --- Basic define usage ---
#define MAGIC_NUMBER 420
#define NAME "HolyC Test"
#define MAX_CHARS (1 << 10)

print_str("--- Basic defines ---\n");
print_int(MAGIC_NUMBER); print_chr('\n'); // should print 420
print_str(NAME); print_chr('\n');         // should print HolyC Test
print_int(MAX_CHARS); print_chr('\n');    // should print 1024

// --- #undef test ---
#undef MAGIC_NUMBER

#ifdef MAGIC_NUMBER
print_str("ERROR: MAGIC_NUMBER still defined!\n");
#else
print_str("SUCCESS: MAGIC_NUMBER was undefined.\n");
#endif

// --- #ifdef / #else test ---
#define SOME_FLAG

#ifdef SOME_FLAG
print_str("SUCCESS: SOME_FLAG is defined.\n");
#else
print_str("ERROR: SOME_FLAG was not detected.\n");
#endif

// --- #ifndef test ---
#ifndef OTHER_FLAG
print_str("SUCCESS: OTHER_FLAG is not defined.\n");
#endif

// --- #elifdef chain test ---
#ifdef FOO
print_str("FOO branch taken\n");
#elifdef YEET
print_str("YEET branch taken\n");
#else
print_str("SUCCESS: Neither FOO nor YEET defined, took else branch.\n");
#endif

// --- #error test (commented out for normal runs) ---
// Uncomment to verify compiler error reporting
// #error "Manual failure test for #error directive"

U32 main(){
    return 0;
}