// directives.HC - Test file for preprocessor directives
#define MAX_COUNT 5
#define PI 3.14159
#define DEBUG_MODE

// Test constants
U0 ShowConstants() {
  Print("MAX_COUNT = %d\n", MAX_COUNT);
  Print("PI = %f\n", PI);
  
  #ifdef DEBUG_MODE
    Print("Debug mode is enabled\n");
  #else
    Print("Debug mode is disabled\n");
  #endif
}

// Test conditional compilation
U0 ConditionalTest() {
  #ifdef FEATURE_A
    Print("Feature A is enabled\n");
  #endif
  
  #ifndef FEATURE_B
    Print("Feature B is not enabled\n");
  #endif
  
  #ifdef FEATURE_C
    Print("Feature C is enabled\n");
  #elifdef FEATURE_D
    Print("Feature D is enabled\n");
  #else
    Print("Neither Feature C nor D is enabled\n");
  #endif
}