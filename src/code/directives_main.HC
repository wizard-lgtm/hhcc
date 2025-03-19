// main.HC - Main program file
#include "math_utils.HC"
#include "directives.HC"

U0 Main() {
  // Test included functions
  PrintMathInfo(4);
  PrintCircleInfo(5.0);
  
  // Test directives
  ShowConstants();
  ConditionalTest();
  


  // Test macro expansion
  I64 x = 10;
  Print("Square of %d is %d\n", x, SQUARE(x));
  Print("Cube of %d is %d\n", x, CUBE(x));
}