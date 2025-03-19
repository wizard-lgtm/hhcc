// math_utils.HC - Basic math utilities
#define SQUARE(x) ((x)*(x))
#define CUBE(x) ((x)*(x)*(x))

U0 PrintMathInfo(I64 num) {
  Print("%d squared is %d\n", num, SQUARE(num));
  Print("%d cubed is %d\n", num, CUBE(num));
}

U0 PrintCircleInfo(F64 radius) {
  #ifndef PI
    #define PI 3.14159
  #endif
  
  Print("Circle with radius %f:\n", radius);
  Print("  Circumference: %f\n", 2 * PI * radius);
  Print("  Area: %f\n", PI * SQUARE(radius));
}