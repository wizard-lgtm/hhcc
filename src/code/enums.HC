
extern U0 stdout_write(U8* str, U64 len);
enum Colors{
    RED = "red",
    GREEN = "green",
    BLUE = "blue",
    YELLOW = "FOUR",
    ORANGE = "five",
};
U32 main(){
    // Example usage of the Colors enum
    Colors myColor = Colors::BLUE;
    stdout_write(myColor, 5);
    return 0; 
}