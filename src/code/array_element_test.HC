extern U0 stdout_write(U8* str, U64 len);
U0 main(){
    U8* str = "hello world\n";
    str[3] = 'p';
    U64 len = 16;
    stdout_write(str, len);
}
