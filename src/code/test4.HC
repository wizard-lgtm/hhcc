U8 simple_func(U8 a, U8 b){
    return a + b;
}
U8 main(){
    U8 a = 20;
    a+= simple_func(1, 2);

    return a;
}