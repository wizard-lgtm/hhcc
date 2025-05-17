extern U32 externed_func(U8 a);
U8 my_func(U8 a, ...){
    return 10;
}
U8 main(){
    U8 b = externed_func(1);
    return b;
}
