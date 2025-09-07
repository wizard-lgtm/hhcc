#include "./lib/print.HC"
class Person{
    U8 age;
    U8* name;
};
 
U32 main(){

    Person p;

    U8* name = "TEST";

    p.age = 10;

    p.name = name;

    print_str(p.name);

    return 0;
}