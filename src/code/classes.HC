#include "lib/print.HC"
class Person{
    U8 age;
    U8* name;
    const U8* test = "Aqweqweqwe";
    U0 change_age(U8 age){
        self.age = age;
    }

    U0 say_age(Person* self){
        print_str("age of ");
        print_str(self.name);
        print_str(" is ");
        print_int(self.age);
        print_str("\n");
    }
};

 
U32 main(){

    Person p2;
    p2.name = "XD";
    p2.age = 21;
    Person p;

    const U8 a = 10;
    U8* name = "TEST";

    p.age = 10;
    p.name = name;

    p = p2;

    p.change_age(16);
    p.say_age();

    print_str(p.test);
    print_chr('\n');

    return 0;
}