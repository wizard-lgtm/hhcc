class Person{
    U8 age;
    U8* name;

    U0 say_hi(Person* self){
    }
};

 
U32 main(){

    Person p;

    U8* name = "TEST";

    p.age = 10;

    p.name = name;

    p.say_hi();

    return 0;
}