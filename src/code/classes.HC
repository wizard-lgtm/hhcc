class Person{
    U8 age;
    U8* name;

    U0 change_age(Person self, U8 age){
        self.age = age;
    }
};

 
U32 main(){

    Person p;

    U8* name = "TEST";

    p.age = 10;

    p.name = name;

    p.change_age(16);

    return 0;
}