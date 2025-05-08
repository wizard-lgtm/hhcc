#include "test3.HC"
class Node{
    Node* next;
    U8 value;
};

U8 main(){
    Node n1;
    Node n2;

    n1.next = &n2;
    n2.next = NULL;

    n1.value = 1;
    n2.value = 2;
    U8 a = 0;
    a++;

    a = n1.next.value + n1.value + pub_func();
     
    return a;
}