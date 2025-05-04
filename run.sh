python3 src/compiler.py src/code/test2.HC -S -emit-llvm -o output.ll
llvm-as output.ll 
clang output.bc -o output
./output
echo $?