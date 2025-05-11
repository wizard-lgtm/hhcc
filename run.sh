python src/compiler.py src/code/test4.HC -S -emit-llvm -o output
./output
echo $?