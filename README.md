# Holy Holy C Compiler.

<center>
  
![ago_downloaded](https://github.com/user-attachments/assets/46a3be88-28fe-47b9-a039-fe7346f4dedd)
</center>

![holy-holy-c-logo](https://github.com/user-attachments/assets/5010cd99-f253-40c3-b643-3f3e86480729)

Holy Holy C is the superset of Holy-C.

In the memory of Terry A. Davis: https://www.youtube.com/watch?v=NYClgSGzWnI

remy clarke said python is kinda suitable for compiler development. it works tho, since the rust came up we don't give a sh\*t about compile times as well. still i'm thinking about writing the compiler in rust when it's done

## Features

- LLVM-based: Compile to any target.
- Full Holy-C syntax support
- Class Methods & New Features
- Cross platform libc (Future planned)

## Current Status

# What's New in v0.1.0

## Features

### Enums
- Added support for enum access using the scope resolution operator (`::`).
- Automatic value assignment for enums.
- Support for string enums.

### Code Generation
- Refactored `codegen.py` into multiple files for better maintainability.
- Enhanced linker for bare-metal targets and added target options.
- Added array declaration functions and array casting to pointers in function calls.
- Improved array access handling, including pointer types and indexing.
- Enhanced expression handling with array access, unary, and postfix operations.

### Functionality Improvements
- Added support for user-typed and pointer parameters in functions.
- Enhanced function call handling with automatic array decay.
- Added direct `for` loop statement support.

### Language Enhancements
- Added support for C-style casting and complex type casting.
- Added handling for character literals in the lexer.


# Prerequirements

sudo apt install python3-llvmlite

# Testing
Create a virtual enviroment and get into it 

```bash
pip3 install llvmlite
python3 src/compiler.py src/code/inlineasm_test.HC -o test
./test
```

## Contribution

You can check issues page for any help, Thx!

our discord server: https://discord.gg/GY5C7XfgXz
