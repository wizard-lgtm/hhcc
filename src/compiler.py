# Holy HolyC Compiler
# https://github.com/wizard-lgtm/hhcc
# Licensed under MIT
# In memory of Terry A. Davis (1954-2023) 
# Rest in peace, Terry. You will be in our memories forever.
import argparse
import os
from lexer import *
from astparser2 import *
from preprocessor import Preprocessor
from target import Target
from codegen.base import Codegen
from linker import Linker  # Import the new Linker class

import platform 
class Compiler:
    file = str
    src = str
    tokens = list[Token]
    astnodes = list[ASTNode]
    debug = bool
    dump_ast = bool
    dump_tokens = bool
    dump_llvmir = bool
    preprocessor: Preprocessor
    lexer: Lexer
    astparser: ASTParser
    target: Target
    linker: Linker
    
    def __init__(self, file, debug=False, dump_ast=False, dump_tokens=False, dump_defines=False, 
                 dump_preprocessed=False, dump_llvmir=False, triple=None, target=None, output_file=None,
                 emit_llvm=False, compile_only=False, link_libs=None, lib_paths=None, object_files=None):
        self.version = "0.0.10"  
        self.file = os.path.abspath(file)
        self.file_directory = os.path.dirname(self.file)
        self.working_directory = os.getcwd()
        self.debug = debug
        self.dump_ast = dump_ast
        self.dump_tokens = dump_tokens
        self.dump_defines = dump_defines
        self.dump_preprocessed = dump_preprocessed
        self.dump_llvmir = dump_llvmir
        self.defines = {} # TODO! -> LLVM IR generate step (constant defines)
        self.triple = triple
        self.output_file = output_file
        self.emit_llvm = emit_llvm
        self.compile_only = compile_only
        self.link_libs = link_libs or []
        self.lib_paths = lib_paths or []
        self.object_files = object_files or []
        
        print(f"hhcc compiler version: {self.version}")

        if self.debug:
            print(f"Compiling file: {self.file}")
            print(f"Working directory: {self.working_directory}")
            print(f"File directory: {self.file_directory}")
        
        with open(file, "r") as src:
            self.src = src.read()
        
        self.preprocessor = Preprocessor(self)
        self.astparser = ASTParser(self.src, self)

        # Set the target architecture, OS, and ABI
        self.target = target

        # Initialize the linker
        self.linker = Linker(self)
        
        # Add any specified object files and libraries
        for obj_file in self.object_files:
            self.linker.add_object_file(obj_file)
        
        for lib in self.link_libs:
            self.linker.add_library(lib)
        
        for path in self.lib_paths:
            self.linker.add_library_path(path)

        if self.debug: 
            print(f"Selected Target: {self.target}")
    
    def compile(self):
        # 1. Preprocessor
        if self.debug:
            print("Running Preprocessor")
        processed_code = self.preprocessor.preprocess()
        
        if self.dump_preprocessed:
            print("==== Preprocessed Code ====")
            print(processed_code)
            print("============================")
        
        if self.dump_defines:
            print("======Defines==========")
            for items in self.defines.items():
                print(items)
            print("================")
        # Update the source after preprocessing
        self.src = processed_code
        
        # 2. Lexical Analysis
        self.lexer = Lexer(self)
        if self.debug:
            print("==== Running Lexer ====")
        self.tokens = self.lexer.tokenize()
        if self.dump_tokens:
            print("==== Tokens ====")
            for token in self.tokens:
                token.print()
            print("================")
        
        # 3. AST Parsing
        if self.debug:
            print("==== Running AST Parser ====")
        self.astparser.load_tokens(self.tokens)
        nodes = self.astparser.parse()
        self.astnodes = nodes
        if self.dump_ast:
            print("==== AST Nodes ====")
            for node in nodes:
                print(node)
            print("==================")
        
        # 4. LLVM IR Generation
        if self.debug:
            print("==== Running Codegen ====")
        codegen = Codegen(self)
        llvmir = codegen.gen()
        
        # Default output file name if not specified
        if not self.output_file:
            base_name = os.path.splitext(os.path.basename(self.file))[0]
            if self.emit_llvm:
                self.output_file = f"{base_name}.ll"
            elif self.compile_only:
                self.output_file = f"{base_name}.o"
            else:
                self.output_file = f"{base_name}.out" if platform.system() != "Windows" else f"{base_name}.exe"
        
        # Output LLVM IR to file
        ir_file = os.path.splitext(self.output_file)[0] + ".ll"
        with open(ir_file, 'w') as f:
            f.write(str(llvmir))
        
        if self.dump_llvmir:
            print("==== Generated LLVM IR Code ====")
            print(llvmir)
            print("==================")
        
        # 5. Compile LLVM IR to object file if not emit-llvm
        if not self.emit_llvm:
            obj_file = self.linker.compile_ir_to_object(ir_file)
            
            # 6. Link if not compile_only
            if not self.compile_only:
                self.linker.link()

        print("Done!")
        return nodes

def __init__(self, file, debug=False, dump_ast=False, dump_tokens=False, dump_defines=False, 
             dump_preprocessed=False, dump_llvmir=False, triple=None, target=None, output_file=None,
             emit_llvm=False, compile_only=False, link_libs=None, lib_paths=None, object_files=None,
             link_with_clang=False):
    self.version = "0.0.9"  # Updated version number
    self.file = os.path.abspath(file)
    self.file_directory = os.path.dirname(self.file)
    self.working_directory = os.getcwd()
    self.debug = debug
    self.dump_ast = dump_ast
    self.dump_tokens = dump_tokens
    self.dump_defines = dump_defines
    self.dump_preprocessed = dump_preprocessed
    self.dump_llvmir = dump_llvmir
    self.defines = {}  # TODO! -> LLVM IR generate step (constant defines)
    self.triple = triple
    self.output_file = output_file
    self.emit_llvm = emit_llvm
    self.compile_only = compile_only
    self.link_libs = link_libs or []
    self.lib_paths = lib_paths or []
    self.object_files = object_files or []
    self.link_with_clang = link_with_clang
    
    print(f"hhcc compiler version: {self.version}")

    if self.debug:
        print(f"Compiling file: {self.file}")
        print(f"Working directory: {self.working_directory}")
        print(f"File directory: {self.file_directory}")
    
    with open(file, "r") as src:
        self.src = src.read()
    
    self.preprocessor = Preprocessor(self)
    self.astparser = ASTParser(self.src, self)

    # Set the target architecture, OS, and ABI

    self.target = target

    # Initialize the linker
    self.linker = Linker(self)
    
    # Add any specified object files
    for obj_file in self.object_files:
        self.linker.add_object_file(obj_file)
    
    # Add any specified libraries
    for lib in self.link_libs:
        self.linker.add_library(lib)
    
    # Add any specified library paths
    for path in self.lib_paths:
        self.linker.add_library_path(path)

    if self.debug: 
        print(f"Selected Target: {self.target}")

# Replace the compile method in the Compiler class with this updated version
def compile(self):
    # 1. Preprocessor
    if self.debug:
        print("Running Preprocessor")
    processed_code = self.preprocessor.preprocess()
    
    if self.dump_preprocessed:
        print("==== Preprocessed Code ====")
        print(processed_code)
        print("============================")
    
    if self.dump_defines:
        print("======Defines==========")
        for items in self.defines.items():
            print(items)
        print("================")
    # Update the source after preprocessing
    self.src = processed_code
    
    # 2. Lexical Analysis
    self.lexer = Lexer(self)
    if self.debug:
        print("==== Running Lexer ====")
    self.tokens = self.lexer.tokenize()
    if self.dump_tokens:
        print("==== Tokens ====")
        for token in self.tokens:
            token.print()
        print("================")
    
    # 3. AST Parsing
    if self.debug:
        print("==== Running AST Parser ====")
    self.astparser.load_tokens(self.tokens)
    nodes = self.astparser.parse()
    self.astnodes = nodes
    if self.dump_ast:
        print("==== AST Nodes ====")
        for node in nodes:
            print(node)
        print("==================")
    
    # 4. LLVM IR Generation
    if self.debug:
        print("==== Running Codegen ====")
    codegen = Codegen(self)
    llvmir = codegen.gen()
    
    # Default output file name if not specified
    if not self.output_file:
        base_name = os.path.splitext(os.path.basename(self.file))[0]
        if self.emit_llvm:
            self.output_file = f"{base_name}.ll"
        elif self.compile_only:
            self.output_file = f"{base_name}.o"
        else:
            self.output_file = f"{base_name}.out" if platform.system() != "Windows" else f"{base_name}.exe"
    
    # Output LLVM IR to file
    ir_file = os.path.splitext(self.output_file)[0] + ".ll"
    with open(ir_file, 'w') as f:
        f.write(str(llvmir))
    
    if self.dump_llvmir:
        print("==== Generated LLVM IR Code ====")
        print(llvmir)
        print("==================")
    
    # 5. Decide what to do based on command line options
    if self.emit_llvm:
        # We've already written the LLVM IR to a file, so we're done
        print(f"LLVM IR written to {ir_file}")
    elif self.link_with_clang and not self.compile_only:
        # Link directly with clang
        self.linker.link_with_clang(ir_file)
    else:
        # Compile LLVM IR to object file
        obj_file = self.linker.compile_ir_to_object(ir_file)
        
        # Link if not compile-only
        if not self.compile_only:
            self.linker.link()

    print("Done!")
    return nodes

def parse_args():
    parser = argparse.ArgumentParser(description="Holy HolyC Compiler")
    parser.add_argument("file", nargs='?', help="Source file to compile")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debugging output")
    parser.add_argument("-da", "--dump-ast", action="store_true", help="Dump the AST tree")
    parser.add_argument("-dt", "--dump-tokens", action="store_true", help="Dump the token list")
    parser.add_argument("-df", "--dump-defines", action="store_true", help="Dump defines")
    parser.add_argument("-dp", "--dump-preprocessed", action="store_true", help="Dump Preprocessed Code")
    parser.add_argument("-dl", "--dump-llvmir", action="store_true", help="Dump llvm_ir")
    parser.add_argument("--target", help="Target in the format <arch>-<os>-<abi>. Default is native target.", default=None)
    parser.add_argument("--triple", help="Target in the format <arch>-<os>-<abi>. Default is native target.", default=None)
    parser.add_argument("-S", action="store_true", help="Compile only; do not assemble or link")
    parser.add_argument("-c", "--compile-only", action="store_true", help="Compile to object file but do not link")
    parser.add_argument("-emit-llvm", action="store_true", help="Generate LLVM IR code")
    parser.add_argument("-o", "--output", help="Output file name", default=None)
    
    # New options for linking
    parser.add_argument("-l", "--link-lib", action="append", dest="link_libs", help="Link against library", default=[])
    parser.add_argument("-L", "--lib-path", action="append", dest="lib_paths", help="Library search path", default=[])
    parser.add_argument("--obj", action="append", dest="object_files", help="Object file to link", default=[])
    parser.add_argument("--use-clang", action="store_true", help="Use clang for linking (if available)")
    
    # List targets option
    parser.add_argument("--list-targets", action="store_true", help="List all available target architectures, OS, vendors, and ABIs")

    return parser.parse_args()

# Update the main function to handle list-targets
if __name__ == "__main__":
    args = parse_args()

    # Handle list-targets command
    if args.list_targets:
        Target.list_targets()
        exit(0)

    # Check if file argument is provided when not listing targets
    if not args.file:
        print("Error: Source file is required unless using --list-targets")
        exit(1)

    # Parse the target argument if provided
    target = None
    if args.target:
        try:
            # Try to create a Target using the string from the command line
            target = Target.from_string(args.target)
        except ValueError as e:
            print(f"Error parsing target: {e}")
            print("Use --list-targets to see available options")
            exit(1)

    # Create the compiler instance with the parsed arguments
    compiler = Compiler(
        args.file, 
        debug=args.debug, 
        dump_ast=args.dump_ast, 
        dump_tokens=args.dump_tokens, 
        dump_defines=args.dump_defines, 
        dump_preprocessed=args.dump_preprocessed, 
        dump_llvmir=args.dump_llvmir, 
        triple=args.triple, 
        target=target, 
        output_file=args.output,
        emit_llvm=args.emit_llvm,
        compile_only=args.compile_only or args.S,
        link_libs=args.link_libs,
        lib_paths=args.lib_paths,
        object_files=args.object_files,
    )
    compiler.compile()