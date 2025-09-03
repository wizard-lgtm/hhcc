
from .symboltable import SymbolTable 
from llvmlite import ir, binding
from typing import TYPE_CHECKING, Dict, Callable, Type
from astnodes import *
from lexer import *

# Import your modules
from . import general
from . import expressions  
from . import controlflow
from . import casting
from . import functions
from . import variables
from . import structures
from . import symboltable
from . import inlineasm


if TYPE_CHECKING:
    from compiler import Compiler  # Only for type hints

# Utility functions for pointer level handling
def count_pointer_level(type_str: str) -> tuple[str, int]:
    """
    Count the pointer level from a type string.
    Returns (base_type, pointer_level)
    
    Examples:
    - "int" -> ("int", 0)
    - "int*" -> ("int", 1)
    - "char**" -> ("char", 2)
    - "MyStruct***" -> ("MyStruct", 3)
    """
    base_type = type_str.rstrip('*')
    pointer_level = len(type_str) - len(base_type)
    return base_type, pointer_level


def apply_pointer_level(base_llvm_type: Any, pointer_level: int) -> Any:
    """
    Apply pointer level to an LLVM base type.
    
    Args:
        base_llvm_type: Base LLVM type (e.g., ir.IntType(32))
        pointer_level: Number of pointer levels (0 = no pointer, 1 = pointer, etc.)
    
    Returns:
        LLVM type with appropriate pointer levels applied
    """
    result_type = base_llvm_type
    for _ in range(pointer_level):
        result_type = ir.PointerType(result_type)
    return result_type


import inspect
import types
class Codegen:
    def _bind_handler_functions(self, modules):
        """Automatically bind all functions from the given modules to this instance."""
        for module in modules:
            # Get all functions from the module
            for name, obj in inspect.getmembers(module, inspect.isfunction):
                # Skip private functions (starting with _) if you want
                # if name.startswith('_'):
                #     continue
                    
                # Bind the function to this instance
                bound_method = types.MethodType(obj, self)
                setattr(self, name, bound_method)
                
    def __init__(self, compiler: "Compiler"):
        # Initialize the new symbol table
        self.symbol_table = SymbolTable()
        
        self.compiler = compiler
        self.astnodes = compiler.astnodes
        if self.compiler.triple:
            self.triple = self.compiler.target.get_llvm_triple()
        else:
            self.triple = ""

        self.node_index = 0
        self.current_node = self.astnodes[self.node_index]
        
        # For storing function and struct information
        self.function_map = {}
        self.struct_table = {}

        self.string_literals = []

        
                # List of modules to auto-bind functions from
        handler_modules = [
            general,
            expressions, 
            controlflow,
            casting,
            functions,
            variables,
            structures,
            symboltable,
            inlineasm
        ]

        self._bind_handler_functions(handler_modules)


        import types
        


        # Node type to handler mapping
        self.node_handlers: Dict[Type, Callable] = {
            ASTNode.ExpressionNode: self.handle_expression,
            ASTNode.Block: self.handle_block,
            ASTNode.VariableDeclaration: self.handle_variable_declaration,
            ASTNode.VariableAssignment: self.handle_variable_assignment,
            ASTNode.Return: self.handle_return,
            ASTNode.FunctionDefinition: self.handle_function_definition,
            ASTNode.IfStatement: self.handle_if_statement,
            ASTNode.WhileLoop: self.handle_while_loop,
            ASTNode.ForLoop: self.handle_for_loop,
            ASTNode.Comment: self.handle_comment,
            ASTNode.FunctionCall: self.handle_function_call,
            ASTNode.Class: self.handle_class,
            ASTNode.Union: self.handle_union,
            ASTNode.Break: self.handle_break,
            ASTNode.Continue: self.handle_continue,
            ASTNode.VariableIncrement: self.handle_variable_increment,
            ASTNode.VariableDecrement: self.handle_variable_decrement,
            ASTNode.Extern: self.handle_extern,
            ASTNode.CompoundVariableDeclaration: self.handle_compound_variable_declaration,
            ASTNode.CompoundVariableAssigment: self.handle_compound_variable_assignment,
            ASTNode.Enum: self.handle_enum,
            ASTNode.ArrayElementAssignment: self.handle_array_element_assignment,
            ASTNode.ArrayDeclaration: self.handle_array_declaration,
            ASTNode.InlineAsm: self.handle_inline_asm
            
        }

        # Define correct LLVM types with appropriate signedness
        # Boolean is represented as i1
        bool_type = ir.IntType(1)
        # Unsigned types
        u8_type = ir.IntType(8)
        u16_type = ir.IntType(16)
        u32_type = ir.IntType(32)
        u64_type = ir.IntType(64)
        # Signed types - in LLVM IR, the types are the same but operations differ
        i8_type = ir.IntType(8)
        i16_type = ir.IntType(16)
        i32_type = ir.IntType(32)
        i64_type = ir.IntType(64)
        # Other types
        void_type = ir.VoidType()
        f32_type = ir.FloatType()
        f64_type = ir.DoubleType()

        self.type_map = {
            Datatypes.BOOL: bool_type,
            Datatypes.U8: u8_type,
            Datatypes.U16: u16_type,
            Datatypes.U32: u32_type,
            Datatypes.U64: u64_type,
            Datatypes.I8: i8_type,
            Datatypes.I16: i16_type,
            Datatypes.I32: i32_type,
            Datatypes.I64: i64_type,
            Datatypes.U0: void_type,
            Datatypes.F32: f32_type,
            Datatypes.F64: f64_type
        }

        self.type_signedness = {
            self.type_map[Datatypes.I8]: True,
            self.type_map[Datatypes.I16]: True,
            self.type_map[Datatypes.I32]: True,
            self.type_map[Datatypes.I64]: True,
            self.type_map[Datatypes.U8]: False,
            self.type_map[Datatypes.U16]: False,
            self.type_map[Datatypes.U32]: False,
            self.type_map[Datatypes.U64]: False,
            self.type_map[Datatypes.BOOL]: False,
        }

        self.signed_int_types = {i8_type, i16_type, i32_type, i64_type}
        self.unsigned_int_types = {bool_type, u8_type, u16_type, u32_type, u64_type}
        self.float_types = {f32_type, f64_type}

    def generation_error(self, message: str, node: 'ASTNode'):
        """Report an error with a formatted message, based on the ASTNode (instead of token)."""
        
        # Retrieve line and column information from the ASTNode (assuming these attributes exist)
        if hasattr(node, 'line') and hasattr(node, 'column'):
            line = node.line
            column = node.column
        else:
            line = column = -1  # If line/column info is missing, set to -1 for safety

        # If source code is available, try to get the line of code where the error occurred
        try:
            error_line = self.compiler.code.splitlines()[line - 1]  # Subtract 1 for 0-based index
        except IndexError:
            error_line = "[ERROR: Line out of range]"

        # Align the caret with the column position (ensure it's within bounds)
        caret_position = " " * (min(column, len(error_line))) + "^"

        # Print error message and source line context
        print(f"Generation Error: {message} at line {line}, column {column}")
        print(f"{error_line}")
        print(f"{caret_position}")
        
        # Print detailed node information
        print(f"Caused by ASTNode: {repr(node)}")

        # Raise an exception with the full error message
        raise Exception(f"{message} at line {line}, column {column}\n"
                        f"{error_line}\n"
                        f"{caret_position}\n"
                        f"Caused by ASTNode: {repr(node)}")

        
    def add_function(self, function: ASTNode.FunctionCall):
        self.function_map[function.name] = function
        
    def lookup_function(self, name: str) -> Optional[ASTNode.FunctionDefinition]:
        """
        Look up a function by name and return the FunctionDefinition.
        Returns None if the function is not found.
        """
        return self.function_map.get(name)

    def current_node(self):
        self.current_node = self.astnodes[self.node_index]
        return self.current_node

    def next_node(self):
        self.node_index += 1
        self.current_node = self.astnodes[self.node_index]
        return self.current_node


    def get_llvm_type(self, type: str):
        if type in self.type_map:
            return self.type_map[type]
        # handle pointer types
        elif type.endswith('*'):
            base_type = self.get_llvm_type(type[:-1]) # delete the * symbol and get it's type
            return ir.PointerType(base_type) 
        else: # other types (classes)
            # TODO! implement
            print("Non-Primitive types did not implemented. Returning a generic type (U8*)")
            return ir.PointerType(ir.IntType(8)) # return generic u8 pointer

    def gen(self):
        # Initialize LLVM
        binding.initialize()
        binding.initialize_native_target()
        binding.initialize_native_asmprinter()
        
        try:
            # Try to use the specified target
            print(self.triple)
            target = binding.Target.from_triple(self.triple)
        except RuntimeError:
            # get native target if specified target doesn't available
            print(f"Warning: Target '{self.triple}' not available, using native target instead")
            target = binding.Target.from_default_triple()
        
        target_machine = target.create_target_machine()
        
        # data layout string
        data_layout = target_machine.target_data
        
        # create the module 
        module = ir.Module(name=self.compiler.file)
        
        # Use the actual triple from the target machine to ensure compatibility
        module.triple = target_machine.triple
        module.data_layout = str(data_layout)
        
        # Build context for code generation
        self.module = module
        self.context = ir.context.Context()
        self.builder = None
        self.function = None
        
        # iterate astnodes for handler 
        for node in self.astnodes:
            self.process_node(node)
            
        return module
    
    def process_node(self, node, **kwargs):
        # Get the node's class type
        node_class = type(node)
        
        # Look up and call the appropriate handler
        if node_class in self.node_handlers:
            return self.node_handlers[node_class](node, **kwargs)
        else:
            print(f"Warning: No handler for node type {node_class.__name__}")
            return None
    
    def turn_variable_type_to_llvm_type(self, type: Datatypes):
        llvm_type  = self.type_map[type]
        return llvm_type
    def get_variable_pointer(self, name):
        """Get the LLVM value pointer for a variable."""
        symbol = self.symbol_table.lookup(name)
        if not symbol:
            raise Exception(f"Undefined variable: {name}")
        
        return symbol.llvm_value





    def handle_comment(self, node: ASTNode.Comment, **kwargs):
        if 'builder' in kwargs and kwargs['builder'] is not None:
            builder = kwargs['builder']
            comment_text = node.text
            if node.is_inline:
                comment_text = "INLINE: " + comment_text
                
            # Add a custom metadata node that we can convert to a comment when printing
            comment_md = builder.module.add_metadata([ir.MetaDataString(builder.module, comment_text)])
    def _create_string_constant(self, builder, string_value):
        """
        Create a string constant and return a pointer to it
        
        Args:
            builder: LLVM IR builder
            string_value: The string value (with or without quotes)
            
        Returns:
            LLVM constant representing i8* pointer to the string
        """
        # Remove quotes if present
        if string_value.startswith('"') and string_value.endswith('"'):
            clean_string = string_value[1:-1]
        elif string_value.startswith("'") and string_value.endswith("'"):
            clean_string = string_value[1:-1]
        else:
            clean_string = string_value
        
        # Handle escape sequences
        clean_string = clean_string.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"').replace("\\'", "'")
        
        # Create the string constant
        string_const = ir.Constant(ir.ArrayType(ir.IntType(8), len(clean_string) + 1), 
                                bytearray(clean_string.encode('utf-8') + b'\0'))
        
        # Create a global variable to hold the string
        global_string = ir.GlobalVariable(self.module, string_const.type, name=f"str_{hash(clean_string) & 0xFFFFFFFF}")
        global_string.initializer = string_const
        global_string.global_constant = True
        global_string.linkage = 'private'
        
        # Return a pointer to the string (i8*)
        return global_string.gep([ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), 0)])