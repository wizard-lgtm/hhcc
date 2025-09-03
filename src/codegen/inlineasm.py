from .symboltable import SymbolTable 
from llvmlite import ir, binding
from typing import TYPE_CHECKING, Dict, Callable, Type, Optional, List
from astnodes import *
from lexer import *

if TYPE_CHECKING:
    from .base import Codegen

def map_constraint_to_type(self, constraint: str) -> ir.Type:
    # Strip output constraint prefix if present
    original_constraint = constraint
    if constraint.startswith("="):
        constraint = constraint[1:]
    
    # For x86-64, all general-purpose registers are 64-bit
    # But we might need to use smaller types depending on the actual usage
    register_constraints = {
        'a': ir.IntType(64),  # RAX (can be used as EAX, AX, AL)
        'b': ir.IntType(64),  # RBX
        'c': ir.IntType(64),  # RCX
        'd': ir.IntType(64),  # RDX
        'S': ir.IntType(64),  # RSI
        'D': ir.IntType(64),  # RDI
        'r': ir.IntType(64),  # Any general purpose register
        'q': ir.IntType(64),  # Any register accessible as 8-bit
        # Extended constraint names that Clang uses
        'ax': ir.IntType(64), # RAX
        'di': ir.IntType(64), # RDI
        'si': ir.IntType(64), # RSI
        'dx': ir.IntType(64), # RDX
    }
    
    if constraint in register_constraints:
        return register_constraints[constraint]
    
    # Other constraints
    if constraint in ["i", "n"]:  # immediate values
        return ir.IntType(32)
    elif constraint == "m":  # memory operand
        return ir.PointerType(ir.IntType(8))
    elif constraint == "f":  # floating point register
        return ir.DoubleType()
    elif constraint == "x":  # SSE register
        return ir.VectorType(ir.FloatType(), 4)
    else:
        # Default to 64-bit integer for unknown register constraints
        return ir.IntType(64)

def load_variable_value(self: "Codegen", variable_name: str, builder: ir.IRBuilder) -> ir.Value:
    """
    Load a variable's value from the symbol table with proper type handling.
    """
    symbol = self.symbol_table.lookup(variable_name)
    if not symbol:
        raise ValueError(f"Variable '{variable_name}' not found in symbol table")
    
    if symbol.llvm_value is None:
        raise ValueError(f"Variable '{variable_name}' has no LLVM value")
    
    llvm_value = symbol.llvm_value
    
    # If the symbol's LLVM value is a pointer (alloca result), load it
    if isinstance(llvm_value.type, ir.PointerType):
        loaded_value = builder.load(llvm_value, name=f"{variable_name}_loaded")
        if self.compiler.debug:
            print(f"Loaded value for {variable_name}: {loaded_value} (type: {loaded_value.type})")
        return loaded_value
    else:
        # Direct value - just return it
        if self.compiler.debug:
            print(f"Direct value for {variable_name}: {llvm_value} (type: {llvm_value.type})")
        return llvm_value

def handle_inline_asm(self: "Codegen", node: ASTNode.InlineAsm, builder: ir.IRBuilder, **kwargs) -> Optional[ir.Value]:
    # Collect input values and types - like Clang does
    input_values = []
    arg_types = []
    
    # Process input constraints and collect actual values
    for i, constraint_info in enumerate(node.input_constraints):
        if isinstance(constraint_info, dict) and 'variable' in constraint_info:
            constraint = constraint_info['constraint']
            variable = constraint_info['variable']
            
            try:
                # Load the actual value from the symbol table (like Clang)
                value = load_variable_value(self, variable, builder)
                
                # Get expected type for this constraint
                expected_type = map_constraint_to_type(self, constraint)
                
                # Ensure compatibility - cast if necessary
                if value.type != expected_type:
                    if isinstance(value.type, ir.IntType) and isinstance(expected_type, ir.IntType):
                        if value.type.width < expected_type.width:
                            value = builder.zext(value, expected_type, name=f"{variable}_ext")
                        elif value.type.width > expected_type.width:
                            value = builder.trunc(value, expected_type, name=f"{variable}_trunc")
                    # For pointer to integer conversion (if needed)
                    elif isinstance(value.type, ir.PointerType) and isinstance(expected_type, ir.IntType):
                        value = builder.ptrtoint(value, expected_type, name=f"{variable}_ptrtoint")
                
                input_values.append(value)
                arg_types.append(value.type)
                
            except Exception as e:
                raise ValueError(f"Error processing input variable '{variable}': {e}")
        else:
            # Handle immediate or other non-variable constraints
            print(f"Warning: Unhandled input constraint format: {constraint_info}")
    
    # Determine return type based on output constraints
    ret_type = ir.VoidType()
    output_variables = []
    
    if node.output_constraints:
        output_info = node.output_constraints[0]
        if isinstance(output_info, dict):
            output_constraint = output_info['constraint']
            if 'variable' in output_info:
                output_variables.append(output_info['variable'])
        else:
            output_constraint = output_info
        
        ret_type = map_constraint_to_type(self, output_constraint)
        if (self.compiler.debug):
            print(f"Output constraint: {output_constraint}, return type: {ret_type}")
    
    # Create function type for the inline assembly
    asm_func_type = ir.FunctionType(ret_type, arg_types)
    if (self.compiler.debug):
        print(f"Function type: {asm_func_type}")
    
    # Build constraint string using Clang's format - this is the key fix!
    constraints = []
    
    # Add output constraints first (like Clang does)
    for constraint_info in node.output_constraints:
        if isinstance(constraint_info, dict):
            constraint = constraint_info['constraint']
        else:
            constraint = constraint_info
            
        # Convert single-letter constraints to Clang's {reg} format for clarity
        if len(constraint) == 2 and constraint.startswith('='):
            reg_letter = constraint[1]
            if reg_letter == 'a':
                constraints.append("={ax}")
            elif reg_letter == 'D':
                constraints.append("={di}")
            elif reg_letter == 'S':
                constraints.append("={si}")
            elif reg_letter == 'd':
                constraints.append("={dx}")
            else:
                constraints.append(constraint)
        else:
            constraints.append(constraint)
    
    # Add input constraints (like Clang does)
    for constraint_info in node.input_constraints:
        if isinstance(constraint_info, dict):
            constraint = constraint_info['constraint']
        else:
            constraint = constraint_info
            
        # Convert single-letter constraints to Clang's {reg} format for clarity
        if len(constraint) == 1:
            if constraint == 'a':
                constraints.append("{ax}")
            elif constraint == 'D':
                constraints.append("{di}")
            elif constraint == 'S':
                constraints.append("{si}")
            elif constraint == 'd':
                constraints.append("{dx}")
            else:
                constraints.append(constraint)
        else:
            constraints.append(constraint)
    
    # Add clobber constraints (like Clang does)
    for clobber in node.clobber_list:
        # Format clobbers properly - they need to be wrapped in ~{}
        if not clobber.startswith('~'):
            constraints.append(f"~{{{clobber}}}")
        else:
            constraints.append(clobber)
    
    constraint_string = ",".join(constraints)
    
    try:
        # Create the inline assembly (like Clang does)
        inline_asm = ir.InlineAsm(
            asm_func_type,
            node.assembly_code,
            constraint_string,
            side_effect=node.is_volatile,
        )
        
        
        # Call the inline assembly with loaded values (like Clang does)
        if input_values:
            result = builder.call(inline_asm, input_values, name="asm_result")
        else:
            result = builder.call(inline_asm, [], name="asm_result")
        
        
        # Handle output variables - store the result back if needed
        if output_variables and result is not None and not isinstance(ret_type, ir.VoidType):
            for var_name in output_variables:
                symbol = self.symbol_table.lookup(var_name)
                if symbol and symbol.llvm_value:
                    if isinstance(symbol.llvm_value.type, ir.PointerType):
                        # Store to the variable (like Clang does)
                        builder.store(result, symbol.llvm_value)
                    else:
                        # Update the symbol directly (less common case)
                        symbol.llvm_value = result
        
        return result if not isinstance(ret_type, ir.VoidType) else None
        
    except Exception as e:
        print(f"Error creating inline assembly: {e}")
        print(f"Function type: {asm_func_type}")
        print(f"Assembly code: '{node.assembly_code}'")
        print(f"Constraint string: '{constraint_string}'")
        print(f"Input values: {input_values}")
        raise RuntimeError(f"Failed to create inline assembly: {e}")

