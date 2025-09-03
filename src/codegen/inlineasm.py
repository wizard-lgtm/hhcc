from .symboltable import SymbolTable 
from llvmlite import ir, binding
from typing import TYPE_CHECKING, Dict, Callable, Type, Optional, List
from astnodes import *
from lexer import *

if TYPE_CHECKING:
    from .base import Codegen

def map_constraint_to_type(self, constraint: str) -> ir.Type:
    """
    Map inline assembly constraints to LLVM IR types.
    Focus on proper register allocation and type compatibility.
    """
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
        print(f"Loaded value for {variable_name}: {loaded_value} (type: {loaded_value.type})")
        return loaded_value
    else:
        # Direct value - just return it
        print(f"Direct value for {variable_name}: {llvm_value} (type: {llvm_value.type})")
        return llvm_value

def handle_inline_asm(self: "Codegen", node: ASTNode.InlineAsm, builder: ir.IRBuilder, **kwargs) -> Optional[ir.Value]:
    """
    Handle inline assembly code generation with proper constraint and register handling.
    
    Key fixes:
    1. Proper value loading from symbol table
    2. Correct constraint string formatting
    3. Better type matching for register constraints
    4. Proper handling of output constraints
    """
    print(f"Handling inline assembly: '{node.assembly_code}'")
    print(f"Input constraints: {node.input_constraints}")
    print(f"Output constraints: {node.output_constraints}")
    print(f"Clobber list: {node.clobber_list}")
    
    # Collect input values and types
    input_values = []
    arg_types = []
    
    for i, constraint_info in enumerate(node.input_constraints):
        if isinstance(constraint_info, dict) and 'variable' in constraint_info:
            constraint = constraint_info['constraint']
            variable = constraint_info['variable']
            
            try:
                # Load the actual value from the symbol table
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
                        print(f"Type converted {variable}: {value.type} -> {expected_type}")
                    # For pointer to integer conversion (if needed)
                    elif isinstance(value.type, ir.PointerType) and isinstance(expected_type, ir.IntType):
                        value = builder.ptrtoint(value, expected_type, name=f"{variable}_ptrtoint")
                        print(f"Pointer to int conversion {variable}: {value.type}")
                
                input_values.append(value)
                arg_types.append(value.type)
                print(f"Input {i}: {variable} = {value} (constraint: {constraint})")
                
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
        print(f"Output constraint: {output_constraint}, return type: {ret_type}")
    
    # Create function type for the inline assembly
    asm_func_type = ir.FunctionType(ret_type, arg_types)
    print(f"Function type: {asm_func_type}")
    
    # Build constraint string - this is critical for proper register allocation
    constraints = []
    
    # Add output constraints first
    for constraint_info in node.output_constraints:
        if isinstance(constraint_info, dict):
            constraint = constraint_info['constraint']
        else:
            constraint = constraint_info
        constraints.append(constraint)
    
    # Add input constraints
    for constraint_info in node.input_constraints:
        if isinstance(constraint_info, dict):
            constraint = constraint_info['constraint']
        else:
            constraint = constraint_info
        constraints.append(constraint)
    
    # Add clobber constraints
    for clobber in node.clobber_list:
        # Format clobbers properly - they need to be wrapped in ~{}
        if not clobber.startswith('~'):
            constraints.append(f"~{{{clobber}}}")
        else:
            constraints.append(clobber)
    
    constraint_string = ",".join(constraints)
    print(f"Final constraint string: '{constraint_string}'")
    
    try:
        # Create the inline assembly
        inline_asm = ir.InlineAsm(
            asm_func_type,
            node.assembly_code,
            constraint_string,
            side_effect=node.is_volatile,
        )
        
        print(f"Created inline asm: {inline_asm}")
        
        # Call the inline assembly
        if input_values:
            result = builder.call(inline_asm, input_values, name="asm_result")
        else:
            result = builder.call(inline_asm, [], name="asm_result")
        
        print(f"Assembly call result: {result}")
        
        # Handle output variables - store the result back if needed
        if output_variables and result is not None and not isinstance(ret_type, ir.VoidType):
            for var_name in output_variables:
                symbol = self.symbol_table.lookup(var_name)
                if symbol and symbol.llvm_value:
                    if isinstance(symbol.llvm_value.type, ir.PointerType):
                        # Store to the variable
                        builder.store(result, symbol.llvm_value)
                        print(f"Stored result to {var_name}")
                    else:
                        # Update the symbol directly (less common case)
                        symbol.llvm_value = result
                        print(f"Updated symbol {var_name} with result")
        
        return result if not isinstance(ret_type, ir.VoidType) else None
        
    except Exception as e:
        print(f"Error creating inline assembly: {e}")
        print(f"Function type: {asm_func_type}")
        print(f"Assembly code: '{node.assembly_code}'")
        print(f"Constraint string: '{constraint_string}'")
        print(f"Input values: {input_values}")
        raise RuntimeError(f"Failed to create inline assembly: {e}")

def debug_print_ir_context(builder: ir.IRBuilder, function: ir.Function) -> None:
    """
    Helper function to print debug information about the current IR context.
    """
    print("=== IR Context Debug ===")
    print(f"Current function: {function.name}")
    print(f"Current block: {builder.block.name}")
    print("Function arguments:")
    for i, arg in enumerate(function.args):
        print(f"  {i}: {arg} (type: {arg.type})")
    print("======================")

# Additional helper for constraint validation
def validate_inline_asm_constraints(constraints: List[str], input_count: int, output_count: int) -> bool:
    """
    Validate that inline assembly constraints are properly formatted.
    """
    if len(constraints) != (input_count + output_count):
        print(f"Warning: Constraint count mismatch. Expected {input_count + output_count}, got {len(constraints)}")
        return False
    
    for i, constraint in enumerate(constraints):
        if i < output_count and not constraint.startswith('='):
            print(f"Warning: Output constraint {i} should start with '=': {constraint}")
            return False
        elif i >= output_count and constraint.startswith('='):
            print(f"Warning: Input constraint {i} should not start with '=': {constraint}")
            return False
    
    return True