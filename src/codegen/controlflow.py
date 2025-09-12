from llvmlite import ir
from typing import TYPE_CHECKING, Dict, Callable, NamedTuple, Type
from astnodes import *
from codegen.symboltable import ScopeType
from lexer import *
class LoopContext(NamedTuple):
    """Context information for a loop to handle break and continue."""
    break_block: ir.Block
    continue_block: ir.Block
    loop_type: str  # 'while' or 'for'
def handle_block(self, node: ASTNode.Block, builder: ir.IRBuilder, **kwargs):
    """
    Handle a block of statements with improved scope management.
    Only create new scopes for blocks that actually need them.
    """
    # Check if this block needs its own scope
    # (e.g., blocks inside if statements, loops, etc. - but NOT function bodies)
    needs_scope = kwargs.get('needs_block_scope', False)
    
    entered_scope = False
    if needs_scope:
        # Enter a BLOCK scope
        self.symbol_table.enter_scope(ScopeType.BLOCK, "block")
        entered_scope = True
    
    try:
        # Process each statement in the block
        for stmt in node.nodes:
            self.process_node(stmt, builder=builder, **kwargs)
    finally:
        if entered_scope:
            self.symbol_table.exit_scope()

def handle_return(self, node: ASTNode.Return, builder: ir.IRBuilder, **kwargs):
    """Handle return statement."""
    # Get return type from function
    function_return_type = builder.function.function_type.return_type
    if node.expression:
        return_value = self.handle_expression(node.expression, builder, function_return_type)
        builder.ret(return_value)
    else:
        builder.ret_void()
        
def handle_if_statement(self, node: ASTNode.IfStatement, builder: ir.IRBuilder, **kwargs):
    """Handle if statement with proper scope management."""
    # Evaluate the condition
    condition = self.handle_expression(node.condition, builder, self.type_map[Datatypes.BOOL])
    if condition.type != ir.IntType(1):
        condition = builder.trunc(condition, ir.IntType(1))

    # Create basic blocks
    then_block = builder.append_basic_block("if.then")
    else_block = builder.append_basic_block("if.else") if node.else_body else None
    merge_block = builder.append_basic_block("if.end")

    # Branch based on the condition
    if else_block:
        builder.cbranch(condition, then_block, else_block)
    else:
        builder.cbranch(condition, then_block, merge_block)

    # Generate code for the 'then' block
    builder.position_at_end(then_block)
    # Pass needs_block_scope=True so the block creates its own scope
    self.process_node(node.if_body, builder=builder, needs_block_scope=True)
    if not builder.block.is_terminated:
        builder.branch(merge_block)

    # Generate code for the 'else' block if it exists
    if else_block:
        builder.position_at_end(else_block)
        self.process_node(node.else_body, builder=builder, needs_block_scope=True)
        if not builder.block.is_terminated:
            builder.branch(merge_block)

    # Position the builder at the merge block
    builder.position_at_end(merge_block)

def handle_while_loop(self, node: ASTNode.WhileLoop, builder: ir.IRBuilder, **kwargs):
    """Handle while loop with proper scope management and break/continue support."""
    # Create basic blocks for the loop
    loop_cond_block = builder.append_basic_block("while.cond")
    loop_body_block = builder.append_basic_block("while.body")
    loop_end_block = builder.append_basic_block("while.end")

    # Create loop context for break/continue
    loop_context = LoopContext(
        break_block=loop_end_block,
        continue_block=loop_cond_block,  # continue goes back to condition
        loop_type="while"
    )
    self.loop_stack.append(loop_context)

    try:
        # Branch to the condition block
        builder.branch(loop_cond_block)

        # Generate code for the condition block
        builder.position_at_end(loop_cond_block)
        condition = self.handle_expression(node.condition, builder, self.type_map[Datatypes.BOOL])
        builder.cbranch(condition, loop_body_block, loop_end_block)

        # Generate code for the body block with its own scope
        builder.position_at_end(loop_body_block)
        
        # Enter a new scope for the loop body
        self.symbol_table.enter_scope(ScopeType.BLOCK, "while_body")
        try:
            for stmt in node.body.nodes:
                if builder.block.is_terminated: 
                    break
                self.process_node(stmt, builder=builder, **kwargs)
        finally:
            self.symbol_table.exit_scope()
        
        # Branch back to condition if not terminated
        if not builder.block.is_terminated:
            builder.branch(loop_cond_block)

    finally:
        # Remove loop context
        self.loop_stack.pop()

    # Position the builder at the end block
    builder.position_at_end(loop_end_block)

def handle_for_loop(self, node: ASTNode.ForLoop, builder: ir.IRBuilder, **kwargs):
    """Handle for loop with proper scope management and break/continue support."""
    # Create basic blocks for the for loop
    loop_init_block = builder.append_basic_block("for.init")
    loop_cond_block = builder.append_basic_block("for.cond")
    loop_body_block = builder.append_basic_block("for.body")
    loop_update_block = builder.append_basic_block("for.update")
    loop_end_block = builder.append_basic_block("for.end")

    # Create loop context for break/continue
    loop_context = LoopContext(
        break_block=loop_end_block,
        continue_block=loop_update_block,  # continue goes to update block
        loop_type="for"
    )
    self.loop_stack.append(loop_context)

    try:
        # Branch to initialization block
        builder.branch(loop_init_block)
        
        # Generate code for initialization block
        builder.position_at_end(loop_init_block)
        
        # Enter a new scope for the entire for loop (including init variable)
        self.symbol_table.enter_scope(ScopeType.BLOCK, "for_loop")
        
        try:
            # Initialize the loop variable(s) (if any)
            if node.initialization:
                self.process_node(node.initialization, builder=builder, **kwargs)

            # Branch to the condition block
            builder.branch(loop_cond_block)
            
            # Generate code for the condition block
            builder.position_at_end(loop_cond_block)
            if node.condition:
                condition = self.handle_expression(node.condition, builder, self.type_map[Datatypes.BOOL])
                builder.cbranch(condition, loop_body_block, loop_end_block)
            else:
                # If there's no condition, assume an infinite loop unless we break somewhere
                builder.branch(loop_body_block)

            # Generate code for the body block
            builder.position_at_end(loop_body_block)
            for stmt in node.body.nodes:
                if builder.block.is_terminated: 
                    break
                self.process_node(stmt, builder=builder, **kwargs)

            # After the body, move to the update block if not terminated
            if not builder.block.is_terminated:
                builder.branch(loop_update_block)

            # Generate code for the update block (if any)
            builder.position_at_end(loop_update_block)
            if node.update:
                self.process_node(node.update, builder=builder, **kwargs)

            # Return to the condition block for the next iteration
            if not builder.block.is_terminated:
                builder.branch(loop_cond_block)
            
        finally:
            # Exit the for loop scope
            self.symbol_table.exit_scope()
    
    finally:
        # Remove loop context
        self.loop_stack.pop()
    
    # Position the builder at the end block
    builder.position_at_end(loop_end_block)

def handle_break(self, node: ASTNode.Break, builder: ir.IRBuilder, **kwargs):
    """Handle break statement."""
    if not self.loop_stack:
        raise RuntimeError(f"Break statement outside of loop at line {getattr(node, 'line_number', 'unknown')}")
    
    # Get the current loop context
    current_loop = self.loop_stack[-1]
    
    # Branch to the break block (loop end)
    builder.branch(current_loop.break_block)

def handle_continue(self, node: ASTNode.Continue, builder: ir.IRBuilder, **kwargs):
    """Handle continue statement."""
    if not self.loop_stack:
        raise RuntimeError(f"Continue statement outside of loop at line {getattr(node, 'line_number', 'unknown')}")
    
    # Get the current loop context
    current_loop = self.loop_stack[-1]
    
    # Branch to the continue block
    # For while loops: goes to condition check
    # For for loops: goes to update block
    builder.branch(current_loop.continue_block)

def handle_comment(self, node: ASTNode.Comment, **kwargs):
    """Handle comment nodes (no-op)."""
    pass

def handle_compound_statement(self, node: ASTNode.Block, builder: ir.IRBuilder, **kwargs):
    """Handle compound statement (alias for handle_block for compatibility)."""
    return self.handle_block(node, builder, **kwargs)

def _get_current_loop_context(self):
    """Get the current loop context, or None if not in a loop."""
    return self.loop_stack[-1] if self.loop_stack else None

def _is_in_loop(self):
    """Check if currently inside a loop."""
    return len(self.loop_stack) > 0