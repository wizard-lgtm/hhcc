from llvmlite import ir
from typing import TYPE_CHECKING, Dict, Callable, Type
from astnodes import *
from lexer import *
def handle_block(self, node: ASTNode.Block, builder: ir.IRBuilder, **kwargs):
    """Handle a block of statements with proper scope management."""
    # Enter a new scope for this block
    self.symbol_table.enter_scope()
    
    # Process each statement in the block
    for stmt in node.nodes:
        self.process_node(stmt, builder=builder, **kwargs)
    
    # Exit the scope when done with the block
    self.symbol_table.exit_scope()
def handle_return(self, node: ASTNode.Return, builder: ir.IRBuilder, **kwargs):
    # If the value is a function
    # Get return type
    function_return_type = builder.function.function_type.return_type
    if node.expression:
        return_value = self.handle_expression(node.expression, builder, function_return_type)
        builder.ret(return_value)
    else:
        builder.ret_void()
        
def handle_if_statement(self, node: ASTNode.IfStatement, builder: ir.IRBuilder, **kwargs):
    # Evaluate the condition
    condition = self.handle_expression(node.condition, builder, self.type_map[Datatypes.BOOL])

    condition = self.handle_expression(node.condition, builder, self.type_map[Datatypes.BOOL])
    if condition.type != ir.IntType(1):
        condition = builder.trunc(condition, ir.IntType(1))


    # Create basic blocks for the 'then', 'else', and 'merge' sections
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
    for stmt in node.if_body.nodes:
        self.process_node(stmt, builder=builder)
    if not builder.block.is_terminated:
        builder.branch(merge_block)

    # Generate code for the 'else' block if it exists
    if else_block:
        builder.position_at_end(else_block)
        for stmt in node.else_body.nodes:
            self.process_node(stmt, builder=builder)
        if not builder.block.is_terminated:
            builder.branch(merge_block)

    # Position the builder at the merge block
    builder.position_at_end(merge_block)

def handle_while_loop(self, node: ASTNode.WhileLoop, builder: ir.IRBuilder):
    # Create basic blocks for the loop
    loop_cond_block = builder.append_basic_block("while.cond")
    loop_body_block = builder.append_basic_block("while.body")
    loop_end_block = builder.append_basic_block("while.end")

    # Branch to the condition block
    builder.branch(loop_cond_block)

    # Generate code for the condition block
    builder.position_at_end(loop_cond_block)
    condition = self.handle_expression(node.condition, builder, self.type_map[Datatypes.BOOL])
    builder.cbranch(condition, loop_body_block, loop_end_block)

    # Generate code for the body block
    builder.position_at_end(loop_body_block)
    for stmt in node.body.nodes:
        self.process_node(stmt, builder=builder)
    builder.branch(loop_cond_block)

    # Position the builder at the end block
    builder.position_at_end(loop_end_block)

def handle_for_loop(self, node: ASTNode.ForLoop, builder: ir.IRBuilder, **kwargs):
    # Create basic blocks for the for loop
    loop_cond_block = builder.append_basic_block("for.cond")
    loop_body_block = builder.append_basic_block("for.body")
    loop_update_block = builder.append_basic_block("for.update")
    loop_end_block = builder.append_basic_block("for.end")

    # Initialize the loop variable(s) (if any)
    if node.initialization:
        self.process_node(node.initialization, builder=builder)

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
        self.process_node(stmt, builder=builder)

    # After the body, move to the update block
    builder.branch(loop_update_block)

    # Generate code for the update block (if any)
    builder.position_at_end(loop_update_block)
    if node.update:
        self.process_node(node.update, builder=builder)

    # Return to the condition block for the next iteration
    builder.branch(loop_cond_block)
    
    # Position the builder at the end block
    builder.position_at_end(loop_end_block)

def handle_break(self, node, **kwargs):
    pass

def handle_continue(self, node, **kwargs):
    pass

def handle_comment(self, node: ASTNode.Comment, **kwargs):
    pass