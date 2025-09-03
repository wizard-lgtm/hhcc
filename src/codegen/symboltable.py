from llvmlite import ir, binding
from typing import TYPE_CHECKING, Dict, Callable, Type
from astnodes import *
from lexer import *

class SymbolKind(Enum):
    """Enum for different kinds of symbols."""
    VARIABLE = auto()
    FUNCTION = auto()
    TYPE = auto()
    PARAMETER = auto()
    CLASS = auto()
    UNION = auto()

class Symbol:
    """
    Represents a symbol in the symbol table.
    A symbol can be a variable, function, type, etc.
    """
    def __init__(self, 
                 name: str, 
                 kind: SymbolKind, 
                 ast_node: Any, 
                 data_type: Any,
                 llvm_type: Any = None, 
                 llvm_value: Optional[ir.Value] = None, 
                 scope_level: int = 0,
                 pointer_level: int = 0,
                 array_dimensions: Optional[List[int]] = None):  # NEW

        self.name = name
        self.kind = kind
        self.ast_node = ast_node
        self.data_type = data_type  # The language type (e.g., 'U8', 'MyStruct')
        self.llvm_type = llvm_type  # The LLVM type
        self.llvm_value = llvm_value  # LLVM value or pointer
        self.scope_level = scope_level
        self.pointer_level = pointer_level  # 0 = not a pointer, 1 = pointer, 2 = double pointer, etc.
        self.array_dimensions = array_dimensions or []  # NEW
        # Additional data for specific symbol kinds

        self.extra_data: Dict[str, Any] = {}

    @property
    def is_array(self) -> bool:
        return bool(self.array_dimensions)
    @property
    def is_pointer(self) -> bool:
        """Backward compatibility property."""
        return self.pointer_level > 0

    def __repr__(self) -> str:
        pointer_str = f", ptr_level={self.pointer_level}" if self.pointer_level > 0 else ""
        array_str = f", dims={self.array_dimensions}" if self.is_array else ""
        return f"Symbol(name='{self.name}', kind={self.kind}, type={self.data_type}, scope={self.scope_level}{pointer_str}{array_str})"


class Scope:
    """Represents a single scope level in the symbol table."""
    def __init__(self, level: int):
        self.level = level
        self.symbols: Dict[str, Symbol] = {}

    def define(self, symbol: Symbol) -> None:
        """Define a symbol in this scope."""
        if symbol.name in self.symbols:
            raise ValueError(f"Symbol '{symbol.name}' already defined in scope {self.level}")
        self.symbols[symbol.name] = symbol

    def lookup(self, name: str) -> Optional[Symbol]:
        """Look up a symbol in this scope."""
        return self.symbols.get(name)

    def get_all_symbols(self) -> List[Symbol]:
        """Get all symbols in this scope."""
        return list(self.symbols.values())


class SymbolTable:
    """
    Symbol table with scope management.
    Manages multiple scopes for block-level variable declarations.
    """
    def __init__(self):
        self.scopes: List[Scope] = [Scope(0)]  # Start with global scope (level 0)
        self.current_scope_level = 0

    @property
    def current_scope(self) -> Scope:
        """Get the current scope."""
        return self.scopes[self.current_scope_level]

    def enter_scope(self) -> int:
        """
        Enter a new scope level.
        Returns the new scope level.
        """
        self.current_scope_level += 1
        # Create the new scope if it doesn't exist
        if len(self.scopes) <= self.current_scope_level:
            self.scopes.append(Scope(self.current_scope_level))
        return self.current_scope_level

    def exit_scope(self) -> int:
        """
        Exit the current scope level.
        Returns the new (previous) scope level.
        """
        if self.current_scope_level > 0:
            self.current_scope_level -= 1
        return self.current_scope_level

    def define(self, symbol: Symbol) -> Symbol:
        """
        Define a symbol in the current scope.
        Sets the scope level on the symbol and adds it to the current scope.
        """
        # Set the scope level on the symbol
        symbol.scope_level = self.current_scope_level
        # Add to current scope
        self.current_scope.define(symbol)
        return symbol

    def lookup(self, name: str, current_scope_only: bool = False) -> Optional[Symbol]:
        """
        Look up a symbol by name.
        If current_scope_only is True, only look in the current scope.
        Otherwise, look in all scopes from current to global.
        """
        if current_scope_only:
            return self.current_scope.lookup(name)
        
        # Search from current scope up to global scope
        for level in range(self.current_scope_level, -1, -1):
            symbol = self.scopes[level].lookup(name)
            if symbol:
                return symbol
        
        return None

    def lookup_in_scope(self, name: str, scope_level: int) -> Optional[Symbol]:
        """Look up a symbol in a specific scope level."""
        if 0 <= scope_level < len(self.scopes):
            return self.scopes[scope_level].lookup(name)
        return None

    def remove(self, name: str, scope_level: Optional[int] = None) -> bool:
        """
        Remove a symbol from the specified scope or current scope if not specified.
        Returns True if the symbol was found and removed, False otherwise.
        """
        if scope_level is None:
            scope_level = self.current_scope_level
        
        if 0 <= scope_level < len(self.scopes):
            scope = self.scopes[scope_level]
            if name in scope.symbols:
                del scope.symbols[name]
                return True
        
        return False

    def exists(self, name: str, current_scope_only: bool = False) -> bool:
        """Check if a symbol exists in the symbol table."""
        return self.lookup(name, current_scope_only) is not None

    def get_all_symbols(self, include_parent_scopes: bool = False) -> List[Symbol]:
        """
        Get all symbols in the current scope.
        If include_parent_scopes is True, include symbols from all parent scopes.
        """
        if not include_parent_scopes:
            return self.current_scope.get_all_symbols()
        
        all_symbols = []
        for level in range(self.current_scope_level + 1):
            all_symbols.extend(self.scopes[level].get_all_symbols())
        
        return all_symbols

    def get_functions(self) -> List[Symbol]:
        """Get all function symbols."""
        functions = []
        for scope in self.scopes:
            for symbol in scope.symbols.values():
                if symbol.kind == SymbolKind.FUNCTION:
                    functions.append(symbol)
        return functions

    def get_types(self) -> List[Symbol]:
        """Get all type symbols (classes, unions, etc.)."""
        types = []
        for scope in self.scopes:
            for symbol in scope.symbols.values():
                if symbol.kind == SymbolKind.TYPE or symbol.kind == SymbolKind.CLASS or symbol.kind == SymbolKind.UNION:
                    types.append(symbol)
        return types

    def dump(self) -> str:
        """Dump the symbol table as a string for debugging."""
        result = "Symbol Table:\n"
        for level, scope in enumerate(self.scopes):
            result += f"  Scope level {level}:\n"
            for name, symbol in scope.symbols.items():
                result += f"    {symbol}\n"
        return result

    def __getitem__(self, name: str) -> Optional[Symbol]:
        """Allow dict-like access to the symbol table."""
        return self.lookup(name)

    def __contains__(self, name: str) -> bool:
        """Allow 'in' operator to check if a symbol exists."""
        return self.exists(name)


# Helper functions for creating common types of symbols
def create_variable_symbol(name: str, ast_node: Any, data_type: Any, 
                          llvm_type: Any = None, llvm_value: Optional[ir.Value] = None, 
                          scope_level: int = 0, pointer_level: int = 0) -> Symbol:
    """Create a variable symbol."""
    return Symbol(name, SymbolKind.VARIABLE, ast_node, data_type, 
                 llvm_type, llvm_value, scope_level, pointer_level)


def create_function_symbol(name: str, ast_node: Any, return_type: Any, 
                          parameter_types: List[Any], llvm_function: Optional[ir.Function] = None, 
                          scope_level: int = 0) -> Symbol:
    """Create a function symbol."""
    symbol = Symbol(name, SymbolKind.FUNCTION, ast_node, return_type, 
                   None, llvm_function, scope_level, 0)  # Functions are not pointers by default
    symbol.extra_data['parameter_types'] = parameter_types
    return symbol


def create_type_symbol(name: str, ast_node: Any, llvm_type: Any = None, 
                      scope_level: int = 0) -> Symbol:
    """Create a type symbol (class, struct, enum, etc.)."""
    symbol = Symbol(name, SymbolKind.TYPE, ast_node, name, llvm_type, None, scope_level, 0)
    return symbol

def create_array_symbol(name: str, ast_node: Any, element_type: Any, dimensions: List[int], 
                        llvm_type: Any = None, llvm_value: Optional[ir.Value] = None, 
                        scope_level: int = 0) -> Symbol:
    """
    Create a symbol representing an array.
    - element_type: type of the elements (e.g., 'U8', 'I32')
    - dimensions: list of sizes for each dimension (e.g., [10] for 1D, [3,4] for 2D)
    """
    symbol = Symbol(name, SymbolKind.VARIABLE, ast_node, element_type, 
                    llvm_type, llvm_value, scope_level, pointer_level=0,  # arrays are pointers
                    array_dimensions=dimensions)
    return symbol