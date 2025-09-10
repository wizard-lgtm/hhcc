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
                 array_dimensions: Optional[List[int]] = None,
                 is_mutable: bool = True
                 ):  # NEW

        self.name = name
        self.kind = kind
        self.ast_node = ast_node
        self.data_type = data_type  # The language type (e.g., 'U8', 'MyStruct')
        self.llvm_type = llvm_type  # The LLVM type
        self.llvm_value = llvm_value  # LLVM value or pointer
        self.scope_level = scope_level
        self.pointer_level = pointer_level  # 0 = not a pointer, 1 = pointer, 2 = double pointer, etc.
        self.array_dimensions = array_dimensions or []  # NEW
        self.is_mutable = is_mutable 
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
        return f"Symbol(name='{self.name}', kind={self.kind}, type={self.data_type}, is_mutable={self.is_mutable} scope={self.scope_level}{pointer_str}{array_str})"

class ScopeType(Enum):
    """Different types of scopes in the language."""
    GLOBAL = auto()      # Global scope
    FUNCTION = auto()    # Function scope (parameters + local vars)
    BLOCK = auto()       # Block scope (for loops, if statements, etc.)
    CLASS = auto()       # Class/struct scope
    NAMESPACE = auto()   # Namespace scope

class ScopeInfo:
    """Information about a specific scope."""
    def __init__(self, scope_type: ScopeType, level: int, name: str = None):
        self.scope_type = scope_type
        self.level = level
        self.name = name  # Function name, class name, etc.
        self.symbols: Dict[str, Symbol] = {}
        
    def __repr__(self):
        return f"ScopeInfo(type={self.scope_type}, level={self.level}, name='{self.name}')"


class Scope:
    """Represents a single scope level in the symbol table (backwards-compatible)."""
    def __init__(self, level: int, scope_type: 'ScopeType' = None, name: str = None):
        self.level = level
        self.symbols: Dict[str, Symbol] = {}
        # Backwards-compatible attributes
        self.scope_type = scope_type or ScopeType.BLOCK
        self.name = name

    def define(self, symbol: Symbol) -> None:
        if symbol.name in self.symbols and symbol.kind != SymbolKind.PARAMETER:
            raise ValueError(f"Symbol '{symbol.name}' already defined in scope {self.level}")
        self.symbols[symbol.name] = symbol

    def lookup(self, name: str) -> Optional[Symbol]:
        return self.symbols.get(name)

    def get_all_symbols(self) -> List[Symbol]:
        return list(self.symbols.values())

    def __repr__(self):
        return f"Scope(level={self.level}, type={self.scope_type}, name={self.name})"


class SymbolTable:
    """
    Symbol table with backwards compatibility for old API (ImprovedSymbolTable).
    """
    def __init__(self):
        self.scopes: List[Scope] = []
        self.current_scope_level = -1
        # Start with global scope
        self.enter_scope(ScopeType.GLOBAL, "global")

    @property
    def current_scope(self) -> Scope:
        return self.scopes[self.current_scope_level]

    # --- New + Old API unified ---
    def enter_scope(self, scope_type: 'ScopeType' = ScopeType.BLOCK, name: str = None) -> int:
        """Enter a new scope (compatible with old API)."""
        self.current_scope_level += 1
        if len(self.scopes) <= self.current_scope_level:
            self.scopes.append(Scope(self.current_scope_level, scope_type, name))
        else:
            scope = self.scopes[self.current_scope_level]
            scope.scope_type = scope_type
            scope.name = name
        print(f"SCOPE: Entering {scope_type} scope '{name}' at level {self.current_scope_level}")
        return self.current_scope_level

    def exit_scope(self) -> int:
        """Exit current scope (compatible with old API)."""
        if self.current_scope_level > 0:
            old_scope = self.current_scope
            print(f"SCOPE: Exiting {old_scope.scope_type} scope '{old_scope.name}' from level {self.current_scope_level}")
            self.current_scope_level -= 1
        return self.current_scope_level

    def define(self, symbol: Symbol) -> Symbol:
        symbol.scope_level = self.current_scope_level
        self.current_scope.define(symbol)
        print(f"SCOPE: Defined '{symbol.name}' in {self.current_scope.scope_type} scope '{self.current_scope.name}' (level {self.current_scope.level})")
        return symbol

    def lookup(self, name: str, current_scope_only: bool = False) -> Optional[Symbol]:
        if current_scope_only:
            return self.current_scope.lookup(name)
        for level in range(self.current_scope_level, -1, -1):
            symbol = self.scopes[level].lookup(name)
            if symbol:
                print(f"SCOPE: Found '{name}' in {self.scopes[level].scope_type} scope '{self.scopes[level].name}' (level {level})")
                return symbol
        print(f"SCOPE: Symbol '{name}' not found in any scope")
        return None

    def get_function_scope(self) -> Optional[Scope]:
        """Backwards-compatible function scope lookup."""
        for level in range(self.current_scope_level, -1, -1):
            scope = self.scopes[level]
            if scope.scope_type == ScopeType.FUNCTION:
                return scope
        return None

    def dump_scopes(self) -> str:
        """Backwards-compatible scope dump."""
        result = "=== SCOPE DUMP ===\n"
        for i, scope in enumerate(self.scopes):
            marker = " <- CURRENT" if i == self.current_scope_level else ""
            result += f"Level {i}: {scope}{marker}\n"
            for name, symbol in scope.symbols.items():
                result += f"  - {name}: {symbol}\n"
        result += "================\n"
        return result


    def __contains__(self, name: str) -> bool:
        """Allow 'in' operator to check if a symbol exists (backwards compat)."""
        return self.exists(name)

    def __iter__(self):
        """Allow iteration over all symbol names (backwards compat)."""
        for scope in self.scopes:
            for name in scope.symbols:
                yield name

    def keys(self):
        """Backwards compat: get all symbol names."""
        return list(iter(self))

    def values(self):
        """Backwards compat: get all symbol objects."""
        for scope in self.scopes:
            for symbol in scope.symbols.values():
                yield symbol

    def items(self):
        """Backwards compat: get all (name, symbol) pairs."""
        for scope in self.scopes:
            for name, symbol in scope.symbols.items():
                yield name, symbol

    def exists(self, name: str, current_scope_only: bool = False) -> bool:
        """Check if a symbol exists in the symbol table (backwards compat)."""
        return self.lookup(name, current_scope_only) is not None

    def __getitem__(self, name: str) -> Optional[Symbol]:
        return self.lookup(name)



# Helper functions for creating common types of symbols
def create_variable_symbol(name: str, ast_node: Any, data_type: Any, 
                          llvm_type: Any = None, llvm_value: Optional[ir.Value] = None, 
                          scope_level: int = 0, pointer_level: int = 0, is_mutable: bool = True) -> Symbol:
    """Create a variable symbol."""
    return Symbol(name, SymbolKind.VARIABLE, ast_node, data_type, 
                 llvm_type, llvm_value, scope_level, pointer_level, is_mutable=is_mutable)


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