# preprocessor.py
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from compiler import Compiler  # Only for type hints

class Preprocessor:
    def __init__(self, compiler: "Compiler"):  # Use string annotation
        self.compiler = compiler