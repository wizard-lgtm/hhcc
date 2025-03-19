from typing import TYPE_CHECKING
from lexer import directives
import os
if TYPE_CHECKING:
    from compiler import Compiler  # Only for type hints

class Preprocessor:
    def __init__(self, compiler: "Compiler"):  # Use string annotation
        self.compiler = compiler
        self.code = self.compiler.src
        self.defines = {}  # Dictionary to store defined macros
    
    def preprocess(self):
        # Find # symbols (start of directives)
        processed_code = self.code
        cursor = 0
        
        while cursor < len(processed_code):
            if processed_code[cursor] == "#":
                start = cursor
                end = start
                
                # Get whole line
                while end < len(processed_code) and processed_code[end] != '\n':
                    end += 1
                
                line = processed_code[start:end]
                
                # Parse directive
                parts = line.split(" ", 1)
                print(parts[0])
                if len(parts) < 1 or parts[0] not in directives.values():
                    self.syntax_error(f"Unspecified directive: {parts[0] if parts else '#'}")
                
                directive = parts[0]
                arguments = parts[1] if len(parts) > 1 else ""
                
                # Handle different directives
                if directive == directives["INCLUDE"]:
                    processed_code = self.handle_include(processed_code, start, end, arguments.strip())
                    # Reset cursor to start position since we've modified the code
                    cursor = start
                elif directive == directives["DEFINE"]:
                    processed_code = self.handle_define(processed_code, start, end, arguments.strip())
                    cursor = start
                # Add other directive handlers here
                
            cursor += 1
            
        # Process all defined macros in the code
        processed_code = self.replace_defines(processed_code)
        
        # Return processed code
        return processed_code
    
    def syntax_error(self, message, line_num=None):
        # TODO! add line and caret
        line_info = f"in line: {line_num}" if line_num else ""
        raise Exception(f"{message} {line_info}")

    def handle_include(self, code, start, end, arguments):
        # Get the file path from arguments
        file_path = arguments.strip('"\'')
        
        # Try to open the file using different paths
        file_content = None
        paths_to_try = [
            file_path,  # Direct path
            os.path.join(self.compiler.file_directory, file_path),  # Relative to source file
            os.path.join(self.compiler.working_directory, file_path)  # Relative to working directory
        ]
        
        for path in paths_to_try:
            try:
                with open(path) as file:
                    file_content = file.read()
                    break
            except FileNotFoundError:
                continue
        
        if file_content is None:
            self.syntax_error(f"Include file not found: {file_path} (tried paths: {paths_to_try})")
        
        # Replace the include directive with the file contents
        return code[:start] + file_content + code[end:]
    
    def handle_define(self, code, start, end, arguments):
        # Parse the define directive
        parts = arguments.strip().split(" ", 1)
        
        if len(parts) < 1:
            self.syntax_error("Invalid #define directive: missing identifier")
        
        identifier = parts[0]
        replacement = parts[1] if len(parts) > 1 else ""
        
        # Store the define in our dictionary
        self.compiler.defines[identifier] = replacement
        
        # Remove the define directive from the code
        return code[:start] + code[end:]
    
    def replace_defines(self, code):
        # Replace all defined macros in the code
        for identifier, replacement in self.compiler.defines.items():
            # Simple token replacement (could be enhanced for function-like macros)
            # Using word boundaries to avoid partial replacements
            import re
            pattern = r'\b' + re.escape(identifier) + r'\b'
            code = re.sub(pattern, replacement, code)
        
        return code