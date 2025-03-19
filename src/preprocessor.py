from typing import TYPE_CHECKING
import re
import os
from lexer import directives

if TYPE_CHECKING:
    from compiler import Compiler  # Only for type hints

class Preprocessor:
    def __init__(self, compiler: "Compiler"):  # Use string annotation
        self.compiler = compiler
        self.code = self.compiler.src
        self.defines = {}  # Dictionary to store defined macros
        self.function_macros = {}  # Dictionary to store function-like macros
    
    def preprocess(self):
        # Find # symbols (start of directives)
        processed_code = self.code
        cursor = 0
        
        while cursor < len(processed_code):
            # Check if we have a directive (# at beginning of line or after whitespace)
            if cursor == 0 and processed_code[cursor] == '#':
                is_directive = True
            elif cursor > 0 and processed_code[cursor] == '#' and (processed_code[cursor-1].isspace() or processed_code[cursor-1] == '\n'):
                is_directive = True
            else:
                is_directive = False
                
            if is_directive:
                start = cursor
                end = start
                
                # Get whole line
                while end < len(processed_code) and processed_code[end] != '\n':
                    end += 1
                
                line = processed_code[start:end].strip()
                
                # Parse directive - get the first word after #
                parts = line.split(None, 1)
                if not parts:
                    self.syntax_error("Empty directive")
                    cursor = end + 1
                    continue
                
                directive_name = parts[0][1:] if parts[0].startswith('#') else parts[0]  # Remove # if present
                arguments = parts[1] if len(parts) > 1 else ""
                
                # Check against directives dictionary
                full_directive = f"#{directive_name}"
                
                if full_directive == directives.get("DEFINE", "#define"):
                    processed_code = self.handle_define(processed_code, start, end, arguments.strip())
                    cursor = start
                elif full_directive == directives.get("INCLUDE", "#include"):
                    processed_code = self.handle_include(processed_code, start, end, arguments.strip())
                    cursor = start
                elif full_directive == directives.get("UNDEF", "#undef"):
                    processed_code = self.handle_undef(processed_code, start, end, arguments.strip())
                    cursor = start
                elif full_directive == directives.get("IFDEF", "#ifdef"):
                    # Handle conditionally including code if macro is defined
                    processed_code = self.handle_ifdef(processed_code, start, end, arguments.strip())
                    cursor = start
                elif full_directive == directives.get("IFNDEF", "#ifndef"):
                    # Handle conditionally including code if macro is not defined
                    processed_code = self.handle_ifndef(processed_code, start, end, arguments.strip())
                    cursor = start
                elif full_directive == directives.get("ENDIF", "#endif"):
                    # Handle end of conditional compilation
                    processed_code = self.handle_endif(processed_code, start, end)
                    cursor = start
                elif full_directive == directives.get("ELSE", "#else"):
                    # Handle else clause in conditional compilation
                    processed_code = self.handle_else(processed_code, start, end)
                    cursor = start
                else:
                    self.syntax_error(f"Unspecified directive: {parts[0]}")
                    cursor = end + 1
            else:
                cursor += 1
        
        # Process all defined macros in the code
        processed_code = self.replace_defines(processed_code)
        
        # Return processed code
        return processed_code
    
    def syntax_error(self, message, line_num=None):
        # Get line number from position if not provided
        if line_num is None and hasattr(self, 'code'):
            line_num = self.code[:self.code.find(message) + 1].count('\n') + 1 if message in self.code else None
        
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
            self.syntax_error(f"Include file not found: {file_path}")
        
        # Replace the include directive with the file contents
        return code[:start] + file_content + code[end:]
    
    def handle_define(self, code, start, end, arguments):
        # Skip if arguments are empty
        if not arguments:
            self.syntax_error("Invalid #define directive: missing identifier")
            return code
        
        # Check if this is a function-like macro
        match = re.match(r'(\w+)\s*\((.*?)\)\s*(.*)', arguments)
        if match:
            # Function-like macro
            macro_name = match.group(1)
            params_str = match.group(2)
            replacement = match.group(3)
            
            # Handle multi-line macros (continuation with backslash)
            line_end = end
            while line_end < len(code) and line_end > 0 and code[line_end-1] == '\\' and code[line_end] == '\n':
                next_line_end = line_end + 1
                while next_line_end < len(code) and code[next_line_end] != '\n':
                    next_line_end += 1
                
                # Remove the backslash and append the next line
                replacement = replacement[:-1] + code[line_end+1:next_line_end].strip()
                line_end = next_line_end + 1
            
            # Parse parameters
            params = [p.strip() for p in params_str.split(',') if p.strip()]
            
            # Handle variadic macros
            is_variadic = False
            if params and params[-1] == "...":
                is_variadic = True
                params = params[:-1]  # Remove ... from params
            
            # Store function-like macro
            self.function_macros[macro_name] = {
                'params': params,
                'replacement': replacement,
                'is_variadic': is_variadic
            }
            if hasattr(self.compiler, 'defines'):
                self.compiler.defines[macro_name] = f"FUNCTION_MACRO({','.join(params)}): {replacement}"
        else:
            # Object-like macro
            parts = arguments.split(None, 1)
            identifier = parts[0]
            replacement = parts[1] if len(parts) > 1 else ""
            
            # Handle multi-line macros
            line_end = end
            while line_end < len(code) and line_end > 0 and code[line_end-1] == '\\' and code[line_end] == '\n':
                next_line_end = line_end + 1
                while next_line_end < len(code) and code[next_line_end] != '\n':
                    next_line_end += 1
                
                # Remove the backslash and append the next line
                if len(parts) > 1:  # Only append if there's already a replacement
                    replacement = replacement[:-1] + code[line_end+1:next_line_end].strip()
                line_end = next_line_end + 1
            
            self.defines[identifier] = replacement
            # Also update compiler.defines if available (for debugging/reporting)
            if hasattr(self.compiler, 'defines'):
                self.compiler.defines[identifier] = replacement
        
        # Remove the define directive from the code
        return code[:start] + code[end+1:]  # Include the newline
    
    def handle_undef(self, code, start, end, arguments):
        # Parse the identifier
        identifier = arguments.strip()
        
        # Remove the macro from dictionaries
        if identifier in self.defines:
            del self.defines[identifier]
            if hasattr(self.compiler, 'defines') and identifier in self.compiler.defines:
                del self.compiler.defines[identifier]
        if identifier in self.function_macros:
            del self.function_macros[identifier]
            if hasattr(self.compiler, 'defines') and identifier in self.compiler.defines:
                del self.compiler.defines[identifier]
        
        # Remove the undef directive from the code
        return code[:start] + code[end+1:]  # Include the newline
    
    # Placeholder implementations for conditional compilation
    def handle_ifdef(self, code, start, end, arguments):
        # For now, we'll just remove the directive and keep the code
        # A full implementation would need to handle matching #endif and conditional evaluation
        return code[:start] + code[end+1:]
        
    def handle_ifndef(self, code, start, end, arguments):
        # For now, we'll just remove the directive and keep the code
        return code[:start] + code[end+1:]
        
    def handle_endif(self, code, start, end):
        # For now, we'll just remove the directive
        return code[:start] + code[end+1:]
        
    def handle_else(self, code, start, end):
        # For now, we'll just remove the directive
        return code[:start] + code[end+1:]
    
    def replace_defines(self, code):
        # First pass: replace all object-like macros
        for identifier, replacement in self.defines.items():
            # Use word boundaries to avoid partial replacements
            pattern = r'\b' + re.escape(identifier) + r'\b'
            code = re.sub(pattern, replacement, code)
        
        # Second pass: replace function-like macros
        for macro_name, macro_info in self.function_macros.items():
            pattern = r'\b' + re.escape(macro_name) + r'\s*\((.*?)\)'
            
            # Find all instances of the macro
            macro_matches = list(re.finditer(pattern, code))
            # Process in reverse to avoid issues with replacement affecting positions
            for match in reversed(macro_matches):
                full_match = match.group(0)
                args_str = match.group(1)
                
                # Parse arguments with proper handling of nested parentheses
                args = []
                current_arg = ""
                paren_level = 0
                in_string = False
                escape_next = False
                
                for char in args_str:
                    if escape_next:
                        current_arg += char
                        escape_next = False
                        continue
                        
                    if char == '\\':
                        current_arg += char
                        escape_next = True
                        continue
                        
                    if char == '"' and not escape_next:
                        in_string = not in_string
                        current_arg += char
                        continue
                        
                    if not in_string:
                        if char == '(':
                            paren_level += 1
                            current_arg += char
                        elif char == ')':
                            paren_level -= 1
                            current_arg += char
                        elif char == ',' and paren_level == 0:
                            args.append(current_arg.strip())
                            current_arg = ""
                        else:
                            current_arg += char
                    else:
                        current_arg += char
                
                if current_arg:
                    args.append(current_arg.strip())
                
                # Generate replacement
                replacement = macro_info['replacement']
                
                # Process stringizing operator (#)
                for i, param in enumerate(macro_info['params']):
                    if i < len(args):
                        # Handle # operator (stringizing)
                        pattern_stringify = r'#\s*' + re.escape(param) + r'\b'
                        replacement = re.sub(pattern_stringify, f'"{args[i]}"', replacement)
                
                # Process token pasting operator (##)
                while '##' in replacement:
                    replacement = re.sub(r'(\w+)\s*##\s*(\w+)', r'\1\2', replacement)
                
                # Replace parameters with arguments
                for i, param in enumerate(macro_info['params']):
                    if i < len(args):
                        pattern_param = r'\b' + re.escape(param) + r'\b'
                        replacement = re.sub(pattern_param, args[i], replacement)
                
                # Handle variadic arguments (__VA_ARGS__)
                if macro_info['is_variadic'] and len(args) > len(macro_info['params']):
                    va_args = args[len(macro_info['params']):]
                    va_args_str = ', '.join(va_args)
                    replacement = replacement.replace('__VA_ARGS__', va_args_str)
                
                # Replace the macro call with its expansion
                start, end = match.span()
                code = code[:start] + replacement + code[end:]
        
        return code