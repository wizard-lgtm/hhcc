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
    
    def remove_comments(self, line):
        """Remove single-line and multi-line comments from a line"""
        # Handle single-line comments //
        comment_pos = line.find('//')
        if comment_pos != -1:
            # Check if // is inside a string literal
            in_string = False
            escape_next = False
            for i, char in enumerate(line[:comment_pos]):
                if escape_next:
                    escape_next = False
                    continue
                if char == '\\':
                    escape_next = True
                elif char == '"' and not escape_next:
                    in_string = not in_string
            
            if not in_string:
                line = line[:comment_pos]
        
        return line.strip()
    
    def is_inside_string_or_char(self, text, position):
        """Check if the position is inside a string literal or character literal"""
        in_string = False
        in_char = False
        escape_next = False
        
        for i in range(position):
            if i >= len(text):
                break
                
            char = text[i]
            
            if escape_next:
                escape_next = False
                continue
                
            if char == '\\':
                escape_next = True
                continue
                
            if not in_char and char == '"' and not escape_next:
                in_string = not in_string
            elif not in_string and char == "'" and not escape_next:
                in_char = not in_char
                
        return in_string or in_char
    
    def apply_macro_replacements(self, line):
        """Apply current macro definitions to a line of code, but skip string/char literals"""
        original_line = line
        
        # First pass: replace all object-like macros
        # Sort by length (longest first) to avoid partial matches
        sorted_defines = sorted(self.defines.items(), key=lambda x: len(x[0]), reverse=True)
        
        for identifier, replacement in sorted_defines:
            # Use word boundaries to avoid partial replacements
            pattern = r'\b' + re.escape(identifier) + r'\b'
            
            # Find all matches
            matches = list(re.finditer(pattern, line))
            
            # Process matches in reverse order to avoid position shifts
            for match in reversed(matches):
                start_pos = match.start()
                end_pos = match.end()
                
                # Check if this match is inside a string or character literal
                if not self.is_inside_string_or_char(line, start_pos):
                    # Safe to replace
                    line = line[:start_pos] + replacement + line[end_pos:]
        
        # Second pass: replace function-like macros
        # Sort by name length (longest first) to handle overlapping macro names
        sorted_function_macros = sorted(self.function_macros.items(), key=lambda x: len(x[0]), reverse=True)
        
        for macro_name, macro_info in sorted_function_macros:
            # Use a more robust pattern that handles whitespace better
            pattern = r'\b' + re.escape(macro_name) + r'\s*\(((?:[^()]*|\([^()]*\))*)\)'
            
            # Find all instances of the macro in this line
            macro_matches = list(re.finditer(pattern, line))
            
            # Process in reverse to avoid issues with replacement affecting positions
            for match in reversed(macro_matches):
                start_pos = match.start()
                end_pos = match.end()
                
                # Check if this match is inside a string or character literal
                if self.is_inside_string_or_char(line, start_pos):
                    continue  # Skip this match, it's inside a string/char literal
                
                full_match = match.group(0)
                args_str = match.group(1)
                
                # Parse arguments with proper handling of nested parentheses
                args = []
                if args_str.strip():  # Only parse if there are arguments
                    current_arg = ""
                    paren_level = 0
                    in_string = False
                    in_char = False
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
                            
                        if char == '"' and not escape_next and not in_char:
                            in_string = not in_string
                            current_arg += char
                            continue
                            
                        if char == "'" and not escape_next and not in_string:
                            in_char = not in_char
                            current_arg += char
                            continue
                            
                        if not in_string and not in_char:
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
                for j, param in enumerate(macro_info['params']):
                    if j < len(args):
                        # Handle # operator (stringizing)
                        pattern_stringify = r'#\s*' + re.escape(param) + r'\b'
                        replacement = re.sub(pattern_stringify, f'"{args[j]}"', replacement)
                
                # Process token pasting operator (##)
                while '##' in replacement:
                    old_replacement = replacement
                    replacement = re.sub(r'(\w+)\s*##\s*(\w+)', r'\1\2', replacement)
                    if old_replacement == replacement:
                        break  # Avoid infinite loop
                
                # Replace parameters with arguments
                for j, param in enumerate(macro_info['params']):
                    if j < len(args):
                        pattern_param = r'\b' + re.escape(param) + r'\b'
                        replacement = re.sub(pattern_param, args[j], replacement)
                
                # Handle variadic arguments (__VA_ARGS__)
                if macro_info['is_variadic'] and len(args) > len(macro_info['params']):
                    va_args = args[len(macro_info['params']):]
                    va_args_str = ', '.join(va_args)
                    replacement = replacement.replace('__VA_ARGS__', va_args_str)
                
                # Replace the macro call with its expansion
                line = line[:start_pos] + replacement + line[end_pos:]
        
        return line
    
    def preprocess(self):
        # Split code into lines for easier processing
        lines = self.code.split('\n')
        processed_lines = []
        
        # Stack to track conditional compilation blocks
        conditional_stack = []
        skip_lines = False
        
        for line_num, line in enumerate(lines):
            # Remove comments before processing directives
            line_without_comments = self.remove_comments(line)
            
            # Skip empty lines after comment removal
            if not line_without_comments:
                if not skip_lines:
                    processed_lines.append(line)  # Keep original line to preserve formatting
                continue
                
            # Check for preprocessor directives only on non-comment content
            if line_without_comments.strip().startswith('#'):
                # This is a preprocessor directive, process it
                directive_line = line_without_comments.strip()
                parts = directive_line.split(None, 1)
                
                if not parts:
                    self.syntax_error("Empty directive", line_num + 1)
                    if not skip_lines:
                        processed_lines.append(line)
                    continue
                
                directive_name = parts[0][1:] if parts[0].startswith('#') else parts[0]  # Remove # if present
                arguments = parts[1] if len(parts) > 1 else ""
                full_directive = f"#{directive_name}"
                
                # Handle conditional compilation directives first
                if full_directive == directives.get("IFDEF", "#ifdef"):
                    condition_met = self.evaluate_ifdef(arguments.strip())
                    conditional_stack.append({'type': 'ifdef', 'condition_met': condition_met, 'has_else': False})
                    skip_lines = not condition_met
                    continue
                    
                elif full_directive == directives.get("IFNDEF", "#ifndef"):
                    condition_met = self.evaluate_ifndef(arguments.strip())
                    conditional_stack.append({'type': 'ifndef', 'condition_met': condition_met, 'has_else': False})
                    skip_lines = not condition_met
                    continue
                    
                elif full_directive in [directives.get("ELIFDEF", "#elifdef"), "#elifdef"]:
                    if not conditional_stack:
                        self.syntax_error("Unexpected #elifdef without #ifdef or #ifndef", line_num + 1)
                        continue
                        
                    current_block = conditional_stack[-1]
                    if current_block['has_else']:
                        self.syntax_error("#elifdef after #else", line_num + 1)
                        continue
                        
                    # If previous condition was met, skip this elifdef
                    if current_block['condition_met']:
                        skip_lines = True
                    else:
                        # Evaluate this elifdef condition
                        condition_met = self.evaluate_ifdef(arguments.strip())
                        current_block['condition_met'] = condition_met
                        skip_lines = not condition_met
                    continue
                    
                elif full_directive in [directives.get("ELIFNDEF", "#elifndef"), "#elifndef"]:
                    if not conditional_stack:
                        self.syntax_error("Unexpected #elifndef without #ifdef or #ifndef", line_num + 1)
                        continue
                        
                    current_block = conditional_stack[-1]
                    if current_block['has_else']:
                        self.syntax_error("#elifndef after #else", line_num + 1)
                        continue
                        
                    # If previous condition was met, skip this elifndef
                    if current_block['condition_met']:
                        skip_lines = True
                    else:
                        # Evaluate this elifndef condition
                        condition_met = self.evaluate_ifndef(arguments.strip())
                        current_block['condition_met'] = condition_met
                        skip_lines = not condition_met
                    continue
                    
                elif full_directive == directives.get("ELSE", "#else"):
                    if not conditional_stack:
                        self.syntax_error("Unexpected #else without #ifdef or #ifndef", line_num + 1)
                        continue
                        
                    current_block = conditional_stack[-1]
                    if current_block['has_else']:
                        self.syntax_error("Multiple #else in conditional block", line_num + 1)
                        continue
                        
                    current_block['has_else'] = True
                    # If previous condition was met, skip else block
                    skip_lines = current_block['condition_met']
                    continue
                    
                elif full_directive == directives.get("ENDIF", "#endif"):
                    if not conditional_stack:
                        self.syntax_error("Unexpected #endif without #ifdef or #ifndef", line_num + 1)
                        continue
                        
                    conditional_stack.pop()
                    # Update skip_lines based on remaining stack
                    skip_lines = any(not block['condition_met'] for block in conditional_stack)
                    continue
                    
                elif full_directive in [directives.get("ERROR", "#error"), "#error"]:
                    # Always process #error, even if in skipped block
                    error_message = arguments.strip().strip('"\'')
                    self.syntax_error(f"#error: {error_message}", line_num + 1)
                    continue
                
                # Skip other directives if we're in a false conditional block
                if skip_lines:
                    continue
                
                # Process other directives
                if full_directive == directives.get("DEFINE", "#define"):
                    # For now, just remove the line and store the define
                    self.handle_define_simple(arguments.strip())
                    continue  # Don't add this line to processed_lines
                elif full_directive == directives.get("INCLUDE", "#include"):
                    # Handle include - the included content is already preprocessed
                    include_content = self.handle_include_simple(arguments.strip())
                    # Split and add the already-processed lines
                    include_lines = include_content.split('\n')
                    processed_lines.extend(include_lines)
                    continue
                elif full_directive == directives.get("UNDEF", "#undef"):
                    # Handle undef
                    self.handle_undef_simple(arguments.strip())
                    continue  # Don't add this line to processed_lines
                else:
                    self.syntax_error(f"Unspecified directive: {parts[0]}", line_num + 1)
                    processed_lines.append(line)
            else:
                # Regular code line, only keep if not in skipped block
                if not skip_lines:
                    # Apply macro replacements to this line before adding it
                    processed_line = self.apply_macro_replacements(line)
                    processed_lines.append(processed_line)
        
        # Check for unmatched conditional directives
        if conditional_stack:
            self.syntax_error("Unmatched conditional directive(s) - missing #endif")
        
        # Join lines back together
        processed_code = '\n'.join(processed_lines)
        
        # Debug output
        print(f"DEBUG: Final defines state: {self.defines}")
        print(f"DEBUG: Final function macros state: {self.function_macros}")
        
        # Return processed code (no need for replace_defines anymore)
        return processed_code
    
    def evaluate_ifdef(self, identifier):
        """Evaluate #ifdef condition"""
        identifier = identifier.strip()
        return identifier in self.defines or identifier in self.function_macros
    
    def evaluate_ifndef(self, identifier):
        """Evaluate #ifndef condition"""
        identifier = identifier.strip()
        return not (identifier in self.defines or identifier in self.function_macros)
    
    def handle_define_simple(self, arguments):
        """Simplified define handler for line-by-line processing"""
        if not arguments:
            self.syntax_error("Invalid #define directive: missing identifier")
            return
        
        # Check if this is a function-like macro
        # IMPORTANT: No whitespace allowed between macro name and opening parenthesis
        match = re.match(r'(\w+)\((.*?)\)\s*(.*)', arguments)
        if match:
            # Function-like macro
            macro_name = match.group(1)
            params_str = match.group(2)
            replacement = match.group(3)
            
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
            
            self.defines[identifier] = replacement
            # Also update compiler.defines if available (for debugging/reporting)
            if hasattr(self.compiler, 'defines'):
                self.compiler.defines[identifier] = replacement
    
    def handle_include_simple(self, arguments):
        """Simplified include handler for line-by-line processing"""
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
            return ""
        
        # Create a temporary preprocessor for the included file
        # to handle any directives it might contain
        temp_preprocessor = Preprocessor.__new__(Preprocessor)
        temp_preprocessor.compiler = self.compiler
        temp_preprocessor.code = file_content
        temp_preprocessor.defines = self.defines.copy()  # Share current defines
        temp_preprocessor.function_macros = self.function_macros.copy()  # Share current macros
        
        # Preprocess the included file
        processed_content = temp_preprocessor.preprocess()
        
        # Update our defines with any new ones from the included file
        self.defines.update(temp_preprocessor.defines)
        self.function_macros.update(temp_preprocessor.function_macros)
        
        return processed_content
    
    def handle_undef_simple(self, arguments):
        """Simplified undef handler for line-by-line processing"""
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
    
    def syntax_error(self, message, line_num=None):
        # Get line number from position if not provided
        if line_num is None and hasattr(self, 'code'):
            line_num = self.code[:self.code.find(message) + 1].count('\n') + 1 if message in self.code else None
        
        line_info = f"in line: {line_num}" if line_num else ""
        raise Exception(f"{message} {line_info}")

    # Keep all the other methods for compatibility, but they won't be used in the main flow
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
        # IMPORTANT: No whitespace allowed between macro name and opening parenthesis
        match = re.match(r'(\w+)\((.*?)\)\s*(.*)', arguments)
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

    def handle_ifdef(self, code, start, end, arguments):
        """
        Handle #ifdef directive - this is a placeholder implementation.
        Note: The main preprocess() method already handles conditional compilation properly.
        This method would be used in a different processing approach.
        """
        # Parse the identifier
        identifier = arguments.strip()
        
        # Find the matching #endif
        endif_pos = self.find_matching_endif(code, end)
        if endif_pos == -1:
            self.syntax_error(f"Unmatched #ifdef for '{identifier}' - missing #endif")
            return code
        
        # Check if the identifier is defined
        is_defined = identifier in self.defines or identifier in self.function_macros
        
        if is_defined:
            # Keep the content between #ifdef and #endif, remove the directives
            content = code[end+1:endif_pos[0]]
            return code[:start] + content + code[endif_pos[1]+1:]
        else:
            # Remove everything from #ifdef to #endif
            return code[:start] + code[endif_pos[1]+1:]

    def handle_ifndef(self, code, start, end, arguments):
        """
        Handle #ifndef directive - this is a placeholder implementation.
        Note: The main preprocess() method already handles conditional compilation properly.
        This method would be used in a different processing approach.
        """
        # Parse the identifier
        identifier = arguments.strip()
        
        # Find the matching #endif
        endif_pos = self.find_matching_endif(code, end)
        if endif_pos == -1:
            self.syntax_error(f"Unmatched #ifndef for '{identifier}' - missing #endif")
            return code
        
        # Check if the identifier is NOT defined
        is_not_defined = not (identifier in self.defines or identifier in self.function_macros)
        
        if is_not_defined:
            # Keep the content between #ifndef and #endif, remove the directives
            content = code[end+1:endif_pos[0]]
            return code[:start] + content + code[endif_pos[1]+1:]
        else:
            # Remove everything from #ifndef to #endif
            return code[:start] + code[endif_pos[1]+1:]

    def handle_endif(self, code, start, end):
        """
        Handle #endif directive - this is a placeholder implementation.
        Note: The main preprocess() method already handles conditional compilation properly.
        This method would be used in a different processing approach.
        """
        # In this implementation approach, #endif is handled by the #ifdef/#ifndef handlers
        # This method would only be called for unmatched #endif
        self.syntax_error("Unmatched #endif without corresponding #ifdef or #ifndef")
        return code

    def handle_else(self, code, start, end):
        """
        Handle #else directive - this is a placeholder implementation.
        Note: The main preprocess() method already handles conditional compilation properly.
        This method would be used in a different processing approach.
        """
        # In this implementation approach, #else would be handled by the #ifdef/#ifndef handlers
        # This method would only be called for unmatched #else
        self.syntax_error("Unmatched #else without corresponding #ifdef or #ifndef")
        return code

    def find_matching_endif(self, code, start_pos):
        """
        Helper method to find the matching #endif for a given #ifdef/#ifndef.
        Returns tuple (start_pos, end_pos) of the #endif directive, or -1 if not found.
        """
        pos = start_pos
        nest_level = 1  # We're already inside one conditional block
        
        while pos < len(code):
            # Find the next preprocessor directive
            hash_pos = code.find('#', pos)
            if hash_pos == -1:
                break
                
            # Get the line containing the directive
            line_start = hash_pos
            while line_start > 0 and code[line_start-1] != '\n':
                line_start -= 1
                
            line_end = hash_pos
            while line_end < len(code) and code[line_end] != '\n':
                line_end += 1
                
            directive_line = code[line_start:line_end].strip()
            
            # Skip if this # is not at the start of the directive (could be in a comment)
            if not directive_line.startswith('#'):
                pos = hash_pos + 1
                continue
                
            # Parse the directive
            parts = directive_line.split(None, 1)
            if not parts:
                pos = line_end + 1
                continue
                
            directive_name = parts[0][1:]  # Remove the #
            
            if directive_name in ['ifdef', 'ifndef']:
                nest_level += 1
            elif directive_name == 'endif':
                nest_level -= 1
                if nest_level == 0:
                    # Found the matching #endif
                    return (hash_pos, line_end)
            
            pos = line_end + 1
        
        return -1  # No matching #endif found

    def handle_error(self, code, start, end, arguments):
        """
        Handle #error directive.
        This should cause compilation to stop with the specified error message.
        """
        error_message = arguments.strip().strip('"\'')
        self.syntax_error(f"#error: {error_message}")
        return code  # This line won't be reached due to the exception

    def handle_warning(self, code, start, end, arguments):
        """
        Handle #warning directive (if supported).
        This should issue a warning but continue compilation.
        """
        warning_message = arguments.strip().strip('"\'')
        print(f"WARNING: #warning: {warning_message}")
        # Remove the warning directive from the code
        return code[:start] + code[end+1:]

    def handle_pragma(self, code, start, end, arguments):
        """
        Handle #pragma directive.
        Pragmas are implementation-specific directives.
        This is a basic implementation that just removes the pragma.
        """
        pragma_args = arguments.strip()
        print(f"INFO: Ignoring pragma: {pragma_args}")
        # Remove the pragma directive from the code
        return code[:start] + code[end+1:]

    def handle_line(self, code, start, end, arguments):
        """
        Handle #line directive.
        This changes the line numbering for error reporting.
        Basic implementation that just removes the directive.
        """
        line_args = arguments.strip()
        print(f"INFO: Line directive ignored: {line_args}")