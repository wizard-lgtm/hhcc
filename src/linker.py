"""
linker.py - Handles linking of object files and external libraries for the Holy HolyC Compiler.
Uses LLVM tools for compilation and linking.
"""

from typing import List, Dict, Optional
import os
import subprocess
import platform
import sys
import shutil

class Linker:
    """
    Handles linking of object files and external libraries for the Holy HolyC Compiler.
    This class manages the creation of executable files from LLVM IR through LLVM tools.
    """
    def __init__(self, compiler):
        self.compiler = compiler
        self.object_files = []
        self.external_libs = []
        self.lib_paths = []
        self.output_file = compiler.output_file or "a.out"
        if platform.system() == "Windows" and not self.output_file.endswith(".exe"):
            self.output_file += ".exe"
            
        # Find LLVM tools
        self.find_llvm_tools()
        
        if self.compiler.debug:
            print(f"Linker initialized with output file: {self.output_file}")
            
    def find_llvm_tools(self):
        """Find and configure paths to LLVM tools."""
        # Try to find llvm-config first to get the paths
        llvm_config = self._find_executable("llvm-config")
        
        if llvm_config:
            try:
                # Get LLVM binary directory
                cmd = [llvm_config, "--bindir"]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    bin_dir = result.stdout.strip()
                    self.llvm_bindir = bin_dir
                    
                    # Get LLVM version
                    cmd = [llvm_config, "--version"]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode == 0:
                        self.llvm_version = result.stdout.strip()
                        if self.compiler.debug:
                            print(f"Found LLVM version: {self.llvm_version}")
            except Exception as e:
                print(f"Error getting LLVM configuration: {e}")
                self.llvm_bindir = None
        else:
            self.llvm_bindir = None
            
        # Find essential LLVM tools
        self.llc = self._find_executable("llc")
        self.llvm_link = self._find_executable("llvm-link")
        
        # Prefer clang for linking if available, fall back to gcc or link.exe
        self.clang = self._find_executable("clang")
        
        if not self.clang:
            if platform.system() == "Windows":
                self.linker = self._find_executable("link")
            else:
                self.linker = self._find_executable("gcc")
        else:
            self.linker = self.clang
            
        if self.compiler.debug:
            print(f"Using llc: {self.llc}")
            print(f"Using llvm-link: {self.llvm_link}")
            print(f"Using linker: {self.linker}")
            
        # Verify we have the necessary tools
        if not self.llc:
            print("WARNING: llc not found. Cannot compile LLVM IR to object files.")
        if not self.llvm_link:
            print("WARNING: llvm-link not found. Cannot link multiple LLVM IR files.")
        if not self.linker:
            print("ERROR: No suitable linker found (clang, gcc, or link.exe).")
            sys.exit(1)
            
    def _find_executable(self, name):
        """Find an executable in PATH or LLVM bin directory."""
        # First try the LLVM bin directory if we know it
        if hasattr(self, 'llvm_bindir') and self.llvm_bindir:
            if platform.system() == "Windows":
                exec_path = os.path.join(self.llvm_bindir, f"{name}.exe")
            else:
                exec_path = os.path.join(self.llvm_bindir, name)
                
            if os.path.exists(exec_path):
                return exec_path
                
        # Try to find it in PATH
        return shutil.which(name)
    
    def add_object_file(self, obj_file: str):
        """Add an object file to be linked."""
        if os.path.exists(obj_file):
            self.object_files.append(obj_file)
        else:
            raise FileNotFoundError(f"Object file not found: {obj_file}")
    
    def add_library(self, lib_name: str):
        """Add an external library to link against."""
        self.external_libs.append(lib_name)
    
    def add_library_path(self, lib_path: str):
        """Add a library search path."""
        if os.path.exists(lib_path) and os.path.isdir(lib_path):
            self.lib_paths.append(lib_path)
        else:
            raise FileNotFoundError(f"Library path not found: {lib_path}")
    
    def compile_ir_to_object(self, ir_file: str) -> str:
        """Compile LLVM IR to an object file using llc."""
        if not self.llc:
            raise RuntimeError("llc not found. Cannot compile LLVM IR to object files.")
            
        obj_file = os.path.splitext(ir_file)[0] + ".o"
        
        # Use llc to compile LLVM IR to an object file
        cmd = [self.llc, "-filetype=obj", "-relocation-model=pic", ir_file, "-o", obj_file]

        if self.compiler.target and self.compiler.target.triple:
            cmd.extend(["-mtriple", self.compiler.target.triple])

        
        if self.compiler.target and self.compiler.target.triple:
            cmd.extend(["-mtriple", self.compiler.target.triple])
        
        if self.compiler.debug:
            print(f"Executing command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error compiling IR to object: {result.stderr}")
            raise RuntimeError(f"Failed to compile IR file {ir_file} to object: {result.stderr}")
        
        self.add_object_file(obj_file)
        return obj_file
    
    def link_ir_files(self, ir_files: List[str]) -> str:
        """Link multiple LLVM IR files into one using llvm-link."""
        if not self.llvm_link:
            raise RuntimeError("llvm-link not found. Cannot link LLVM IR files.")
            
        output_ir = os.path.splitext(self.output_file)[0] + ".linked.ll"
        
        cmd = [self.llvm_link, "-o", output_ir]
        cmd.extend(ir_files)
        
        if self.compiler.debug:
            print(f"Executing command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error linking IR files: {result.stderr}")
            raise RuntimeError(f"Failed to link IR files: {result.stderr}")
        
        return output_ir
    
    def link(self) -> str:
        """
        Link all object files and libraries to create an executable.
        Returns the path to the created executable.
        """
        if not self.object_files:
            raise ValueError("No object files to link")
        
        # If using clang as linker
        if self.linker == self.clang:
            cmd = [self.linker]

            cmd.extend(["-fno-pie", "-no-pie"])  

            
            # Add object files
            cmd.extend(self.object_files)
            
            # Add library paths
            for path in self.lib_paths:
                cmd.extend(["-L", path])
                
            # Add libraries
            for lib in self.external_libs:
                cmd.extend(["-l", lib])
                
            # Set output file
            cmd.extend(["-o", self.output_file])
            
            # Add target triple if available
            if self.compiler.target and self.compiler.target.triple:
                cmd.extend([f"--target={self.compiler.target.triple}"])
        
        # If using gcc
        elif "gcc" in os.path.basename(self.linker):
            cmd = [self.linker]
            
            # Add object files
            cmd.extend(self.object_files)
            
            # Add library paths
            for path in self.lib_paths:
                cmd.extend(["-L", path])
                
            # Add libraries
            for lib in self.external_libs:
                cmd.extend(["-l", lib])
                
            # Set output file
            cmd.extend(["-o", self.output_file])
        
        # If using MSVC link.exe
        elif "link" in os.path.basename(self.linker).lower():
            cmd = [self.linker]
            
            # Add object files
            cmd.extend(self.object_files)
            
            # Add library paths
            for path in self.lib_paths:
                cmd.append(f"/LIBPATH:{path}")
            
            # Add libraries
            for lib in self.external_libs:
                cmd.append(f"{lib}.lib")
            
            # Set output file
            cmd.append(f"/OUT:{self.output_file}")
        
        else:
            # Generic case - assume gcc-like syntax
            cmd = [self.linker]
            cmd.extend(self.object_files)
            
            for path in self.lib_paths:
                cmd.extend(["-L", path])
                
            for lib in self.external_libs:
                cmd.extend(["-l", lib])
                
            cmd.extend(["-o", self.output_file])
        
        if self.compiler.debug:
            print(f"Executing linker command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Linking error: {result.stderr}")
            raise RuntimeError(f"Failed to link objects: {result.stderr}")
        
        print(f"Successfully linked executable: {self.output_file}")
        return self.output_file
        
    def link_with_clang(self, ir_file: str) -> str:
        """
        Use clang to directly compile and link LLVM IR to an executable.
        This is an alternative to the llc + linker approach.
        """
        if not self.clang:
            raise RuntimeError("clang not found. Cannot link with clang.")
            
        cmd = [self.clang]
        
        # Add IR file
        cmd.append(ir_file)
        
        # Add library paths
        for path in self.lib_paths:
            cmd.extend(["-L", path])
            
        # Add libraries
        for lib in self.external_libs:
            cmd.extend(["-l", lib])
            
        # Set output file
        cmd.extend(["-o", self.output_file])
        
        # Add target triple if available
        if self.compiler.target and self.compiler.target.triple:
            cmd.extend([f"--target={self.compiler.target.triple}"])
        
        if self.compiler.debug:
            print(f"Executing clang link command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Clang linking error: {result.stderr}")
            raise RuntimeError(f"Failed to link with clang: {result.stderr}")
        
        print(f"Successfully linked executable with clang: {self.output_file}")
        return self.output_file