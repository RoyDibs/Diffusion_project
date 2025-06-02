#!/usr/bin/env python3
"""
Setup script that delegates to CMake for building the diffusion_env Python extension.

This approach leverages your existing CMake expertise and deal.II's robust
CMake integration rather than trying to replicate that configuration manually
in setuptools. Think of this as a bridge that lets Python's packaging system
use your familiar CMake build process.
"""

import os
import subprocess
import sys
from pathlib import Path
import shutil

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

class CMakeBuildExt(build_ext):
    """
    Custom build extension that delegates compilation to CMake.
    
    This class replaces setuptools' default C++ compilation with CMake,
    allowing us to use your proven deal.II build patterns while still
    creating a Python module that pip can install and manage.
    """
    
    def build_extension(self, ext):
        """Build the extension using CMake instead of setuptools."""
        
        # Ensure we have a clean build directory
        build_dir = Path(self.build_temp).resolve()
        source_dir = Path(__file__).parent.resolve()
        
        # Create build directory if it doesn't exist
        build_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"CMake build directory: {build_dir}")
        print(f"Source directory: {source_dir}")
        
        # Prepare CMake configuration arguments
        cmake_args = [
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE=Debug",  # Use Debug for better error messages
        ]
        
        # Pass along the DEAL_II_DIR if it's set in the environment
        if "DEAL_II_DIR" in os.environ:
            cmake_args.append(f"-DDEAL_II_DIR={os.environ['DEAL_II_DIR']}")
            print(f"Using DEAL_II_DIR: {os.environ['DEAL_II_DIR']}")
        
        # Check that CMake is available
        try:
            subprocess.run(["cmake", "--version"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError(
                "CMake is required but not found. Please install CMake:\n"
                "  - On Ubuntu/Debian: sudo apt-get install cmake\n"
                "  - On CentOS/RHEL: sudo yum install cmake\n"
                "  - Or download from: https://cmake.org/download/"
            )
        
        # Configure the project with CMake
        print("\n" + "="*50)
        print("Configuring project with CMake...")
        print("="*50)
        
        try:
            subprocess.run([
                "cmake",
                str(source_dir),  # Where CMakeLists.txt is located
                *cmake_args
            ], cwd=build_dir, check=True)
            
            print("CMake configuration completed successfully!")
            
        except subprocess.CalledProcessError as e:
            print(f"\nCMake configuration failed with return code: {e.returncode}")
            print("This usually means there's an issue with finding dependencies.")
            print("\nTroubleshooting steps:")
            print("1. Ensure DEAL_II_DIR is set: export DEAL_II_DIR=/path/to/dealii")
            print("2. Ensure pybind11 is installed: pip install pybind11")
            print("3. Check that all source files exist in the current directory")
            raise RuntimeError("CMake configuration failed") from e
        
        # Build the project with CMake
        print("\n" + "="*50)
        print("Building extension with CMake...")
        print("="*50)
        
        try:
            subprocess.run([
                "cmake", "--build", ".", 
                "--config", "Debug",
                "--parallel", "4"  # Use parallel build for speed
            ], cwd=build_dir, check=True)
            
            print("CMake build completed successfully!")
            
        except subprocess.CalledProcessError as e:
            print(f"\nCMake build failed with return code: {e.returncode}")
            print("This usually means there was a compilation or linking error.")
            print("Check the output above for specific error messages.")
            raise RuntimeError("CMake build failed") from e
        
        # Find the built extension module
        # CMake creates the .so file with a specific name pattern
        extension_pattern = f"diffusion_env*.so"
        built_extensions = list(build_dir.glob(extension_pattern))
        
        if not built_extensions:
            raise RuntimeError(
                f"CMake build completed but no extension module found.\n"
                f"Expected to find {extension_pattern} in {build_dir}\n"
                f"Available files: {list(build_dir.glob('*'))}"
            )
        
        # Use the first matching extension (there should only be one)
        built_extension = built_extensions[0]
        print(f"Found built extension: {built_extension}")
        
        # Copy the extension to where setuptools expects it
        dest_path = Path(self.get_ext_fullpath(ext.name))
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Copying extension to: {dest_path}")
        shutil.copy2(built_extension, dest_path)
        
        print("\n" + "="*50)
        print("Extension build completed successfully!")
        print("="*50)

# Create a minimal extension placeholder
# The real extension is built by CMake, but setuptools needs this for its bookkeeping
ext_modules = [
    Extension(
        'diffusion_env',
        sources=[],  # Sources are handled by CMake
    )
]

# Main setup configuration
setup(
    name="diffusion_env",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="deal.II-based diffusion environment for reinforcement learning",
    long_description="""
        A high-performance finite element environment for solving transient diffusion
        equations, built using deal.II and wrapped for Python using pybind11.
        
        This package uses CMake for building, leveraging deal.II's robust build
        system integration while providing a clean Python interface for
        reinforcement learning applications.
    """,
    ext_modules=ext_modules,
    cmdclass={'build_ext': CMakeBuildExt},
    zip_safe=False,
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.15.0",
        "pybind11>=2.6.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
    ],
)

if __name__ == "__main__":
    print("Building diffusion_env using CMake integration")
    print("This build process uses your familiar deal.II CMake patterns")
    print("while creating a Python extension module for RL applications.")
    print("")