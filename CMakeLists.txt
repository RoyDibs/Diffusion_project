# CMakeLists.txt - Compatibility-aware approach for deal.II + pybind11
# This version solves the CMake signature compatibility issue by separating
# the deal.II configuration domain from the pybind11 configuration domain

cmake_minimum_required(VERSION 3.12.0)

# Find deal.II using your proven, familiar approach
# This leverages all the same configuration mechanisms you use successfully
find_package(deal.II 9.0.0 REQUIRED
  HINTS ${DEAL_II_DIR} /home/droysar1/dealii-install)

if(${deal.II_FOUND})
  message(STATUS "SUCCESS: Found deal.II at ${DEAL_II_DIR}")
  message(STATUS "deal.II version: ${DEAL_II_PACKAGE_VERSION}")
  message(STATUS "deal.II build type: ${DEAL_II_BUILD_TYPE}")
else()
  message(FATAL_ERROR "FAILED: Could not find deal.II")
endif()

# Find Python using standard CMake discovery mechanisms
find_package(Python COMPONENTS Interpreter Development REQUIRED)
message(STATUS "Found Python: ${Python_EXECUTABLE}")
message(STATUS "Python version: ${Python_VERSION}")

# Discover pybind11 using Python's knowledge of where it installed itself
# This approach works reliably with pip-installed pybind11 in virtual environments
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
  OUTPUT_VARIABLE pybind11_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE pybind11_RESULT
)

# Verify that pybind11 discovery succeeded
if(NOT pybind11_RESULT EQUAL 0)
  message(FATAL_ERROR "Failed to find pybind11. Make sure it's installed in your Python environment:
    pip install pybind11
    
    Current Python executable: ${Python_EXECUTABLE}")
endif()

message(STATUS "Found pybind11 at: ${pybind11_DIR}")

# Load pybind11's CMake configuration using the discovered location
find_package(pybind11 REQUIRED PATHS ${pybind11_DIR} NO_DEFAULT_PATH)

# Define your project name
PROJECT(diffusion_env)

# Verify that our source files exist before attempting compilation
# This provides clear error messages if files are missing or misnamed
set(DEALII_SOURCE_FILES
  DiffusionEnvironment.cc
)

set(PYBIND11_SOURCE_FILES
  python_wrapper.cc
)

# Check deal.II source files
foreach(SOURCE_FILE ${DEALII_SOURCE_FILES})
  if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE}")
    message(FATAL_ERROR "deal.II source file not found: ${SOURCE_FILE}
    Make sure all source files are present in: ${CMAKE_CURRENT_SOURCE_DIR}")
  endif()
endforeach()

# Check pybind11 wrapper files
foreach(SOURCE_FILE ${PYBIND11_SOURCE_FILES})
  if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE}")
    message(FATAL_ERROR "pybind11 wrapper file not found: ${SOURCE_FILE}
    Make sure all source files are present in: ${CMAKE_CURRENT_SOURCE_DIR}")
  endif()
endforeach()

message(STATUS "All source files found")

# ARCHITECTURAL STRATEGY: Create two separate targets to avoid CMake signature conflicts
# 
# Target 1: A static library containing the deal.II computational core
# This target uses deal.II's traditional CMake configuration approach
# and is completely independent of Python/pybind11 concerns
add_library(diffusion_core STATIC
  ${DEALII_SOURCE_FILES}
)

# Apply deal.II configuration to the core library using your familiar approach
# This macro works exactly the same as in your regular deal.II programs
# because it's operating on a clean target with no previous CMake signature usage
DEAL_II_SETUP_TARGET(diffusion_core DEBUG)

# Set C++17 standard explicitly for the core library
# deal.II requires this, and explicit specification prevents subtle issues
set_target_properties(diffusion_core PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  POSITION_INDEPENDENT_CODE ON  # Required for linking into shared libraries
)

# Target 2: A Python extension module that wraps the computational core
# This target uses pybind11's modern CMake configuration approach
# and focuses purely on the Python interface concerns
pybind11_add_module(diffusion_env
  ${PYBIND11_SOURCE_FILES}
)

# Connect the Python wrapper to the deal.II computational core
# This uses modern CMake target-based linking that pybind11 expects
# PRIVATE means the core library is needed for building diffusion_env
# but not for anything that might link against diffusion_env
target_link_libraries(diffusion_env PRIVATE diffusion_core)

# Configure the Python extension for your HPC environment
# These settings ensure the extension works correctly in cluster environments
set_target_properties(diffusion_env PROPERTIES
  CXX_STANDARD 17                    # Match the core library's C++ standard
  CXX_STANDARD_REQUIRED ON           # Make C++17 compliance mandatory
)

# Add compile definitions that help in HPC environments
# These flags improve compatibility with cluster software environments
target_compile_definitions(diffusion_env PRIVATE
  -DWITH_MPI=OFF                     # Usually safer for Python extensions
)

# Provide comprehensive build summary for debugging and verification
message(STATUS "")
message(STATUS "=== Build Architecture Summary ===")
message(STATUS "Target 1 (diffusion_core):")
message(STATUS "  Type: Static library")
message(STATUS "  Purpose: deal.II computational engine")
message(STATUS "  Configuration: DEAL_II_SETUP_TARGET")
message(STATUS "  Sources: ${DEALII_SOURCE_FILES}")
message(STATUS "")
message(STATUS "Target 2 (diffusion_env):")
message(STATUS "  Type: Python extension module")
message(STATUS "  Purpose: Python interface wrapper")
message(STATUS "  Configuration: pybind11_add_module")
message(STATUS "  Sources: ${PYBIND11_SOURCE_FILES}")
message(STATUS "")
message(STATUS "=== Configuration Details ===")
message(STATUS "  deal.II: ${DEAL_II_PACKAGE_VERSION} at ${DEAL_II_DIR}")
message(STATUS "  Python: ${Python_VERSION} at ${Python_EXECUTABLE}")
message(STATUS "  pybind11: Found via pip installation")
message(STATUS "  Architecture: Separated configuration domains")
message(STATUS "=== End Summary ===")
message(STATUS "")