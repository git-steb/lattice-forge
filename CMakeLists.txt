cmake_minimum_required(VERSION 3.12)
project(closest_index_cpp)

# ---------------------------
# General Project Settings
# ---------------------------

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)

# Set architecture for macOS (if applicable)
if(APPLE)
    set(CMAKE_OSX_ARCHITECTURES "arm64")
endif()

# Set build type to Debug if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

# Ensure debugging information is included and optimizations are turned off for debug builds
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -ggdb")
set(CMAKE_SHARED_LINKER_FLAGS_DEBUG "-g -O0 -ggdb")

# Verbose output during compilation
set(CMAKE_VERBOSE_MAKEFILE ON)

# ---------------------------
# Dependencies
# ---------------------------

# Allow users to specify pybind11 and Eigen3 paths via environment variables or CMake options
find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)

# Check for pybind11 and Eigen3
if(NOT pybind11_FOUND)
    message(FATAL_ERROR "pybind11 not found. Please install pybind11 or set PYBIND11_ROOT.")
endif()

if(NOT Eigen3_FOUND)
    message(FATAL_ERROR "Eigen3 not found. Please install Eigen3 or set EIGEN3_ROOT.")
endif()

# ---------------------------
# pybind11 Module Target
# ---------------------------

# Define the pybind11 module target
pybind11_add_module(closest_index_cpp LatticeForge/closest_index_cpp.cpp)
target_link_libraries(closest_index_cpp PRIVATE Eigen3::Eigen)

# Add target for generating dSYM for debugging
add_custom_command(TARGET closest_index_cpp POST_BUILD
    COMMAND dsymutil $<TARGET_FILE:closest_index_cpp> -o $<TARGET_FILE:closest_index_cpp>.dSYM
    COMMENT "Generating dSYM bundle for debugging")

# ---------------------------
# Voronoi Visualization Setup
# ---------------------------

# Paths for the script and output files
set(SCRIPT_PATH "${CMAKE_SOURCE_DIR}/notebooks/voronoi_lattice_visualization.py")
set(NOTEBOOK_PATH "${CMAKE_SOURCE_DIR}/docs/voronoi_lattice_visualization.ipynb")
set(HTML_PATH "${CMAKE_SOURCE_DIR}/docs/index.html")
set(DOCS_DIR "${CMAKE_SOURCE_DIR}/docs")

# Ensure required tools are available
find_program(JUPYTEXT_EXECUTABLE jupytext)
find_program(JUPYTER_EXECUTABLE jupyter)

if(NOT JUPYTEXT_EXECUTABLE)
    message(FATAL_ERROR "jupytext not found. Please install jupytext with 'pip install jupytext'.")
endif()

if(NOT JUPYTER_EXECUTABLE)
    message(FATAL_ERROR "jupyter not found. Please install jupyter with 'pip install jupyter'.")
endif()

# Custom command to convert the Python script to a Jupyter notebook using jupytext
add_custom_command(
    OUTPUT ${NOTEBOOK_PATH}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${DOCS_DIR}
    COMMAND ${JUPYTEXT_EXECUTABLE} --to notebook --output ${NOTEBOOK_PATH} ${SCRIPT_PATH}
    DEPENDS ${SCRIPT_PATH}
    COMMENT "Converting ${SCRIPT_PATH} to Jupyter notebook using jupytext..."
)

# Custom command to execute the notebook and convert it to HTML using nbconvert
add_custom_command(
    OUTPUT ${HTML_PATH}
    COMMAND ${JUPYTER_EXECUTABLE} nbconvert --to html --execute --output ${HTML_PATH} ${NOTEBOOK_PATH}
    DEPENDS ${NOTEBOOK_PATH}
    COMMENT "Executing and converting ${NOTEBOOK_PATH} to HTML using nbconvert..."
)

# Custom target to build the HTML, ensuring the correct order of execution
add_custom_target(
    generate ALL
    DEPENDS ${HTML_PATH}
    COMMENT "Generating HTML from Python script via Jupyter notebook..."
)

# Clean target to remove generated files
add_custom_target(
    clean_generated
    COMMAND ${CMAKE_COMMAND} -E remove ${NOTEBOOK_PATH} ${HTML_PATH}
    COMMENT "Cleaning generated notebook and HTML files..."
)
