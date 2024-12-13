cmake_minimum_required(VERSION 3.12)
project(closest_index_cpp)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)

# Ensure debugging information is included and optimizations are turned off for debug builds
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0 -ggdb")
set(CMAKE_SHARED_LINKER_FLAGS_DEBUG "${CMAKE_SHARED_LINKER_FLAGS_DEBUG} -g -O0 -ggdb")

# Set architecture for macOS
set(CMAKE_OSX_ARCHITECTURES "arm64")

# Set build type to Debug if not specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

# Add pybind11 and Eigen3 to the CMake prefix path
set(PYBIND11_DIR "/Users/sbergner/micromamba/lib/python3.11/site-packages/pybind11/share/cmake/pybind11")
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${PYBIND11_DIR})

find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Verbose output during compilation
set(CMAKE_VERBOSE_MAKEFILE ON)

# Adding debug info flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -ggdb")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -g -O0 -ggdb")

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

# Set the paths for the script and output files
set(SCRIPT_PATH "${CMAKE_SOURCE_DIR}/notebooks/voronoi_lattice_visualization.py")
set(NOTEBOOK_PATH "${CMAKE_SOURCE_DIR}/docs/voronoi_lattice_visualization.ipynb")
set(HTML_PATH "${CMAKE_SOURCE_DIR}/docs/index.html")
set(DOCS_DIR "${CMAKE_SOURCE_DIR}/docs")

# Custom command to convert the Python script to a Jupyter notebook using jupytext
add_custom_command(
    OUTPUT ${NOTEBOOK_PATH}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${DOCS_DIR}
    COMMAND jupytext --to notebook --output ${NOTEBOOK_PATH} ${SCRIPT_PATH}
    DEPENDS ${SCRIPT_PATH}
    COMMENT "Converting ${SCRIPT_PATH} to Jupyter notebook using jupytext..."
)

# Custom command to convert the Jupyter notebook to HTML using nbconvert
add_custom_command(
    OUTPUT ${HTML_PATH}
    COMMAND jupyter nbconvert --to html --output ${HTML_PATH} ${NOTEBOOK_PATH}
    DEPENDS ${NOTEBOOK_PATH}
    COMMENT "Converting ${NOTEBOOK_PATH} to HTML using nbconvert..."
)

# Custom target to build the HTML
add_custom_target(
    generate ALL
    DEPENDS ${HTML_PATH}
)

# Custom target to clean generated files
add_custom_target(
    clean_generated
    COMMAND ${CMAKE_COMMAND} -E remove ${NOTEBOOK_PATH} ${HTML_PATH}
    COMMENT "Cleaning generated files..."
)
