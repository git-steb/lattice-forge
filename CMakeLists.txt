cmake_minimum_required(VERSION 3.12)
project(closest_index_cpp)

set(CMAKE_CXX_STANDARD 14)

# Ensure debugging information is included and optimizations are turned off for debug builds
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0 -ggdb")
set(CMAKE_SHARED_LINKER_FLAGS_DEBUG "${CMAKE_SHARED_LINKER_FLAGS_DEBUG} -g -O0 -ggdb")
set(CMAKE_OSX_ARCHITECTURES "arm64")

# Set build type to Debug if not specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

# Add pybind11 directory to CMAKE_PREFIX_PATH
set(PYBIND11_DIR "/Users/sbergner/micromamba/lib/python3.11/site-packages/pybind11/share/cmake/pybind11")
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${PYBIND11_DIR})

find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Verbose output during compilation
set(CMAKE_VERBOSE_MAKEFILE ON)

# Adding debug info flags directly
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -ggdb")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -g -O0 -ggdb")

# Define the target
pybind11_add_module(closest_index_cpp LatticeForge/closest_index_cpp.cpp)

target_link_libraries(closest_index_cpp PRIVATE Eigen3::Eigen)

# Add target for generating dSYM
add_custom_command(TARGET closest_index_cpp POST_BUILD
    COMMAND dsymutil $<TARGET_FILE:closest_index_cpp> -o $<TARGET_FILE:closest_index_cpp>.dSYM
    COMMENT "Generating dSYM bundle for debugging")
