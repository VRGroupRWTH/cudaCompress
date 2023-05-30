cmake_minimum_required(VERSION 3.18)

# Project Config
# ==============
PROJECT(cudaCompress VERSION 1.0 LANGUAGES CXX CUDA)

## Custom CMake Modules
list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(FastMath)
include(MaxErrorWarning)

## Enforce C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
## Use certain optimization level
#set(CMAKE_CXX_FLAGS_DEBUG -g)
#set(CMAKE_CXX_FLAGS_RELEASE -O3)

## Set the directory where the binaries will be stored
set(EXECUTABLE_OUTPUT_PATH
  ${PROJECT_BINARY_DIR}/bin
  CACHE PATH
  "Directory where all executables will be stored"
)

## Set the directory where the libraries will be stored
set(LIBRARY_OUTPUT_PATH
  ${PROJECT_BINARY_DIR}/lib
  CACHE PATH
  "Directory where all the libraries will be stored"
)

set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/build)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})

set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

set(PROJECT_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)

#include_directories(${PROJECT_INCLUDE_DIR})
#include_directories(${PROJECT_SOURCE_DIR})

# Dependencies
# ============
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

# Subprojects
# ============
add_subdirectory(extern/glm)
add_subdirectory(src/cudaCompress)
add_subdirectory(src/CPU)
add_subdirectory(src/examples)
add_subdirectory(src/ParticleTracer)
