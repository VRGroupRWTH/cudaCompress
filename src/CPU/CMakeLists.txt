cmake_minimum_required(VERSION 3.18)

# Project Configuration
# =====================
project(CPU VERSION 1.0 LANGUAGES CXX CUDA)

# Dependencies
# ============
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

# Source files
# ============
## CPP
SET (${PROJECT_NAME}_SOURCE_FILES
  ArithmeticCPU.cpp
	EncodeCPU.cpp
	EncoderTestSuite.cpp
	GolombRiceCPU.cpp
	HistogramCPU.cpp
	HuffmanCPU.cpp
	HuffmanTableCPU.cpp
	main.cpp
	RBUCCPU.cpp
	RunLengthCPU.cpp
	tools/stb_image.c
	YCoCgCPU.cpp
)

## H
SET(${PROJECT_NAME}_HEADER_FILES
  ArithmeticCPU.h
	EncodeCPU.h
	EncoderTestSuite.h
	GolombRiceCPU.h
	HistogramCPU.h
	HuffmanCPU.h
	HuffmanTableCPU.h
	RBUCCPU.h
	RunLengthCPU.h
	tools/Entropy.h
	tools/stb_image.h
	YCoCgCPU.h
)

# Library (Target)
# ================
ADD_LIBRARY(${PROJECT_NAME} STATIC ${${PROJECT_NAME}_SOURCE_FILES})

# Target Configuration
# ====================
#SET (CMAKE_CXX_FLAGS -std=c++0x)
#add_definitions(-DUNICODE -D_UNICODE)
#ADD_DEFINITIONS(-D_CONSOLE)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_11 cuda_std_11)
target_compile_definitions(${PROJECT_NAME} PRIVATE UNICODE _UNICODE)
target_compile_definitions(${PROJECT_NAME} PRIVATE _CONSOLE NOMINMAX)

set(CPU_INCLUDE_DIRS
  ${PROJECT_SOURCE_DIR}/../../include
  ${PROJECT_SOURCE_DIR}/../../src
)

## section: include directory
#INCLUDE_DIRECTORIES(
#  ${CPU_INCLUDE_DIRS}
#  ${CUDA_TOOLKIT_INCLUDE}
#)
target_include_directories(${PROJECT_NAME} PUBLIC ${CPU_INCLUDE_DIRS} ${CUDA_TOOLKIT_INCLUDE})

#SET_SOURCE_FILES_PROPERTIES(${PROJECT_NAME}_HEADER_FILES PROPERTIES HEADER_FILE_ONLY TRUE)
LIST(APPEND ${PROJECT_NAME}_SOURCE_FILES ${${PROJECT_NAME}_HEADER_FILES})

## section: add dependency
# dependency determines overall build order.
#ADD_DEPENDENCIES(${PROJECT_NAME} 
#        ${CUDA_CUDART_LIBRARY}
#        ${BASE_DIR}/cudaCompress/build/lib/libcudaCompress.so
#)

## section: set link libraries
#TARGET_LINK_LIBRARIES(${PROJECT_NAME}
#  ${CUDA_CUDART_LIBRARY}
#)

install(FILES ${HFILES_PUBLIC}
  DESTINATION include
)

install(TARGETS CPU 
  DESTINATION lib
  EXPORT CPU-targets
)

