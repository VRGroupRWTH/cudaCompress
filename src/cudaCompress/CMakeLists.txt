cmake_minimum_required(VERSION 3.18)

# Project Configuration
# =====================
PROJECT(cudaCompress VERSION 1.0 LANGUAGES CXX CUDA)

SET (CUDA_COMPRESS_LIB ${PROJECT_NAME})
SET (CUDA_COMPRESS_KERNEL_LIB ${PROJECT_NAME}Kernel)

option(CUDA_VERBOSE_PTXAS
  "On to enable verbose output from the PTXAS assembler."
  OFF
)

# Dependencies
# ============
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

# Source files
# ============
## CPP
SET (COMPRESS_SOURCE_FILES
	Timing.cpp
	scan/scan_plan.cpp
	InstanceImpl.cpp
	reduce/reduce_plan.cpp
	profiler/profilerlogwriter.cpp
	profiler/profilerstreamwriter.cpp
	profiler/profiler.cpp
	util/CudaTimer.cpp
	Encode.cpp
	Instance.cpp
)

## H
SET(COMPRESS_HEADER_FILES
	scan/scan_plan.h
	scan/scan_globals.h
	tools/Operator.h
	tools/Functor.h
	InstanceImpl.h
	cudaUtil.h
	RunLength.h
	PackInc.h
	HuffmanTable.h
	reduce/reduce_plan.h
	reduce/reduce_globals.h
	Histogram.h
	profiler/profilerlogwriter.hpp
	profiler/profilerstreamwriter.hpp
	profiler/profiler.hpp
	HuffmanDesign.h
	util.h
	Huffman.h
	../../include/cudaCompress/Instance.h
	../../include/cudaCompress/BitStream.h
	../../include/cudaCompress/Encode.h
	../../include/cudaCompress/EncodeCommon.h
	../../include/cudaCompress/Timing.h
	../../include/cudaCompress/global.h
	../../include/cudaCompress/util/Quantize.h
	../../include/cudaCompress/util/DWT.h
	../../include/cudaCompress/util/YCoCg.h
	../../include/cudaCompress/util/Bits.h
	../../include/cudaCompress/util/CudaTimer.h
)

# CU
set (CUFILES
	Huffman.cu
    HuffmanTable.cu
	RunLength.cu
	PackInc.cu
	Histogram.cu
	util/DWTInt.cu
	util/YCoCg.cu
	util/Bits.cu
	util/DWTFloat.cu
	util/Quantize.cu
)

 # CUH
set (CUHFILES
	BitStreamGPU.cuh
	util/Quantize.cuh
	util/DWTFloatFilters.cuh
	util/DWTCommon.cuh
)

# Library (Target)
# ================
add_library(${PROJECT_NAME} STATIC 
	${COMPRESS_SOURCE_FILES}
	${COMPRESS_HEADER_FILES}
	${CUHFILES}
	${CUFILES}
)

# Target Configuration
# ====================
#target_enable_fast_math(${PROJECT_NAME} True)
#target_set_max_warning(${PROJECT_NAME})
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_11 cuda_std_11)
target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
#	-fno-strict-aliasing
	-lineinfo
#	-use_fast_math
#	-Xptxas
#	-dlcm=cg
>)

target_compile_definitions(${PROJECT_NAME} PRIVATE UNICODE _UNICODE)
target_compile_definitions(${PROJECT_NAME} PRIVATE _CONSOLE NOMINMAX)

set(CUDA_COMPRESS_INCLUDE_DIRS
  ${PROJECT_SOURCE_DIR}/../../include
  ${PROJECT_SOURCE_DIR}/../../src
)

## section: include directory
target_include_directories(${PROJECT_NAME} PUBLIC ${CUDA_COMPRESS_INCLUDE_DIRS})

if (CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(bits 64)
else ()
  set(bits 32)
endif ()

set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "${bits}d")
set_target_properties(${PROJECT_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)

source_group("CUDA Source Files" FILES ${CUFILES})
source_group("CUDA Header Files" FILES ${CUHFILES})

target_link_libraries (${PROJECT_NAME}
	PUBLIC CUDA::cudart_static
)

#install(FILES ${COMPRESS_HEADER_FILES} ${CUHFILES}
#  DESTINATION include
#)

#install(TARGETS ${PROJECT_NAME} 
#  DESTINATION lib
#  EXPORT ${PROJECT_NAME}-targets
#)

#install(EXPORT  ${PROJECT_NAME}-config DESTINATION cmake)
#export (TARGETS ${PROJECT_NAME}       FILE        ${PROJECT_NAME}-config.cmake)
