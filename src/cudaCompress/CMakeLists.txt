PROJECT(cudaCompress)

cmake_minimum_required(VERSION 2.8)

find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)


## section: Macro
#MACRO(ADD_MSVC_PRECOMPILED_HEADER PrecompiledHeader PrecompiledSource SourcesVar)
#  IF(MSVC)
#    GET_FILENAME_COMPONENT(PrecompiledBasename ${PrecompiledHeader} NAME_WE)
#    SET(PrecompiledBinary "${CMAKE_CURRENT_BINARY_DIR}/${PrecompiledBasename}.pch")
#    SET(Sources ${${SourcesVar}})
#
#    SET_SOURCE_FILES_PROPERTIES(${PrecompiledSource}
#                                PROPERTIES COMPILE_FLAGS "/Yc\"${PrecompiledHeader}\" /Fp\"${PrecompiledBinary}\""
#                                           OBJECT_OUTPUTS "${PrecompiledBinary}")
#    SET_SOURCE_FILES_PROPERTIES(${Sources}
#                                PROPERTIES COMPILE_FLAGS "/Yu\"${PrecompiledBinary}\" /FI\"${PrecompiledBinary}\" /Fp\"${PrecompiledBinary}\""
#                                           OBJECT_DEPENDS "${PrecompiledBinary}")  
#    # Add precompiled header to SourcesVar
#    LIST(APPEND ${SourcesVar} ${PrecompiledSource})
#  ENDIF(MSVC)
#ENDMACRO(ADD_MSVC_PRECOMPILED_HEADER)

option(CUDA_VERBOSE_PTXAS
  "On to enable verbose output from the PTXAS assembler."
  OFF
  )

SET (CMAKE_CXX_COMPILER g++)
SET (CMAKE_CXX_FLAGS "-std=c++0x -O3 -ffast-math -Wall")
LIST(APPEND CUDA_NVCC_FLAGS --compiler-options -fno-strict-aliasing -lineinfo -use_fast_math -Xptxas -dlcm=cg)
SET (CUDA_PROPAGATE_HOST_FLAGS OFF)

## start setting
SET (CUDA_COMPRESS_LIB cudaCompress)
SET (CUDA_COMPRESS_KERNEL_LIB cudaCompressKernel)
SET(PROJECT_NAME cudaCompress)

add_definitions(-DUNICODE -D_UNICODE)

set(CUDA_COMPRESS_INCLUDE_DIRS
  ${PROJECT_SOURCE_DIR}/../../include
  ${PROJECT_SOURCE_DIR}/../../src
)


## section: include directory
INCLUDE_DIRECTORIES(
  $(SolutionDir)include
	$(SolutionDir)src
        ${CUDA_COMPRESS_INCLUDE_DIRS}
        ${CUDA_TOOLKIT_INCLUDE}
  )

if (CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(bits 64)
else ()
  set(bits 32)
endif (CMAKE_SIZEOF_VOID_P EQUAL 8)

#string(TOUPPER "${CMAKE_CONFIGURATION_TYPES}" configs)
#foreach (config ${configs})
#  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${config} "${EXECUTABLE_OUTPUT_PATH}")
#  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${config} "${LIBRARY_OUTPUT_PATH}")
#  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${config} "${LIBRARY_OUTPUT_PATH}")
#  set(CMAKE_${config}_POSTFIX ${bits})
#endforeach(config {configs})
set(CMAKE_DEBUG_POSTFIX "${bits}d")

## section: source files
# Add your source files here (one file per line), please SORT in alphabetical order for future maintenance
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

### section: header files
## Add your header files here(one file per line), please SORT in alphabetical order for future maintenance!
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

set (CUHFILES
	BitStreamGPU.cuh
	util/Quantize.cuh
	util/DWTFloatFilters.cuh
	util/DWTCommon.cuh
  )

set (${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
    ${CMAKE_SOURCE_DIR}/lib)

source_group("CUDA Source Files" FILES ${CUFILES})
source_group("CUDA Header Files" FILES ${CUHFILES})

set(GENCODE_SM20 -gencode=arch=compute_20,code=sm_20 -gencode=arch=compute_20,code=compute_20)
set(GENCODE_SM30 -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_30,code=compute_30)
set(GENCODE_SM35 -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_35,code=compute_35)

option(CUDPP_GENCODE_SM20
       "ON to generate code for Compute Capability 2.0 devices (e.g. Tesla C2050)"
       ON)

option(CUDPP_GENCODE_SM30
       "ON to generate code for Compute Capability 3.0 devices (e.g. Tesla K10)"
       OFF)

option(CUDPP_GENCODE_SM35
       "ON to generate code for Compute Capability 3.5 devices (e.g. Tesla K20)"
       OFF)

if (CUDPP_GENCODE_SM20)
  set(GENCODE ${GENCODE} ${GENCODE_SM20})
endif(CUDPP_GENCODE_SM20)

if (CUDPP_GENCODE_SM30)
  set(GENCODE ${GENCODE} ${GENCODE_SM30})
endif(CUDPP_GENCODE_SM30)

if (CUDPP_GENCODE_SM35)
  set(GENCODE ${GENCODE} ${GENCODE_SM35})
endif(CUDPP_GENCODE_SM35)

if (CUDA_VERBOSE_PTXAS)
  set(VERBOSE_PTXAS --ptxas-options=-v)
endif (CUDA_VERBOSE_PTXAS)

#cuda_compile( ${CUDA_OS} ${CUFILES} OPTIONS ${GENCODE} ${VERBOSE_PTXAS}
#)

cuda_add_library(${CUDA_COMPRESS_LIB} SHARED 
   ${COMPRESS_SOURCE_FILES}
   ${COMPRESS_HEADER_FILES}
   ${CUHFILES}
   ${CUFILES}
  OPTIONS ${GENCODE} ${VERBOSE_PTXAS}
  )

## section: precompiled header	
#ADD_MSVC_PRECOMPILED_HEADER("precompiled.h" "precompiled.cpp" MySources)
#ADD_LIBRARY(MyLibrary ${MySources})

SET_SOURCE_FILES_PROPERTIES(${CUDA_COMPRESS_LIB}_HEADER_FILES
                            PROPERTIES HEADER_FILE_ONLY TRUE)
#LIST(APPEND ${CUDA_COMPRESS_LIB}_SOURCE_FILES ${${CUDA_COMPRESS_LIB}_HEADER_FILES})

## section: add definitions
#   add prefix -D. example> -DSHP
#  - DO NOT add  the following definitions(already defined in ${OSP_DEFINITIONS}:
#   -DSHP, -DWIN32, -D_WINDOWS, -D_DEBUG, -D_USRDLL, -D_CRT_SECURE_NO_DEPRECATE
ADD_DEFINITIONS(
    -D_CONSOLE
	-DNOMINMAX
    )

## section: add target

#ADD_LIBRARY(${CUDA_COMPRESS_LIB} SHARED ${${CUDA_COMPRESS_LIB}_SOURCE_FILES} ${CUDO_OS} )

## section: add dependency
# dependency determines overall build order.
ADD_DEPENDENCIES(${CUDA_COMPRESS_LIB}
        ${CUDA_CUDART_LIBRARY}
	)
#ADD_DEPENDENCIES(${CUDA_COMPRESS_KERNEL_LIB}
#        ${CUDA_CUDART_LIBRARY}
#       )
#

## section: set link libraries
TARGET_LINK_LIBRARIES( ${CUDA_COMPRESS_LIB}
        ${CUDA_CUDART_LIBRARY}
        )
#TARGET_LINK_LIBRARIES( ${CUDA_COMPRESS_KERNEL_LIB}
#        ${CUDA_CUDART_LIBRARY}
#        )
#

install(FILES ${HFILES_PUBLIC}
  DESTINATION include
  )

install(TARGETS cudaCompress 
  DESTINATION lib
  EXPORT cudaCompress-targets
  )

#install(TARGETS cudaCompressKernel
#  DESTINATION lib
#  EXPORT cudaCompress-targets
#  )

