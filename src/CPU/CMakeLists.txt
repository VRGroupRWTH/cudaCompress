PROJECT(CPU)

cmake_minimum_required(VERSION 2.8)

find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)

## section: Macro
MACRO(ADD_MSVC_PRECOMPILED_HEADER PrecompiledHeader PrecompiledSource SourcesVar)
  IF(MSVC)
    GET_FILENAME_COMPONENT(PrecompiledBasename ${PrecompiledHeader} NAME_WE)
    SET(PrecompiledBinary "${CMAKE_CURRENT_BINARY_DIR}/${PrecompiledBasename}.pch")
    SET(Sources ${${SourcesVar}})

    SET_SOURCE_FILES_PROPERTIES(${PrecompiledSource}
                                PROPERTIES COMPILE_FLAGS "/Yc\"${PrecompiledHeader}\" /Fp\"${PrecompiledBinary}\""
                                           OBJECT_OUTPUTS "${PrecompiledBinary}")
    SET_SOURCE_FILES_PROPERTIES(${Sources}
                                PROPERTIES COMPILE_FLAGS "/Yu\"${PrecompiledBinary}\" /FI\"${PrecompiledBinary}\" /Fp\"${PrecompiledBinary}\""
                                           OBJECT_DEPENDS "${PrecompiledBinary}")  
    # Add precompiled header to SourcesVar
    LIST(APPEND ${SourcesVar} ${PrecompiledSource})
  ENDIF(MSVC)
ENDMACRO(ADD_MSVC_PRECOMPILED_HEADER)

option(CUDA_VERBOSE_PTXAS
  "On to enable verbose output from the PTXAS assembler."
  OFF
  )

SET (CMAKE_CXX_FLAGS -std=c++0x)

## start setting
SET (this_target CPU)
SET(PROJECT_NAME CPU)

add_definitions(-DUNICODE -D_UNICODE)

set(CPU_INCLUDE_DIRS
  ${PROJECT_SOURCE_DIR}/../../include
  ${PROJECT_SOURCE_DIR}/../../src
)

## section: include directory
INCLUDE_DIRECTORIES(
  $(SolutionDir)include
        $(SolutionDir)src
        ${CPU_INCLUDE_DIRS}
        ${CUDA_TOOLKIT_INCLUDE}
  )

## section: source files
# Add your source files here (one file per line), please SORT in alphabetical order for future maintenance
SET (${this_target}_SOURCE_FILES
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

## section: header files
# Add your header files here(one file per line), please SORT in alphabetical order for future maintenance!
SET(${this_target}_HEADER_FILES
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



## section: precompiled header	
#ADD_MSVC_PRECOMPILED_HEADER("precompiled.h" "precompiled.cpp" MySources)
#ADD_LIBRARY(MyLibrary ${MySources})

SET_SOURCE_FILES_PROPERTIES(${this_target}_HEADER_FILES
                            PROPERTIES HEADER_FILE_ONLY TRUE)
LIST(APPEND ${this_target}_SOURCE_FILES ${${this_target}_HEADER_FILES})

## section: add definitions
#   add prefix -D. example> -DSHP
#  - DO NOT add  the following definitions(already defined in ${OSP_DEFINITIONS}:
#   -DSHP, -DWIN32, -D_WINDOWS, -D_DEBUG, -D_USRDLL, -D_CRT_SECURE_NO_DEPRECATE
ADD_DEFINITIONS(
    -D_CONSOLE
    )

## section: add target

ADD_LIBRARY(${this_target} SHARED ${${this_target}_SOURCE_FILES})
#ADD_EXECUTABLE(${this_target} ${${this_target}_SOURCE_FILES})

## section: add dependency
# dependency determines overall build order.
ADD_DEPENDENCIES(${this_target} 
        ${CUDA_CUDART_LIBRARY}
        ${BASE_DIR}/cudaCompress/build/lib/libcudaCompress.so
	)

## section: set link libraries
TARGET_LINK_LIBRARIES( ${this_target}
        ${CUDA_CUDART_LIBRARY}
        )

install(FILES ${HFILES_PUBLIC}
  DESTINATION include
  )

install(TARGETS CPU 
  DESTINATION lib
  EXPORT CPU-targets
  )

