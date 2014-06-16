PROJECT(examples)

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

## start setting
SET (this_target examples)

SET (CMAKE_CXX_COMPILER g++)
SET (CMAKE_CXX_FLAGS "-std=c++0x -fopenmp")

set(EXAMPLES_INCLUDE_DIRS
  ${PROJECT_SOURCE_DIR}/../../include
  ${PROJECT_SOURCE_DIR}/../../src
)

## section: include directory
INCLUDE_DIRECTORIES(
  $(SolutionDir)include
        ${CUDA_TOOLKIT_INCLUDE}
        ${EXAMPLES_INCLUDE_DIRS}
  )

## section: source files
# Add your source files here (one file per line), please SORT in alphabetical order for future maintenance
SET (${this_target}_SOURCE_FILES
    CompressHeightfield.cpp
	CompressImage.cpp
	CompressVolume.cpp
	GPUResources.cpp
	main.cpp
	examples.cpp
	simpleExample.cpp
	tools/rawfile.cpp
	tthread/tinythread.cpp
    )

## section: header files
# Add your header files here(one file per line), please SORT in alphabetical order for future maintenance!
SET(${this_target}_HEADER_FILES
    CompressHeightfield.h
	CompressImage.h
	CompressVolume.h
	cudaUtil.h
	global.h
	GPUResources.h
	examples.h
	tools/entropy.h
	tools/imgtools.h
	tools/rawfile.h
	tthread/tinythread.h
    )

SOURCE_GROUP("" FILES 
	CompressHeightfield.cpp
	CompressImage.cpp
	CompressVolume.cpp
	main.cpp
	GPUResources.cpp
	examples.cpp
	simpleExample.cpp
	CompressHeightfield.h
	CompressImage.h
	CompressVolume.h
	cudaUtil.h
	examples.h
	GPUResources.h
	global.h
	)
SOURCE_GROUP("tools" FILES 
	tools/rawfile.cpp
	tools/entropy.h
	tools/imgtools.h
	tools/rawfile.h
	)
SOURCE_GROUP("tthread" FILES 
	tthread/tinythread.cpp
	tthread/tinythread.h
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
    -DNOMINMAX
	-D_CONSOLE
    )

## section: add target


ADD_EXECUTABLE(${this_target} ${${this_target}_SOURCE_FILES})
#ADD_LIBRARY(${this_target} SHARED ${${this_target}_SOURCE_FILES})

## section: add dependency
# dependency determines overall build order.
ADD_DEPENDENCIES( ${this_target} 
	${CUDA_CUDART_LIBRARY}
	)

## section: set link libraries
TARGET_LINK_LIBRARIES( ${this_target}
	${CUDA_CUDART_LIBRARY} CPU cudaCompress
        )

install(FILES ${HFILES_PUBLIC}
  DESTINATION include
  )

install(TARGETS examples 
  DESTINATION lib
  EXPORT examples-targets
  )

