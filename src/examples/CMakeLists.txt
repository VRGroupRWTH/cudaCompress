cmake_minimum_required(VERSION 3.18)

# Project Configuration
# =====================
PROJECT(examples VERSION 1.0 LANGUAGES CXX)

SET (this_target ${PROJECT_NAME})

# Dependencies
# ============
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

# Source files
# ============
## CPP
SET (${PROJECT_NAME}_SOURCE_FILES
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

## H
SET(${PROJECT_NAME}_HEADER_FILES
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

# Executable (Target)
# ===================
ADD_EXECUTABLE(${PROJECT_NAME} ${${PROJECT_NAME}_SOURCE_FILES})

# Target Configuration
# ====================
#SET (CMAKE_CXX_COMPILER g++)
#SET (CMAKE_CXX_FLAGS "-std=c++0x -fopenmp")
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_11)

## CPP
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

set(EXAMPLES_INCLUDE_DIRS
  ${PROJECT_SOURCE_DIR}/../../include
  ${PROJECT_SOURCE_DIR}/../../src
)

## section: include directory
#INCLUDE_DIRECTORIES(
#    ${CUDA_TOOLKIT_INCLUDE}
#    ${EXAMPLES_INCLUDE_DIRS}
#)
target_include_directories(${PROJECT_NAME} PRIVATE ${EXAMPLES_INCLUDE_DIRS} ${CUDA_TOOLKIT_INCLUDE})

#SET_SOURCE_FILES_PROPERTIES(${PROJECT_NAME}_HEADER_FILES
#                            PROPERTIES HEADER_FILE_ONLY TRUE
#)
LIST(APPEND ${PROJECT_NAME}_SOURCE_FILES ${${PROJECT_NAME}_HEADER_FILES})

## section: add dependency
# dependency determines overall build order.
#ADD_DEPENDENCIES( ${this_target} 
#	${CUDA_CUDART_LIBRARY}
#)

## section: set link libraries
TARGET_LINK_LIBRARIES(${PROJECT_NAME}
	PRIVATE ${CUDA_CUDART_LIBRARY} CPU cudaCompress OpenMP::OpenMP_CXX
)

## section: add definitions
#   add prefix -D. example> -DSHP
#  - DO NOT add  the following definitions(already defined in ${OSP_DEFINITIONS}:
#   -DSHP, -DWIN32, -D_WINDOWS, -D_DEBUG, -D_USRDLL, -D_CRT_SECURE_NO_DEPRECATE
#ADD_DEFINITIONS(
#    -DNOMINMAX
#	-D_CONSOLE
#    )
target_compile_definitions(${PROJECT_NAME} PRIVATE NOMINMAX _CONSOLE)

install(FILES ${HFILES_PUBLIC}
  DESTINATION include
  )

install(TARGETS ${PROJECT_NAME} 
  DESTINATION lib
  EXPORT ${PROJECT_NAME}-targets
  )

