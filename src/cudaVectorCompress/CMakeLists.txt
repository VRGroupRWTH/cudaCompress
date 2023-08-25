cmake_minimum_required(VERSION 3.18)

# Project Configuration
# =====================
PROJECT(cudaVectorCompress VERSION 1.0 LANGUAGES CXX CUDA)

SET (this_target ${PROJECT_NAME})

# Dependencies
# ============
CPMAddPackage("gh:gabime/spdlog#v1.11.0")
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

# Source files
# ============
## CPP
SET (${PROJECT_NAME}_SOURCE_FILES
	tthread/tinythread.cpp
	CompressVolume.cpp
	GPUResources.cpp
	ParticleHelpers.cpp
	main.cpp
)

## H
SET(${PROJECT_NAME}_HEADER_FILES
	tthread/tinythread.h
	CompressVolume.h
	cudaUtil.h
	Datasets.h
	GPUResources.h
	helper_math.h
	ParticleHelpers.h
	half.h
)

SET(${PROJECT_NAME}_CU_FILES
	cudaFuncs.cu
	cudaHelpers.cu
)

SET(${PROJECT_NAME}_CUH_FILES
	cudaFuncs.cuh
	cudaHelpers.cuh
)


# Executable (Target)
# ===================
ADD_EXECUTABLE(${PROJECT_NAME} ${${PROJECT_NAME}_SOURCE_FILES} ${${PROJECT_NAME}_HEADER_FILES} ${${PROJECT_NAME}_CU_FILES} ${${PROJECT_NAME}_CUH_FILES})

# Target Configuration
# ====================
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_11 cuda_std_11)
target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
#	-fno-strict-aliasing
	-lineinfo
#	-use_fast_math
#	-Xptxas
#	-dlcm=cg
>)
set_target_properties(${PROJECT_NAME} 
             PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set(CUDAVECTORCOMPRESS_INCLUDE_DIRS
  ${PROJECT_SOURCE_DIR}/../../include
  ${PROJECT_SOURCE_DIR}/../../src
)

## section: set link libraries
target_link_libraries(${PROJECT_NAME}
	PRIVATE CUDA::cudart_static cudaCompress OpenMP::OpenMP_CXX spdlog::spdlog
)

## section: include directory
target_include_directories(${PROJECT_NAME} PRIVATE ${CUDAVECTORCOMPRESS_INCLUDE_DIRS} ${CUDA_TOOLKIT_INCLUDE})


## section: add definitions
#   add prefix -D. example> -DSHP
#  - DO NOT add  the following definitions(already defined in ${OSP_DEFINITIONS}:
#   -DSHP, -DWIN32, -D_WINDOWS, -D_DEBUG, -D_USRDLL, -D_CRT_SECURE_NO_DEPRECATE
target_compile_definitions(${PROJECT_NAME} PRIVATE NOMINMAX _CONSOLE)

source_group("CUDA Source Files" FILES ${${PROJECT_NAME}_CU_FILES})
source_group("CUDA Header Files" FILES ${${PROJECT_NAME}_CUH_FILES})

#install(FILES ${HFILES_PUBLIC}
#  DESTINATION include
#)

#install(TARGETS ${PROJECT_NAME} 
#  DESTINATION lib
#  EXPORT ${PROJECT_NAME}-targets
#)

