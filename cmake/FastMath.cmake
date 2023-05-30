# Credits: https://github.com/CelestiaProject/Celestia/blob/master/cmake/FastMath.cmake

function(enable_fast_math flag)
  if(NOT ${flag})
    return()
  endif()

  if(MSVC)
    add_compile_options(/fp:fast)
  else()
    #add_compile_options(-ffast-math -fno-finite-math-only)
    add_compile_options(-ffast-math)
  endif()
endfunction()

function(target_enable_fast_math target flag)
  if(NOT ${flag})
    return()
  endif()

  if(MSVC)
    #target_compile_options(${target} PRIVATE /fp:fast)
    target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:
      /fp:fast
    >)
  else()
    #target_compile_options(${target} PRIVATE -ffast-math -fno-finite-math-only)
    #target_compile_options(${target} PRIVATE -ffast-math)
    target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:
      -ffast-math
    >)
  endif()
endfunction()