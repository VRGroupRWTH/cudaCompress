# Credits: https://stackoverflow.com/questions/2368811/how-to-set-warning-level-in-cmake/50882216#50882216

function(set_max_warning)
  if(MSVC)
    add_compile_options(/W4 /WX)
  else()
    add_compile_options(-Wall -Wextra -Wpedantic -Werror)
  endif()
endfunction()

function(target_set_max_warning target)
  if(MSVC)
    #target_compile_options(${target} PRIVATE /W4 /WX)
    target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:
      /W4 /WX
    >)
  else()
    #target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:
      -Wall -Wextra -Wpedantic -Werror
    >)
  endif()
endfunction()