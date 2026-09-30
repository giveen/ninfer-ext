# Test declarations are included from tests/CMakeLists.txt so executable paths and
# CTest working directories stay under build/tests.
function(ninfer_test_includes target)
  ninfer_internal_includes(${target})
  target_include_directories(${target} PRIVATE ${PROJECT_SOURCE_DIR}/tests)
endfunction()

function(ninfer_add_test name)
  cmake_parse_arguments(PARSE_ARGV 1 arg "NEEDS_SOURCE_DIR" "" "SOURCES;LIBRARIES;ARGS")
  add_executable(${name} ${arg_SOURCES})
  ninfer_test_includes(${name})
  target_link_libraries(${name} PRIVATE ${arg_LIBRARIES})
  if(arg_NEEDS_SOURCE_DIR)
    target_compile_definitions(${name} PRIVATE
      NINFER_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
      NINFER_PYTHON_EXECUTABLE="${Python3_EXECUTABLE}")
  endif()
  add_test(NAME ${name} COMMAND ${name} ${arg_ARGS})
endfunction()

# Opt-in real-model Engine test. `artifact_var` is the NINFER_ARTIFACT_* cache
# variable holding the explicit artifact path for this test; tests driven by
# NINFER_TEST_ARTIFACT receive it through the environment, and a test declaring
# ARGS (the loading test) receives the artifact and its options on the command
# line. An unconfigured test skips (77) instead of inheriting a shared artifact,
# and the `real` label selects the whole suite with `ctest -L real`.
function(ninfer_add_real_test name artifact_var)
  cmake_parse_arguments(PARSE_ARGV 2 arg "" "" "SOURCES;LIBRARIES;ARGS")
  set(real_args ${arg_ARGS})
  if(arg_ARGS)
    if(DEFINED ${artifact_var} AND NOT "${${artifact_var}}" STREQUAL "")
      list(PREPEND real_args --artifact "${${artifact_var}}")
    endif()
  endif()
  ninfer_add_test(${name} SOURCES ${arg_SOURCES} LIBRARIES ${arg_LIBRARIES} ARGS ${real_args})
  set_tests_properties(${name} PROPERTIES LABELS real SKIP_RETURN_CODE 77)
  if(NOT arg_ARGS)
    if(DEFINED ${artifact_var} AND NOT "${${artifact_var}}" STREQUAL "")
      set_tests_properties(${name} PROPERTIES ENVIRONMENT "NINFER_TEST_ARTIFACT=${${artifact_var}}")
    else()
      set_tests_properties(${name} PROPERTIES
        ENVIRONMENT_MODIFICATION "NINFER_TEST_ARTIFACT=unset:")
    endif()
  endif()
endfunction()

# Apply these to the translation unit containing the oracle, including shared
# test support libraries. Executable options do not propagate into those libraries.
function(ninfer_op_oracle_options target)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-fno-fast-math>
      $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>)
  endif()
endfunction()

function(ninfer_add_op_test name)
  ninfer_add_test(${name} ${ARGN})
  set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77)
  ninfer_op_oracle_options(${name})
endfunction()
