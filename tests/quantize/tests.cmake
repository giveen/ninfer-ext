ninfer_add_test(ninfer_exl3_viterbi_reference_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_exl3_viterbi_reference.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/exl3_viterbi_reference.cpp"
  LIBRARIES ninfer_artifact)

ninfer_add_test(ninfer_exl3_trellis_encoder_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_exl3_trellis_encoder.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/exl3_viterbi_reference.cpp"
  LIBRARIES ninfer_quantize ninfer_artifact)
set_tests_properties(ninfer_exl3_trellis_encoder_test PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_exl3_hadamard_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_exl3_hadamard.cpp"
  LIBRARIES ninfer_quantize)
set_tests_properties(ninfer_exl3_hadamard_test PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_exl3_hessian_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_exl3_hessian.cpp"
  LIBRARIES ninfer_quantize)
set_tests_properties(ninfer_exl3_hessian_test PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_exl3_block_ldl_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_exl3_block_ldl.cpp"
  LIBRARIES ninfer_quantize)
set_tests_properties(ninfer_exl3_block_ldl_test PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_exl3_ldlq_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_exl3_ldlq.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/exl3_viterbi_reference.cpp"
  LIBRARIES ninfer_quantize ninfer_artifact)
set_tests_properties(ninfer_exl3_ldlq_test PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_exl3_pipeline_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_exl3_pipeline.cpp"
  LIBRARIES ninfer_quantize ninfer_artifact)
set_tests_properties(ninfer_exl3_pipeline_test PROPERTIES SKIP_RETURN_CODE 77)

# Not a ctest: driven by tools/exl3/compare_tensor.py against exllamav3.
add_executable(ninfer_exl3_tensor_probe "${CMAKE_CURRENT_LIST_DIR}/exl3_tensor_probe.cpp")
ninfer_internal_includes(ninfer_exl3_tensor_probe)
target_link_libraries(ninfer_exl3_tensor_probe PRIVATE ninfer_quantize)

# CPU-only host-surface test for the quantizer app; it links no CUDA kernel library.
ninfer_add_test(ninfer_exl3_quantize_source_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_exl3_quantize_source.cpp"
          "${PROJECT_SOURCE_DIR}/apps/quantize/options.cpp"
          "${PROJECT_SOURCE_DIR}/apps/quantize/parameter_reader.cpp"
          "${PROJECT_SOURCE_DIR}/apps/quantize/hessian_io.cpp"
          "${PROJECT_SOURCE_DIR}/apps/quantize/source_writer.cpp"
  LIBRARIES ninfer_artifact)
target_include_directories(ninfer_exl3_quantize_source_test PRIVATE
  ${PROJECT_SOURCE_DIR}/apps/quantize)

add_test(NAME ninfer_exl3_quantize_interop_test
  COMMAND ${Python3_EXECUTABLE} -B "${CMAKE_CURRENT_LIST_DIR}/quantize_interop.py"
    $<TARGET_FILE:ninfer-quantize>)
