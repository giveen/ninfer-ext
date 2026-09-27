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
