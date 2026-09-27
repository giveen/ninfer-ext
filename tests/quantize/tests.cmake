ninfer_add_test(ninfer_exl3_viterbi_reference_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_exl3_viterbi_reference.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/exl3_viterbi_reference.cpp"
  LIBRARIES ninfer_artifact)
