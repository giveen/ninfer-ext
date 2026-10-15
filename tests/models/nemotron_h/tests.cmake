# The hybrid block list, with per-layer expert banks: the widths derived from the checkpoint's own
# numbers, and every input the parser is supposed to refuse.
ninfer_add_test(ninfer_nemotron_h_config_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_config.cpp"
  LIBRARIES ninfer_model_loading)

# The composition's own test: it walks the tiny model block by block against the reference dump at
# NINFER_NEMOTRON_REF, so an unset variable skips rather than fails.
ninfer_add_test(ninfer_nemotron_h_forward_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_forward.cpp"
  LIBRARIES ninfer_model_runtime)

set_tests_properties(ninfer_nemotron_h_forward_test PROPERTIES SKIP_RETURN_CODE 77)
