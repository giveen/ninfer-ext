# The hybrid block list, with per-layer expert banks: the widths derived from the checkpoint's own
# numbers, and every input the parser is supposed to refuse.
ninfer_add_test(ninfer_nemotron_h_config_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_config.cpp"
  LIBRARIES ninfer_model_loading)
