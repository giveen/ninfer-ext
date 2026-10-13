# The measurement-applicability identity is built beside the Qwen one, and reuses its SHA-256.
target_sources(ninfer_model_runtime PRIVATE "${CMAKE_CURRENT_LIST_DIR}/measurement.cpp")
