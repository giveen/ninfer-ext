# The frontend renders the chat template, so it lives in the runtime library beside the Qwen frontend.
target_sources(ninfer_model_runtime PRIVATE "${CMAKE_CURRENT_LIST_DIR}/frontend.cpp")
