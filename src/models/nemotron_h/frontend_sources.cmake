# The frontend renders this family's chat template, owns its output decoder, and compiles the
# tool-call contract, so it lives in the runtime library beside the other families' frontends.
target_sources(ninfer_model_runtime PRIVATE "${CMAKE_CURRENT_LIST_DIR}/frontend.cpp")
