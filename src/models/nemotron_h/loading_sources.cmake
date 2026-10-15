# Loading side of the Nemotron-H family: the hybrid block list, with per-layer MoE banks, is parsed
# from the artifact's own config. Its forward, program and execution sources join this manifest as they
# land, in the same loading/execution split the other families use (see src/models/CMakeLists.txt).
target_sources(ninfer_model_loading PRIVATE "${CMAKE_CURRENT_LIST_DIR}/config.cpp")
