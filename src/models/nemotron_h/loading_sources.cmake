# Loading side of the Nemotron-H family: the hybrid block list, with per-layer MoE banks, is parsed
# from the artifact's own config, and the loader that declares, plans and binds every parameter the
# block list implies joins it here. Its forward, program and execution sources join the execution
# manifest as they land, in the same loading/execution split the other families use (see
# src/models/CMakeLists.txt).
target_sources(ninfer_model_loading PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/config.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/load.cpp")
