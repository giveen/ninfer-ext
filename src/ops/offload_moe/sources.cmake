target_sources(ninfer_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/offload_moe.cpp"
                                  "${CMAKE_CURRENT_LIST_DIR}/kernels.cu"
                                  "${CMAKE_CURRENT_LIST_DIR}/a4_kernels.cu")
