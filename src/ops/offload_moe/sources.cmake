target_sources(ninfer_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/offload_moe.cpp"
                                  "${CMAKE_CURRENT_LIST_DIR}/kernels.cu"
                                  "${CMAKE_CURRENT_LIST_DIR}/exl3_kernels.cu"
                                  "${CMAKE_CURRENT_LIST_DIR}/exl3_prefill.cu"
                                  "${CMAKE_CURRENT_LIST_DIR}/a4_kernels.cu"
                                  "${CMAKE_CURRENT_LIST_DIR}/cpu_nvfp4.cpp"
                                  "${CMAKE_CURRENT_LIST_DIR}/cpu_nvfp4_avx2.cpp"
                                  "${CMAKE_CURRENT_LIST_DIR}/cpu_nvfp4_vnni.cpp")
# Each host-kernel build carries its own instruction set; cpu_nvfp4.cpp dispatches at runtime.
set_source_files_properties("${CMAKE_CURRENT_LIST_DIR}/cpu_nvfp4_avx2.cpp"
                            PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
set_source_files_properties("${CMAKE_CURRENT_LIST_DIR}/cpu_nvfp4_vnni.cpp"
                            PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c;-mavxvnni")
