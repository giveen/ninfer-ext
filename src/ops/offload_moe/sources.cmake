target_sources(ninfer_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/offload_moe.cpp"
                                  "${CMAKE_CURRENT_LIST_DIR}/kernels.cu"
                                  "${CMAKE_CURRENT_LIST_DIR}/exl3_kernels.cu"
                                  "${CMAKE_CURRENT_LIST_DIR}/exl3_prefill.cu"
                                  "${CMAKE_CURRENT_LIST_DIR}/a4_kernels.cu"
                                  "${CMAKE_CURRENT_LIST_DIR}/cpu_nvfp4.cpp"
                                  "${CMAKE_CURRENT_LIST_DIR}/cpu_nvfp4_avx2.cpp"
                                  "${CMAKE_CURRENT_LIST_DIR}/cpu_nvfp4_vnni.cpp")
# Each host-kernel build carries its own instruction set; cpu_nvfp4.cpp dispatches at runtime.
if(MSVC)
  # cl has no -m switches. /arch:AVX2 makes the AVX2/FMA/F16C intrinsics available; AVX-VNNI has
  # no /arch value, and the intrinsic emits the instruction wherever it is used, which is safe
  # because cpu_nvfp4.cpp only calls into this file after checking CPUID.
  set_source_files_properties("${CMAKE_CURRENT_LIST_DIR}/cpu_nvfp4_avx2.cpp"
                              PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
  set_source_files_properties("${CMAKE_CURRENT_LIST_DIR}/cpu_nvfp4_vnni.cpp"
                              PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
else()
  set_source_files_properties("${CMAKE_CURRENT_LIST_DIR}/cpu_nvfp4_avx2.cpp"
                              PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
  set_source_files_properties("${CMAKE_CURRENT_LIST_DIR}/cpu_nvfp4_vnni.cpp"
                              PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c;-mavxvnni")
endif()
