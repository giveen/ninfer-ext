# Offline weight quantization maths (EXL3). Not linked into the Engine, CLI or serving.
add_library(ninfer_quantize STATIC
  exl3/hadamard.cu
  exl3/trellis_encoder.cu
)
ninfer_internal_includes(ninfer_quantize)
ninfer_cuda_archive(ninfer_quantize)
target_link_libraries(ninfer_quantize PUBLIC ninfer_core CUDA::cudart)
