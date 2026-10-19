// AVX2 build of the host NVFP4 expert kernels (compiled with -mavx2 -mfma -mf16c).
#define NINFER_CPU_NVFP4_NS avx2
#define NINFER_CPU_NVFP4_VNNI 0
#include "ops/offload_moe/cpu_nvfp4_body.inc"
