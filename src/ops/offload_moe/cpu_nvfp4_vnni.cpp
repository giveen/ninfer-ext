// AVX-VNNI build of the host NVFP4 expert kernels (compiled with -mavx2 -mfma -mf16c -mavxvnni):
// one vpdpbusd per four products instead of vpmaddubsw + vpmaddwd.
#define NINFER_CPU_NVFP4_NS vnni
#define NINFER_CPU_NVFP4_VNNI 1
#include "ops/offload_moe/cpu_nvfp4_body.inc"
