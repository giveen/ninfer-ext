#include "ops/offload_moe/cpu_nvfp4.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#if defined(_MSC_VER)
#    include <immintrin.h>
#    include <intrin.h>
#endif

namespace ninfer::ops::cpu {

namespace avx2 {
void gate_up_rows(const Nvfp4Expert&, int, int, const Q8Row*, int, float* const*);
void down_rows(const Nvfp4Expert&, int, int, const Q8Row*, int, float* const*);
} // namespace avx2
namespace vnni {
void gate_up_rows(const Nvfp4Expert&, int, int, const Q8Row*, int, float* const*);
void down_rows(const Nvfp4Expert&, int, int, const Q8Row*, int, float* const*);
} // namespace vnni

namespace {

struct Isa {
    bool supported = false;
    bool vnni      = false;
};

Isa detect() noexcept {
#if defined(_MSC_VER)
    // GCC/Clang answer this from their own builtins; MSVC needs the leaves and the XCR0 state
    // checked directly. FMA, F16C and AVX2 are only usable when the OS saves the YMM state, which
    // is why OSXSAVE and XCR0 gate them.
    int regs[4] = {};
    __cpuidex(regs, 1, 0);
    const bool osxsave            = (regs[2] & (1 << 27)) != 0;
    const bool avx                = (regs[2] & (1 << 28)) != 0;
    const bool fma                = (regs[2] & (1 << 12)) != 0;
    const bool f16c               = (regs[2] & (1 << 29)) != 0;
    const unsigned long long xcr0 = osxsave ? _xgetbv(0) : 0ULL;
    const bool ymm_state          = (xcr0 & 0x6ULL) == 0x6ULL;
    __cpuidex(regs, 7, 0);
    const bool avx2    = (regs[1] & (1 << 5)) != 0;
    const bool avxvnni = (regs[0] & (1 << 4)) != 0;

    Isa isa;
    isa.supported = avx && ymm_state && avx2 && fma && f16c;
    isa.vnni      = isa.supported && avxvnni;
    return isa;
#else
    __builtin_cpu_init();
    Isa isa;
    isa.supported = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") &&
                    __builtin_cpu_supports("f16c");
    isa.vnni = isa.supported && __builtin_cpu_supports("avxvnni");
    return isa;
#endif
}

const Isa& isa() noexcept {
    static const Isa value = detect();
    return value;
}

void require(int r0, int r1, int rows, int tokens) {
    if (!isa().supported) { throw std::runtime_error("host NVFP4 experts need AVX2, FMA and F16C"); }
    if (r0 < 0 || r1 < r0 || r1 > rows) { throw std::invalid_argument("host NVFP4 row range"); }
    if (tokens < 1 || tokens > 8) { throw std::invalid_argument("host NVFP4 token count"); }
}

} // namespace

void quantize_q8_0(const float* x, int n, std::int8_t* codes, float* scales) {
    if (n % 32 != 0) { throw std::invalid_argument("Q8_0 rows are whole 32-value blocks"); }
    for (int b = 0; b < n / 32; ++b) {
        const float* v = x + b * 32;
        float amax     = 0.0F;
        for (int i = 0; i < 32; ++i) { amax = std::max(amax, std::fabs(v[i])); }
        const float d   = amax / 127.0F;
        const float inv = d != 0.0F ? 1.0F / d : 0.0F;
        scales[b]       = d;
        for (int i = 0; i < 32; ++i) {
            codes[b * 32 + i] = static_cast<std::int8_t>(std::nearbyint(v[i] * inv));
        }
    }
}

void nvfp4_gate_up_rows(const Nvfp4Expert& expert, int r0, int r1, const Q8Row* x, int tokens,
                        float* const* h) {
    require(r0, r1, kNvfp4Intermediate, tokens);
    if (isa().vnni) {
        vnni::gate_up_rows(expert, r0, r1, x, tokens, h);
    } else {
        avx2::gate_up_rows(expert, r0, r1, x, tokens, h);
    }
}

void nvfp4_down_rows(const Nvfp4Expert& expert, int r0, int r1, const Q8Row* h, int tokens,
                     float* const* out) {
    require(r0, r1, kNvfp4Hidden, tokens);
    if (isa().vnni) {
        vnni::down_rows(expert, r0, r1, h, tokens, out);
    } else {
        avx2::down_rows(expert, r0, r1, h, tokens, out);
    }
}

bool nvfp4_host_supported() noexcept { return isa().supported; }

const char* nvfp4_host_isa() noexcept { return isa().vnni ? "avx-vnni" : "avx2"; }

} // namespace ninfer::ops::cpu
