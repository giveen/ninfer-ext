#pragma once

#include "core/weight.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

enum class Q8AttnInputScheduleId {
    DecodeR8Direct,
    SplitKMmaDirect,
    MmaR32C64,
    MmaR32C128,
    MmaR64C64,
    MmaR64C96,
    MmaR64C128,
    MmaR128C64,
    MmaR128C80,
    DFlash2SmallT,
    DFlash2MmaR16C64K128,
    DFlash2MmaR32C32K128,
    DFlash2MmaR32C64K128,
    DFlash2MmaR32C64,
    DFlash2MmaR64C128,
};

struct Q8AttnInputProblem {
    std::int32_t input_rows;  // Rows of x (the hidden input).
    std::int32_t query_rows;  // Rows of the query output.
    std::int32_t kv_rows;     // Rows of each of the key and value outputs.
    std::int32_t parent_rows; // Output rows of the parent weight (Weight::n).
    std::int32_t padded_k;    // Padded K of the parent weight.
    std::int32_t cols;        // Token count (the second extent of x).
};

struct Q8AttnInputPlan {
    Q8AttnInputScheduleId schedule;
};

const char* q8_attn_input_schedule_name(Q8AttnInputScheduleId schedule) noexcept;
bool q8_attn_input_admits(const Q8AttnInputProblem& problem) noexcept;
Q8AttnInputPlan q8_attn_input_resolve_plan(const Q8AttnInputProblem& problem);

// One plan per output arity: the four-output forms write `gate`, the three-output forms do not.
// `gate` is a caller-provided output and every other tensor is required; no workspace is used.
// Each form re-resolves the problem from the tensors it is given and rejects a plan whose schedule
// does not match that resolution.

void q8_attn_input_execute_plan(const Q8AttnInputPlan& plan, const Tensor& x, const Weight& weight,
                                Tensor& q, Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream);
void q8_attn_input_execute_plan(const Q8AttnInputPlan& plan, const Tensor& x, const Weight& weight,
                                Tensor& q, Tensor& k, Tensor& v, cudaStream_t stream);
void q8_attn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                            Tensor& k, Tensor& v, cudaStream_t stream);
void q8_attn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k, Tensor& v,
                            cudaStream_t stream);

} // namespace ninfer::ops::detail
