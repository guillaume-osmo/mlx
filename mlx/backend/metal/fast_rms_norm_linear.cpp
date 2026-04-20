// Copyright © 2026 Apple Inc.

#include <algorithm>
#include <string>

#include "mlx/allocator.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/jit/includes.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/fast_primitives.h"
#include "mlx/utils.h"

namespace mlx::core::fast {

namespace {

const char* rms_norm_linear_kernel_source() {
  return R"metal(
#include <metal_common>
#include <metal_simdgroup>

using namespace metal;

constant bool has_norm_weight [[function_constant(20)]];
constant bool has_bias [[function_constant(21)]];

template <typename T>
[[kernel]] void rms_norm_linear_kernel(
    const device T* x,
    const device T* norm_weight,
    const device T* linear_weight,
    const device T* bias,
    device T* out,
    constant uint& rows,
    constant uint& in_features,
    constant uint& out_features,
    constant float& eps,
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  if (row >= rows) {
    return;
  }

  constexpr int SIMD_SIZE = 32;
  threadgroup float local_sums[SIMD_SIZE];
  threadgroup float local_inv_rms[1];

  size_t row_offset = size_t(row) * in_features;
  float sumsq = 0.0f;
  for (uint col = lid; col < in_features; col += lsize) {
    float xv = static_cast<float>(x[row_offset + col]);
    sumsq += xv * xv;
  }
  sumsq = simd_sum(sumsq);

  if (simd_group_id == 0) {
    local_sums[simd_lane_id] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simd_lane_id == 0) {
    local_sums[simd_group_id] = sumsq;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simd_group_id == 0) {
    float block_sum = simd_sum(local_sums[simd_lane_id]);
    if (simd_lane_id == 0) {
      local_inv_rms[0] = metal::precise::rsqrt(block_sum / float(in_features) + eps);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float inv_rms = local_inv_rms[0];
  size_t out_row_offset = size_t(row) * out_features;
  for (uint out_col = lid; out_col < out_features; out_col += lsize) {
    float dot = 0.0f;
    size_t weight_offset = size_t(out_col) * in_features;
    for (uint col = 0; col < in_features; ++col) {
      float xv = static_cast<float>(x[row_offset + col]) * inv_rms;
      if (has_norm_weight) {
        xv *= static_cast<float>(norm_weight[col]);
      }
      dot += xv * static_cast<float>(linear_weight[weight_offset + col]);
    }
    if (has_bias) {
      dot += static_cast<float>(bias[out_col]);
    }
    out[out_row_offset + out_col] = static_cast<T>(dot);
  }
}

#define instantiate_rms_norm_linear(name, type) \
  template [[host_name("rms_norm_linear_" #name)]] [[kernel]] decltype(rms_norm_linear_kernel<type>) rms_norm_linear_kernel<type>;

instantiate_rms_norm_linear(float32, float)
instantiate_rms_norm_linear(float16, half)
instantiate_rms_norm_linear(bfloat16, bfloat16_t)
)metal";
}

bool is_supported_rms_norm_linear_dtype(Dtype t) {
  return t == float32 || t == float16 || t == bfloat16;
}

uint32_t pick_threads_per_group(
    uint32_t in_features,
    uint32_t out_features,
    uint32_t max_threads_per_group) {
  uint32_t target = 256u;
  if (in_features <= 128u && out_features <= 128u) {
    target = 128u;
  }
  if (out_features <= 64u) {
    target = 64u;
  }
  if (out_features <= 32u) {
    target = 32u;
  }
  target = std::min(target, std::max(1u, max_threads_per_group));
  if (target >= 32u) {
    target = std::max(32u, (target / 32u) * 32u);
  }
  return std::max(1u, target);
}

} // namespace

void RMSNormLinear::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  outputs = fallback_(inputs);
}

} // namespace mlx::core::fast
