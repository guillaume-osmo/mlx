// Copyright © 2024 Apple Inc.
// Fused GRU cell for RNN on Metal. See Apple Metal docs:
// https://developer.apple.com/documentation/metal

#include <metal_math>
#include <metal_stdlib>

#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

// --- Fast path (approximate, for speed) ---
inline float fast_sigmoid_impl(float x) {
  float y = 1.0f / (1.0f + metal::fast::exp(-metal::abs(x)));
  return (x < 0.0f) ? 1.0f - y : y;
}
inline float fast_tanh_impl(float x) {
  float c = metal::min(metal::max(x, -20.0f), 20.0f);
  return metal::fast::tanh(c);
}

// --- Stable path (precise, no NaN; matches mx.tanh / mx.sigmoid) ---
inline float stable_sigmoid_impl(float x) {
  if (x >= 0.0f) {
    return 1.0f / (1.0f + metal::precise::exp(-x));
  }
  float ex = metal::precise::exp(x);
  return ex / (1.0f + ex);
}
inline float stable_tanh_impl(float x) {
  return metal::precise::tanh(x);
}

// Selectable: use_precise_math != 0 => stable path, else fast path.
inline float gru_sigmoid(float x, constant uint& use_precise_math) {
  return use_precise_math ? stable_sigmoid_impl(x) : fast_sigmoid_impl(x);
}
inline float gru_tanh(float x, constant uint& use_precise_math) {
  return use_precise_math ? stable_tanh_impl(x) : fast_tanh_impl(x);
}
inline float4 gru_sigmoid4(float4 x, constant uint& use_precise_math) {
  return float4(
      gru_sigmoid(x.x, use_precise_math),
      gru_sigmoid(x.y, use_precise_math),
      gru_sigmoid(x.z, use_precise_math),
      gru_sigmoid(x.w, use_precise_math));
}
inline float4 gru_tanh4(float4 x, constant uint& use_precise_math) {
  return float4(
      gru_tanh(x.x, use_precise_math),
      gru_tanh(x.y, use_precise_math),
      gru_tanh(x.z, use_precise_math),
      gru_tanh(x.w, use_precise_math));
}

// Fused GRU cell: one kernel for gating. Inputs are pre-computed (e.g. by
// GEMM). Optimized: vectorized float4 path when 4 consecutive h fit, else
// scalar.
[[host_name("gru_cell_fused_float")]] [[kernel]] void gru_cell_fused_float(
    const device float* input_proj, // [B, 3H] input projection
    const device float* hidden_proj, // [B, 3H] hidden projection
    const device float* hidden_prev, // [B, H] previous hidden state
    device float* output, // [B, H] output hidden state
    constant uint& batch_size,
    constant uint& hidden_size,
    constant uint& use_precise_math, // 0 = fast (approx), 1 = stable (precise tanh/sigmoid)
    uint idx [[thread_position_in_grid]]) {
  uint stride_3h = 3u * hidden_size;
  uint h_quads = (hidden_size + 3u) / 4u;
  uint total_quads = batch_size * h_quads;
  if (idx >= total_quads)
    return;

  uint batch_idx = idx / h_quads;
  uint h_base = (idx % h_quads) * 4u;
  uint base = batch_idx * stride_3h + h_base;
  uint prev_base = batch_idx * hidden_size + h_base;

  // Vectorized path: 4 consecutive elements, coalesced float4 load/store
  // GRU: n = tanh(x_n + r * h_proj_n), out = (1-z)*n + z*h_prev (h_proj_n =
  // hidden_proj[2*H:])
  if (h_base + 4u <= hidden_size) {
    float4 x_r4 = *reinterpret_cast<const device float4*>(input_proj + base) +
        *reinterpret_cast<const device float4*>(hidden_proj + base);
    float4 x_z4 = *reinterpret_cast<const device float4*>(
                      input_proj + base + hidden_size) +
        *reinterpret_cast<const device float4*>(
            hidden_proj + base + hidden_size);
    float4 x_n4 = *reinterpret_cast<const device float4*>(
        input_proj + base + 2u * hidden_size);
    float4 h_proj_n4 = *reinterpret_cast<const device float4*>(
        hidden_proj + base + 2u * hidden_size);
    float4 h_prev4 =
        *reinterpret_cast<const device float4*>(hidden_prev + prev_base);

    float4 r4 = gru_sigmoid4(x_r4, use_precise_math);
    float4 z4 = gru_sigmoid4(x_z4, use_precise_math);
    float4 n4 = gru_tanh4(x_n4 + r4 * h_proj_n4, use_precise_math);
    *reinterpret_cast<device float4*>(output + prev_base) =
        (1.0f - z4) * n4 + z4 * h_prev4;
    return;
  }

  // Scalar tail (last partial quad per row)
  for (uint i = 0u; i < 4u && (h_base + i) < hidden_size; i++) {
    uint b = base + i;
    uint pb = prev_base + i;
    float x_r = input_proj[b] + hidden_proj[b];
    float x_z = input_proj[b + hidden_size] + hidden_proj[b + hidden_size];
    float x_n = input_proj[b + 2u * hidden_size];
    float h_proj_n = hidden_proj[b + 2u * hidden_size];
    float h_prev = hidden_prev[pb];
    float r = gru_sigmoid(x_r, use_precise_math);
    float z = gru_sigmoid(x_z, use_precise_math);
    float n = gru_tanh(x_n + r * h_proj_n, use_precise_math);
    output[pb] = (1.0f - z) * n + z * h_prev;
  }
}

// Same as above but with recurrent bias bhn [H] for n-gate; avoids per-step add
// in Python.
[[host_name("gru_cell_fused_float_bias")]] [[kernel]] void
gru_cell_fused_float_bias(
    const device float* input_proj,
    const device float* hidden_proj,
    const device float* hidden_prev,
    const device float* bhn,
    device float* output,
    constant uint& batch_size,
    constant uint& hidden_size,
    constant uint& use_precise_math,
    uint idx [[thread_position_in_grid]]) {
  uint stride_3h = 3u * hidden_size;
  uint h_quads = (hidden_size + 3u) / 4u;
  uint total_quads = batch_size * h_quads;
  if (idx >= total_quads)
    return;

  uint batch_idx = idx / h_quads;
  uint h_base = (idx % h_quads) * 4u;
  uint base = batch_idx * stride_3h + h_base;
  uint prev_base = batch_idx * hidden_size + h_base;

  if (h_base + 4u <= hidden_size) {
    float4 x_r4 = *reinterpret_cast<const device float4*>(input_proj + base) +
        *reinterpret_cast<const device float4*>(hidden_proj + base);
    float4 x_z4 = *reinterpret_cast<const device float4*>(
                      input_proj + base + hidden_size) +
        *reinterpret_cast<const device float4*>(
            hidden_proj + base + hidden_size);
    float4 x_n4 = *reinterpret_cast<const device float4*>(
        input_proj + base + 2u * hidden_size);
    float4 h_proj_n4 = *reinterpret_cast<const device float4*>(
                           hidden_proj + base + 2u * hidden_size) +
        *reinterpret_cast<const device float4*>(bhn + h_base);
    float4 h_prev4 =
        *reinterpret_cast<const device float4*>(hidden_prev + prev_base);

    float4 r4 = gru_sigmoid4(x_r4, use_precise_math);
    float4 z4 = gru_sigmoid4(x_z4, use_precise_math);
    float4 n4 = gru_tanh4(x_n4 + r4 * h_proj_n4, use_precise_math);
    *reinterpret_cast<device float4*>(output + prev_base) =
        (1.0f - z4) * n4 + z4 * h_prev4;
    return;
  }

  for (uint i = 0u; i < 4u && (h_base + i) < hidden_size; i++) {
    uint b = base + i;
    uint pb = prev_base + i;
    float x_r = input_proj[b] + hidden_proj[b];
    float x_z = input_proj[b + hidden_size] + hidden_proj[b + hidden_size];
    float x_n = input_proj[b + 2u * hidden_size];
    float h_proj_n = hidden_proj[b + 2u * hidden_size] + bhn[h_base + i];
    float h_prev = hidden_prev[pb];
    float r = gru_sigmoid(x_r, use_precise_math);
    float z = gru_sigmoid(x_z, use_precise_math);
    float n = gru_tanh(x_n + r * h_proj_n, use_precise_math);
    output[pb] = (1.0f - z) * n + z * h_prev;
  }
}

// BFloat16 path: vectorized float4 when possible, else scalar; math in float,
// bf16 on write.
[[host_name("gru_cell_fused_bfloat16")]] [[kernel]] void
gru_cell_fused_bfloat16(
    const device bfloat16_t* input_proj,
    const device bfloat16_t* hidden_proj,
    const device bfloat16_t* hidden_prev,
    device bfloat16_t* output,
    constant uint& batch_size,
    constant uint& hidden_size,
    constant uint& use_precise_math,
    uint idx [[thread_position_in_grid]]) {
  uint stride_3h = 3u * hidden_size;
  uint h_quads = (hidden_size + 3u) / 4u;
  uint total_quads = batch_size * h_quads;
  if (idx >= total_quads)
    return;

  uint batch_idx = idx / h_quads;
  uint h_base = (idx % h_quads) * 4u;
  uint base = batch_idx * stride_3h + h_base;
  uint prev_base = batch_idx * hidden_size + h_base;

  if (h_base + 4u <= hidden_size) {
    float4 x_r4 = float4(
                      input_proj[base],
                      input_proj[base + 1],
                      input_proj[base + 2],
                      input_proj[base + 3]) +
        float4(hidden_proj[base],
               hidden_proj[base + 1],
               hidden_proj[base + 2],
               hidden_proj[base + 3]);
    float4 x_z4 = float4(
                      input_proj[base + hidden_size],
                      input_proj[base + hidden_size + 1],
                      input_proj[base + hidden_size + 2],
                      input_proj[base + hidden_size + 3]) +
        float4(hidden_proj[base + hidden_size],
               hidden_proj[base + hidden_size + 1],
               hidden_proj[base + hidden_size + 2],
               hidden_proj[base + hidden_size + 3]);
    float4 x_n4 = float4(
        input_proj[base + 2u * hidden_size],
        input_proj[base + 2u * hidden_size + 1],
        input_proj[base + 2u * hidden_size + 2],
        input_proj[base + 2u * hidden_size + 3]);
    float4 h_proj_n4 = float4(
        hidden_proj[base + 2u * hidden_size],
        hidden_proj[base + 2u * hidden_size + 1],
        hidden_proj[base + 2u * hidden_size + 2],
        hidden_proj[base + 2u * hidden_size + 3]);
    float4 h_prev4 = float4(
        hidden_prev[prev_base],
        hidden_prev[prev_base + 1],
        hidden_prev[prev_base + 2],
        hidden_prev[prev_base + 3]);

    float4 r4 = gru_sigmoid4(x_r4, use_precise_math);
    float4 z4 = gru_sigmoid4(x_z4, use_precise_math);
    float4 n4 = gru_tanh4(x_n4 + r4 * h_proj_n4, use_precise_math);
    float4 out4 = (1.0f - z4) * n4 + z4 * h_prev4;
    output[prev_base] = bfloat16_t(out4.x);
    output[prev_base + 1] = bfloat16_t(out4.y);
    output[prev_base + 2] = bfloat16_t(out4.z);
    output[prev_base + 3] = bfloat16_t(out4.w);
    return;
  }

  for (uint i = 0u; i < 4u && (h_base + i) < hidden_size; i++) {
    uint b = base + i;
    uint pb = prev_base + i;
    float x_r = float(input_proj[b]) + float(hidden_proj[b]);
    float x_z = float(input_proj[b + hidden_size]) +
        float(hidden_proj[b + hidden_size]);
    float x_n = float(input_proj[b + 2u * hidden_size]);
    float h_proj_n = float(hidden_proj[b + 2u * hidden_size]);
    float h_prev = float(hidden_prev[pb]);
    float r = gru_sigmoid(x_r, use_precise_math);
    float z = gru_sigmoid(x_z, use_precise_math);
    float n = gru_tanh(x_n + r * h_proj_n, use_precise_math);
    output[pb] = bfloat16_t((1.0f - z) * n + z * h_prev);
  }
}

// BFloat16 path with bhn
[[host_name("gru_cell_fused_bfloat16_bias")]] [[kernel]] void
gru_cell_fused_bfloat16_bias(
    const device bfloat16_t* input_proj,
    const device bfloat16_t* hidden_proj,
    const device bfloat16_t* hidden_prev,
    const device bfloat16_t* bhn,
    device bfloat16_t* output,
    constant uint& batch_size,
    constant uint& hidden_size,
    constant uint& use_precise_math,
    uint idx [[thread_position_in_grid]]) {
  uint stride_3h = 3u * hidden_size;
  uint h_quads = (hidden_size + 3u) / 4u;
  uint total_quads = batch_size * h_quads;
  if (idx >= total_quads)
    return;

  uint batch_idx = idx / h_quads;
  uint h_base = (idx % h_quads) * 4u;
  uint base = batch_idx * stride_3h + h_base;
  uint prev_base = batch_idx * hidden_size + h_base;

  if (h_base + 4u <= hidden_size) {
    float4 x_r4 = float4(
                      input_proj[base],
                      input_proj[base + 1],
                      input_proj[base + 2],
                      input_proj[base + 3]) +
        float4(hidden_proj[base],
               hidden_proj[base + 1],
               hidden_proj[base + 2],
               hidden_proj[base + 3]);
    float4 x_z4 = float4(
                      input_proj[base + hidden_size],
                      input_proj[base + hidden_size + 1],
                      input_proj[base + hidden_size + 2],
                      input_proj[base + hidden_size + 3]) +
        float4(hidden_proj[base + hidden_size],
               hidden_proj[base + hidden_size + 1],
               hidden_proj[base + hidden_size + 2],
               hidden_proj[base + hidden_size + 3]);
    float4 x_n4 = float4(
        input_proj[base + 2u * hidden_size],
        input_proj[base + 2u * hidden_size + 1],
        input_proj[base + 2u * hidden_size + 2],
        input_proj[base + 2u * hidden_size + 3]);
    float4 h_proj_n4 = float4(
                           hidden_proj[base + 2u * hidden_size],
                           hidden_proj[base + 2u * hidden_size + 1],
                           hidden_proj[base + 2u * hidden_size + 2],
                           hidden_proj[base + 2u * hidden_size + 3]) +
        float4(bhn[h_base], bhn[h_base + 1], bhn[h_base + 2], bhn[h_base + 3]);
    float4 h_prev4 = float4(
        hidden_prev[prev_base],
        hidden_prev[prev_base + 1],
        hidden_prev[prev_base + 2],
        hidden_prev[prev_base + 3]);

    float4 r4 = gru_sigmoid4(x_r4, use_precise_math);
    float4 z4 = gru_sigmoid4(x_z4, use_precise_math);
    float4 n4 = gru_tanh4(x_n4 + r4 * h_proj_n4, use_precise_math);
    float4 out4 = (1.0f - z4) * n4 + z4 * h_prev4;
    output[prev_base] = bfloat16_t(out4.x);
    output[prev_base + 1] = bfloat16_t(out4.y);
    output[prev_base + 2] = bfloat16_t(out4.z);
    output[prev_base + 3] = bfloat16_t(out4.w);
    return;
  }

  for (uint i = 0u; i < 4u && (h_base + i) < hidden_size; i++) {
    uint b = base + i;
    uint pb = prev_base + i;
    float x_r = float(input_proj[b]) + float(hidden_proj[b]);
    float x_z = float(input_proj[b + hidden_size]) +
        float(hidden_proj[b + hidden_size]);
    float x_n = float(input_proj[b + 2u * hidden_size]);
    float h_proj_n =
        float(hidden_proj[b + 2u * hidden_size]) + float(bhn[h_base + i]);
    float h_prev = float(hidden_prev[pb]);
    float r = gru_sigmoid(x_r, use_precise_math);
    float z = gru_sigmoid(x_z, use_precise_math);
    float n = gru_tanh(x_n + r * h_proj_n, use_precise_math);
    output[pb] = bfloat16_t((1.0f - z) * n + z * h_prev);
  }
}

[[host_name("gru_cell_fused_float_vjp")]] [[kernel]] void
gru_cell_fused_float_vjp(
    const device float* input_proj,   // [B, 3H]
    const device float* hidden_proj, // [B, 3H]
    const device float* hidden_prev, // [B, H]
    const device float* cot_out,     // [B, H]
    device float* d_input_proj,      // [B, 3H]
    device float* d_hidden_proj,     // [B, 3H]
    device float* d_hidden_prev,     // [B, H]
    constant uint& batch_size,
    constant uint& hidden_size,
    constant uint& use_precise_math,
    uint idx [[thread_position_in_grid]]) {
  uint stride_3h = 3u * hidden_size;
  uint h_quads = (hidden_size + 3u) / 4u;
  uint total_quads = batch_size * h_quads;
  if (idx >= total_quads)
    return;

  uint batch_idx = idx / h_quads;
  uint h_base = (idx % h_quads) * 4u;
  uint base = batch_idx * stride_3h + h_base;
  uint prev_base = batch_idx * hidden_size + h_base;

  if (h_base + 4u <= hidden_size) {
    float4 x_r4 = *reinterpret_cast<const device float4*>(input_proj + base) +
        *reinterpret_cast<const device float4*>(hidden_proj + base);
    float4 x_z4 =
        *reinterpret_cast<const device float4*>(input_proj + base + hidden_size) +
        *reinterpret_cast<const device float4*>(
            hidden_proj + base + hidden_size);
    float4 x_n4 = *reinterpret_cast<const device float4*>(
        input_proj + base + 2u * hidden_size);
    float4 h_n4 = *reinterpret_cast<const device float4*>(
        hidden_proj + base + 2u * hidden_size);
    float4 h_prev4 =
        *reinterpret_cast<const device float4*>(hidden_prev + prev_base);

    float4 r4 = gru_sigmoid4(x_r4, use_precise_math);
    float4 z4 = gru_sigmoid4(x_z4, use_precise_math);
    float4 n4 = gru_tanh4(x_n4 + r4 * h_n4, use_precise_math);
    float4 cot4 = *reinterpret_cast<const device float4*>(cot_out + prev_base);

    float4 dn4 = cot4 * (1.0f - z4);
    float4 dz4 = cot4 * (h_prev4 - n4);
    float4 dn_pre4 = dn4 * (1.0f - n4 * n4);
    float4 dr4 = dn_pre4 * h_n4;

    float4 d_x_r4 = dr4 * r4 * (1.0f - r4);
    float4 d_x_z4 = dz4 * z4 * (1.0f - z4);
    float4 d_x_n4 = dn_pre4;
    float4 d_h_n4 = dn_pre4 * r4;

    *reinterpret_cast<device float4*>(d_input_proj + base) = d_x_r4;
    *reinterpret_cast<device float4*>(d_input_proj + base + hidden_size) = d_x_z4;
    *reinterpret_cast<device float4*>(d_input_proj + base + 2u * hidden_size) =
        d_x_n4;

    *reinterpret_cast<device float4*>(d_hidden_proj + base) = d_x_r4;
    *reinterpret_cast<device float4*>(d_hidden_proj + base + hidden_size) = d_x_z4;
    *reinterpret_cast<device float4*>(d_hidden_proj + base + 2u * hidden_size) =
        d_h_n4;

    *reinterpret_cast<device float4*>(d_hidden_prev + prev_base) = cot4 * z4;
    return;
  }

  for (uint i = 0u; i < 4u && (h_base + i) < hidden_size; i++) {
    uint b = base + i;
    uint pb = prev_base + i;

    float x_r = input_proj[b] + hidden_proj[b];
    float x_z = input_proj[b + hidden_size] + hidden_proj[b + hidden_size];
    float x_n = input_proj[b + 2u * hidden_size];
    float h_n = hidden_proj[b + 2u * hidden_size];
    float h_prev = hidden_prev[pb];

    float r = gru_sigmoid(x_r, use_precise_math);
    float z = gru_sigmoid(x_z, use_precise_math);
    float n_pre = x_n + r * h_n;
    float n = gru_tanh(n_pre, use_precise_math);
    float cot = cot_out[pb];

    float dn = cot * (1.0f - z);
    float dz = cot * (h_prev - n);
    float dn_pre = dn * (1.0f - n * n);
    float dr = dn_pre * h_n;

    float d_x_r = dr * r * (1.0f - r);
    float d_x_z = dz * z * (1.0f - z);
    float d_x_n = dn_pre;
    float d_h_n = dn_pre * r;

    d_input_proj[b] = d_x_r;
    d_input_proj[b + hidden_size] = d_x_z;
    d_input_proj[b + 2u * hidden_size] = d_x_n;

    d_hidden_proj[b] = d_x_r;
    d_hidden_proj[b + hidden_size] = d_x_z;
    d_hidden_proj[b + 2u * hidden_size] = d_h_n;

    d_hidden_prev[pb] = cot * z;
  }
}

[[host_name("gru_cell_fused_float_vjp_bias")]] [[kernel]] void
gru_cell_fused_float_vjp_bias(
    const device float* input_proj,   // [B, 3H]
    const device float* hidden_proj, // [B, 3H]
    const device float* hidden_prev, // [B, H]
    const device float* bhn,         // [H]
    const device float* cot_out,     // [B, H]
    device float* d_input_proj,      // [B, 3H]
    device float* d_hidden_proj,     // [B, 3H]
    device float* d_hidden_prev,     // [B, H]
    constant uint& batch_size,
    constant uint& hidden_size,
    constant uint& use_precise_math,
    uint idx [[thread_position_in_grid]]) {
  uint stride_3h = 3u * hidden_size;
  uint h_quads = (hidden_size + 3u) / 4u;
  uint total_quads = batch_size * h_quads;
  if (idx >= total_quads)
    return;

  uint batch_idx = idx / h_quads;
  uint h_base = (idx % h_quads) * 4u;
  uint base = batch_idx * stride_3h + h_base;
  uint prev_base = batch_idx * hidden_size + h_base;

  if (h_base + 4u <= hidden_size) {
    float4 x_r4 = *reinterpret_cast<const device float4*>(input_proj + base) +
        *reinterpret_cast<const device float4*>(hidden_proj + base);
    float4 x_z4 =
        *reinterpret_cast<const device float4*>(input_proj + base + hidden_size) +
        *reinterpret_cast<const device float4*>(
            hidden_proj + base + hidden_size);
    float4 x_n4 = *reinterpret_cast<const device float4*>(
        input_proj + base + 2u * hidden_size);
    float4 h_n4 = *reinterpret_cast<const device float4*>(
                     hidden_proj + base + 2u * hidden_size) +
        *reinterpret_cast<const device float4*>(bhn + h_base);
    float4 h_prev4 =
        *reinterpret_cast<const device float4*>(hidden_prev + prev_base);

    float4 r4 = gru_sigmoid4(x_r4, use_precise_math);
    float4 z4 = gru_sigmoid4(x_z4, use_precise_math);
    float4 n4 = gru_tanh4(x_n4 + r4 * h_n4, use_precise_math);
    float4 cot4 = *reinterpret_cast<const device float4*>(cot_out + prev_base);

    float4 dn4 = cot4 * (1.0f - z4);
    float4 dz4 = cot4 * (h_prev4 - n4);
    float4 dn_pre4 = dn4 * (1.0f - n4 * n4);
    float4 dr4 = dn_pre4 * h_n4;

    float4 d_x_r4 = dr4 * r4 * (1.0f - r4);
    float4 d_x_z4 = dz4 * z4 * (1.0f - z4);
    float4 d_x_n4 = dn_pre4;
    float4 d_h_n4 = dn_pre4 * r4;

    *reinterpret_cast<device float4*>(d_input_proj + base) = d_x_r4;
    *reinterpret_cast<device float4*>(d_input_proj + base + hidden_size) = d_x_z4;
    *reinterpret_cast<device float4*>(d_input_proj + base + 2u * hidden_size) =
        d_x_n4;

    *reinterpret_cast<device float4*>(d_hidden_proj + base) = d_x_r4;
    *reinterpret_cast<device float4*>(d_hidden_proj + base + hidden_size) = d_x_z4;
    *reinterpret_cast<device float4*>(d_hidden_proj + base + 2u * hidden_size) =
        d_h_n4;

    *reinterpret_cast<device float4*>(d_hidden_prev + prev_base) = cot4 * z4;
    return;
  }

  for (uint i = 0u; i < 4u && (h_base + i) < hidden_size; i++) {
    uint b = base + i;
    uint pb = prev_base + i;
    uint h_idx = h_base + i;

    float x_r = input_proj[b] + hidden_proj[b];
    float x_z = input_proj[b + hidden_size] + hidden_proj[b + hidden_size];
    float x_n = input_proj[b + 2u * hidden_size];
    float h_n = hidden_proj[b + 2u * hidden_size] + bhn[h_idx];
    float h_prev = hidden_prev[pb];

    float r = gru_sigmoid(x_r, use_precise_math);
    float z = gru_sigmoid(x_z, use_precise_math);
    float n_pre = x_n + r * h_n;
    float n = gru_tanh(n_pre, use_precise_math);
    float cot = cot_out[pb];

    float dn = cot * (1.0f - z);
    float dz = cot * (h_prev - n);
    float dn_pre = dn * (1.0f - n * n);
    float dr = dn_pre * h_n;

    float d_x_r = dr * r * (1.0f - r);
    float d_x_z = dz * z * (1.0f - z);
    float d_x_n = dn_pre;
    float d_h_n = dn_pre * r;

    d_input_proj[b] = d_x_r;
    d_input_proj[b + hidden_size] = d_x_z;
    d_input_proj[b + 2u * hidden_size] = d_x_n;

    d_hidden_proj[b] = d_x_r;
    d_hidden_proj[b + hidden_size] = d_x_z;
    d_hidden_proj[b + 2u * hidden_size] = d_h_n;

    d_hidden_prev[pb] = cot * z;
  }
}
