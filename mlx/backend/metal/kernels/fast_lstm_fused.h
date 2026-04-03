// Copyright © 2024 Apple Inc.
// Persistent fused LSTM kernel using simdgroup_matrix for the recurrent GEMM.
//
// Single dispatch for all T timesteps.  Each threadgroup handles B_TILE
// batch elements (passed at runtime, always padded to 8 for simdgroup_matrix).
// The recurrent matmul h×Wh_t is computed with hardware 8×8 MMA.
//
// Dynamic B_TILE sizing:
//   TG memory = B_TILE_PAD * H * sizeof(T_h) + B_TILE_PAD * 4H * sizeof(float)
//   B_TILE_PAD = ((B_TILE + 7) / 8) * 8  (rounded up to 8 for simdgroup)
//   Must fit in 32KB.
//
// Float32 variant: shared_h in float, Wh_t in float
// BFloat16 variant: shared_h in bfloat16, Wh_t in bfloat16, accum in float32

#pragma once

#include <metal_stdlib>
#include <metal_math>
#include <metal_simdgroup_matrix>

#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

inline float fused_sigmoid(float x) {
  if (x >= 0.0f) {
    return 1.0f / (1.0f + metal::precise::exp(-x));
  }
  float ex = metal::precise::exp(x);
  return ex / (1.0f + ex);
}

inline float fused_tanh(float x) {
  return metal::precise::tanh(x);
}

// ====================================================================
// Float32 fused GEMM kernel
// ====================================================================

[[host_name("lstm_fused_smm_float")]]
[[kernel]] void lstm_fused_smm_float(
    const device float* input_proj,   // [B, T, 4H]
    const device float* Wh_t,         // [H, 4H] row-major (TRANSPOSED)
    const device float* h_init,       // [B, H]
    const device float* c_init,       // [B, H]
    device float* out_h,              // [B, T, H]
    device float* out_c,              // [B, T, H]
    constant uint& batch_size,
    constant uint& seq_len,
    constant uint& hidden_size,
    constant uint& b_tile,            // actual batches per TG
    constant uint& b_tile_pad,        // b_tile rounded up to multiple of 8
    threadgroup float* shared_h       [[threadgroup(0)]],  // [b_tile_pad * H]
    threadgroup float* gate_buf       [[threadgroup(1)]],  // [b_tile_pad * 4H]
    uint tid        [[thread_position_in_threadgroup]],
    uint sg_id      [[simdgroup_index_in_threadgroup]],
    uint tg_idx     [[threadgroup_position_in_grid]],
    uint tg_size    [[threads_per_threadgroup]]) {

  const uint H  = hidden_size;
  const uint H4 = 4u * H;
  const uint K_TILES = H / 8u;
  const uint N_TILES = H4 / 8u;
  const uint BTP = b_tile_pad;   // always multiple of 8
  const uint M_TILES = BTP / 8u; // number of 8-row tiles in batch dimension
  const uint b_base = tg_idx * b_tile;

  // --- Initialize shared_h from h_init (zero-pad inactive rows) ---
  for (uint i = tid; i < BTP * H; i += tg_size) {
    uint b = i / H;
    uint h = i % H;
    uint bg = b_base + b;
    shared_h[i] = (b < b_tile && bg < batch_size) ? h_init[bg * H + h] : 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // --- Timestep loop ---
  for (uint t = 0u; t < seq_len; ++t) {

    // === PHASE 1: GEMM  shared_h [BTP, H] × Wh_t [H, 4H] → gate_buf [BTP, 4H] ===
    //
    // Tile over M (batch) and N (gate) dimensions.
    // Each simdgroup handles one (m, n) output tile.
    // Total output tiles = M_TILES * N_TILES.
    uint total_mn = M_TILES * N_TILES;
    for (uint mn = sg_id; mn < total_mn; mn += 32u) {
      uint m = mn / N_TILES;
      uint n = mn % N_TILES;

      simdgroup_matrix<float, 8, 8> C(0);

      for (uint k = 0u; k < K_TILES; ++k) {
        simdgroup_matrix<float, 8, 8> A;
        simdgroup_matrix<float, 8, 8> B;

        simdgroup_load(A, shared_h + m * 8u * H + k * 8u, H);
        simdgroup_load(B, Wh_t + k * 8u * H4 + n * 8u, H4);
        simdgroup_multiply_accumulate(C, A, B, C);
      }

      simdgroup_store(C, gate_buf + m * 8u * H4 + n * 8u, H4);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // === PHASE 2: Add input_proj + activations + state update ===
    for (uint i = tid; i < b_tile * H; i += tg_size) {
      uint b = i / H;
      uint h = i % H;
      uint bg = b_base + b;
      if (bg >= batch_size) continue;

      uint x_off = bg * seq_len * H4 + t * H4;

      float gi = gate_buf[b * H4 + h]           + input_proj[x_off + h];
      float gf = gate_buf[b * H4 + H + h]       + input_proj[x_off + H + h];
      float gg = gate_buf[b * H4 + 2u * H + h]  + input_proj[x_off + 2u * H + h];
      float go = gate_buf[b * H4 + 3u * H + h]  + input_proj[x_off + 3u * H + h];

      float ig = fused_sigmoid(gi);
      float fg = fused_sigmoid(gf);
      float g  = fused_tanh(gg);
      float og = fused_sigmoid(go);

      float c_prev = (t == 0u)
          ? c_init[bg * H + h]
          : out_c[bg * seq_len * H + (t - 1u) * H + h];

      float c_new = fg * c_prev + ig * g;
      float h_new = og * fused_tanh(c_new);

      uint o_idx = bg * seq_len * H + t * H + h;
      out_h[o_idx] = h_new;
      out_c[o_idx] = c_new;

      shared_h[b * H + h] = h_new;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// ====================================================================
// BFloat16 fused GEMM kernel — mixed precision
// ====================================================================

[[host_name("lstm_fused_smm_bfloat16")]]
[[kernel]] void lstm_fused_smm_bfloat16(
    const device float* input_proj,       // [B, T, 4H] float32
    const device bfloat16_t* Wh_t_bf16,   // [H, 4H] bfloat16
    const device float* h_init,           // [B, H] float32
    const device float* c_init,           // [B, H] float32
    device float* out_h,                  // [B, T, H] float32
    device float* out_c,                  // [B, T, H] float32
    constant uint& batch_size,
    constant uint& seq_len,
    constant uint& hidden_size,
    constant uint& b_tile,
    constant uint& b_tile_pad,
    threadgroup bfloat16_t* shared_h_bf   [[threadgroup(0)]],  // [b_tile_pad * H]
    threadgroup float* gate_buf           [[threadgroup(1)]],  // [b_tile_pad * 4H]
    uint tid        [[thread_position_in_threadgroup]],
    uint sg_id      [[simdgroup_index_in_threadgroup]],
    uint tg_idx     [[threadgroup_position_in_grid]],
    uint tg_size    [[threads_per_threadgroup]]) {

  const uint H  = hidden_size;
  const uint H4 = 4u * H;
  const uint K_TILES = H / 8u;
  const uint N_TILES = H4 / 8u;
  const uint BTP = b_tile_pad;
  const uint M_TILES = BTP / 8u;
  const uint b_base = tg_idx * b_tile;

  for (uint i = tid; i < BTP * H; i += tg_size) {
    uint b = i / H;
    uint bg = b_base + b;
    uint h = i % H;
    float val = (b < b_tile && bg < batch_size) ? h_init[bg * H + h] : 0.0f;
    shared_h_bf[i] = static_cast<bfloat16_t>(val);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint t = 0u; t < seq_len; ++t) {

    uint total_mn = M_TILES * N_TILES;
    for (uint mn = sg_id; mn < total_mn; mn += 32u) {
      uint m = mn / N_TILES;
      uint n = mn % N_TILES;

      simdgroup_matrix<float, 8, 8> C(0);

      for (uint k = 0u; k < K_TILES; ++k) {
        simdgroup_matrix<bfloat16_t, 8, 8> A;
        simdgroup_matrix<bfloat16_t, 8, 8> B;

        simdgroup_load(A, shared_h_bf + m * 8u * H + k * 8u, H);
        simdgroup_load(B, Wh_t_bf16 + k * 8u * H4 + n * 8u, H4);
        simdgroup_multiply_accumulate(C, A, B, C);
      }

      simdgroup_store(C, gate_buf + m * 8u * H4 + n * 8u, H4);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < b_tile * H; i += tg_size) {
      uint b = i / H;
      uint h = i % H;
      uint bg = b_base + b;
      if (bg >= batch_size) continue;

      uint x_off = bg * seq_len * H4 + t * H4;

      float gi = gate_buf[b * H4 + h]           + input_proj[x_off + h];
      float gf = gate_buf[b * H4 + H + h]       + input_proj[x_off + H + h];
      float gg = gate_buf[b * H4 + 2u * H + h]  + input_proj[x_off + 2u * H + h];
      float go = gate_buf[b * H4 + 3u * H + h]  + input_proj[x_off + 3u * H + h];

      float ig = fused_sigmoid(gi);
      float fg = fused_sigmoid(gf);
      float g  = fused_tanh(gg);
      float og = fused_sigmoid(go);

      float c_prev = (t == 0u)
          ? c_init[bg * H + h]
          : out_c[bg * seq_len * H + (t - 1u) * H + h];

      float c_new = fg * c_prev + ig * g;
      float h_new = og * fused_tanh(c_new);

      uint o_idx = bg * seq_len * H + t * H + h;
      out_h[o_idx] = h_new;
      out_c[o_idx] = c_new;

      shared_h_bf[b * H + h] = static_cast<bfloat16_t>(h_new);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}
