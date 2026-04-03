// Copyright © 2024 Apple Inc.
// Full-sequence LSTM Metal kernel — single dispatch for all T timesteps.
//
// Two variants:
//   v1 (lstm_sequence_fused_float): Original layout, Wh [4H, H] row-major.
//       One threadgroup per batch, 4H threads.  Each thread reads its own
//       Wh row — stride-H between adjacent threads → poor coalescing.
//
//   v2 (lstm_sequence_v2_float): Transposed layout, Wh_t [H, 4H] row-major.
//       Same threadgroup structure but threads read contiguous Wh_t columns
//       → perfect coalescing (32× fewer cache lines per simdgroup).
//       Expected ~4-6× faster than the per-step matmul+gate approach.

#include <metal_stdlib>
#include <metal_math>
#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

// ---- Activation helpers (precise path only) ----

inline float seq_sigmoid(float x) {
  if (x >= 0.0f) {
    return 1.0f / (1.0f + metal::precise::exp(-x));
  }
  float ex = metal::precise::exp(x);
  return ex / (1.0f + ex);
}

inline float seq_tanh(float x) {
  return metal::precise::tanh(x);
}

// ====================================================================
// v1: original layout (Wh [4H, H], uncoalesced).  Kept as reference.
// ====================================================================

[[host_name("lstm_sequence_fused_float")]]
[[kernel]] void lstm_sequence_fused_float(
    const device float* input_proj,   // [B, T, 4H]
    const device float* Wh,           // [4H, H]
    const device float* h_init,       // [B, H]
    const device float* c_init,       // [B, H]
    device float* out_h,              // [B, T, H]
    device float* out_c,              // [B, T, H]
    constant uint& batch_size,
    constant uint& seq_len,
    constant uint& hidden_size,
    threadgroup float* shared_h       [[threadgroup(0)]],  // [H]
    threadgroup float* shared_c       [[threadgroup(1)]],  // [H]
    threadgroup float* shared_gates   [[threadgroup(2)]],  // [4H]
    uint tid        [[thread_position_in_threadgroup]],
    uint batch_idx  [[threadgroup_position_in_grid]]) {

  uint H = hidden_size;
  uint H4 = 4u * H;

  if (batch_idx >= batch_size || tid >= H4)
    return;

  uint h_idx = tid % H;

  // Initialize shared state
  if (tid < H) {
    shared_h[tid] = h_init[batch_idx * H + tid];
    shared_c[tid] = c_init[batch_idx * H + tid];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Pointer to Wh row for this thread's gate element
  const device float* wh_row = Wh + tid * H;

  for (uint t = 0u; t < seq_len; ++t) {
    // Dot product: sum_k(shared_h[k] * Wh[tid, k])
    // Pairwise summation for precision
    float dot = 0.0f;
    {
      uint k = 0u;
      for (; k + 7u < H; k += 8u) {
        float a0 = shared_h[k+0u] * wh_row[k+0u];
        float a1 = shared_h[k+1u] * wh_row[k+1u];
        float a2 = shared_h[k+2u] * wh_row[k+2u];
        float a3 = shared_h[k+3u] * wh_row[k+3u];
        float a4 = shared_h[k+4u] * wh_row[k+4u];
        float a5 = shared_h[k+5u] * wh_row[k+5u];
        float a6 = shared_h[k+6u] * wh_row[k+6u];
        float a7 = shared_h[k+7u] * wh_row[k+7u];
        dot += (a0 + a1) + (a2 + a3) + (a4 + a5) + (a6 + a7);
      }
      for (; k < H; ++k) {
        dot += shared_h[k] * wh_row[k];
      }
    }

    // Add input projection
    uint x_base = batch_idx * seq_len * H4 + t * H4;
    dot += input_proj[x_base + tid];

    // Store gate value to shared memory
    shared_gates[tid] = dot;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // First H threads apply gate activations and update state
    if (tid < H) {
      float ig = seq_sigmoid(shared_gates[0u * H + h_idx]);
      float fg = seq_sigmoid(shared_gates[1u * H + h_idx]);
      float gg = seq_tanh(shared_gates[2u * H + h_idx]);
      float og = seq_sigmoid(shared_gates[3u * H + h_idx]);

      float c_new = fg * shared_c[h_idx] + ig * gg;
      float h_new = og * seq_tanh(c_new);

      uint out_idx = batch_idx * seq_len * H + t * H + h_idx;
      out_h[out_idx] = h_new;
      out_c[out_idx] = c_new;

      shared_h[h_idx] = h_new;
      shared_c[h_idx] = c_new;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// ====================================================================
// v3: multi-batch with tiled Wh in threadgroup memory.
//
// Key optimization: pack B_TILE batches per threadgroup so they SHARE
// Wh tile reads from L2.  This reduces total L2 traffic by B_TILE×.
//
// Memory layout:
//   - shared_h [B_TILE * H]: hidden state for all batches in this TG
//   - wh_tile  [TILE_K * 4H]: cooperatively loaded Wh tile (shared!)
//   - c_state read/written from device memory (doesn't fit in TG memory)
//
// Thread mapping: B_TILE * H threads total.
//   local_b = tid / H  (which batch within the TG)
//   h_idx   = tid % H  (which hidden unit)
//   Each thread computes 4 gate dot products for its (batch, hidden_unit).
//
// For B=32, H=200, B_TILE=5: 1000 threads, 7 TGs.
// TG memory: 5*200*4 = 4KB (shared_h) + TILE_K*800*4 (wh_tile).
//   TILE_K=8 → 25.6KB. Total = 29.6KB. Fits in 32KB.
//
// L2 reads: 7 TGs × 640KB/step × 40 steps = 179MB (vs 820MB for v2).
// ====================================================================

[[host_name("lstm_sequence_v3_float")]]
[[kernel]] void lstm_sequence_v3_float(
    const device float* input_proj,   // [B, T, 4H] row-major, contiguous
    const device float* Wh_t,         // [H, 4H] row-major (TRANSPOSED)
    const device float* h_init,       // [B, H]
    const device float* c_init,       // [B, H]
    device float* out_h,              // [B, T, H]
    device float* out_c,              // [B, T, H]
    constant uint& batch_size,
    constant uint& seq_len,
    constant uint& hidden_size,
    constant uint& b_tile,            // batches per threadgroup
    constant uint& tile_k,            // K-dimension tile size
    threadgroup float* shared_h       [[threadgroup(0)]],  // [B_TILE * H]
    threadgroup float* wh_tile        [[threadgroup(1)]],  // [TILE_K * 4H]
    uint tid        [[thread_position_in_threadgroup]],
    uint tg_idx     [[threadgroup_position_in_grid]],
    uint tg_size    [[threads_per_threadgroup]]) {

  uint H = hidden_size;
  uint H4 = 4u * H;
  uint B_TILE = b_tile;
  uint TILE_K = tile_k;

  uint local_b = tid / H;
  uint h_idx = tid % H;
  uint b_global = tg_idx * B_TILE + local_b;

  // Guard: this thread might be padding
  bool active = (local_b < B_TILE) && (b_global < batch_size);

  // Initialize shared hidden state from h_init
  if (active) {
    shared_h[local_b * H + h_idx] = h_init[b_global * H + h_idx];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // We keep c_state in device memory to save TG memory.
  // Use the output c buffer as scratch: c_prev starts at c_init,
  // then each step writes c_new to out_c and reads it next step.

  for (uint t = 0u; t < seq_len; ++t) {
    // Accumulate 4 gate dot products in registers
    float dot_i = 0.0f, dot_f = 0.0f, dot_g = 0.0f, dot_o = 0.0f;

    // Tile over K dimension: load Wh_t[k_base:k_base+TILE_K, 0:4H]
    // into threadgroup memory, then compute partial dot products.
    for (uint k_base = 0u; k_base < H; k_base += TILE_K) {
      uint tile_rows = min(TILE_K, H - k_base);
      uint tile_elems = tile_rows * H4;

      // Cooperative load: all threads load wh_tile from L2
      for (uint i = tid; i < tile_elems; i += tg_size) {
        uint tk = i / H4;
        uint tn = i % H4;
        wh_tile[tk * H4 + tn] = Wh_t[(k_base + tk) * H4 + tn];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      // Each active thread accumulates partial dot products
      if (active) {
        for (uint dk = 0u; dk < tile_rows; ++dk) {
          float hk = shared_h[local_b * H + k_base + dk];
          uint wbase = dk * H4;
          dot_i += hk * wh_tile[wbase + h_idx];
          dot_f += hk * wh_tile[wbase + H + h_idx];
          dot_g += hk * wh_tile[wbase + 2u * H + h_idx];
          dot_o += hk * wh_tile[wbase + 3u * H + h_idx];
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (active) {
      // Add input projection
      uint x_base = b_global * seq_len * H4 + t * H4;
      dot_i += input_proj[x_base + h_idx];
      dot_f += input_proj[x_base + H + h_idx];
      dot_g += input_proj[x_base + 2u * H + h_idx];
      dot_o += input_proj[x_base + 3u * H + h_idx];

      // Gate activations
      float ig = seq_sigmoid(dot_i);
      float fg = seq_sigmoid(dot_f);
      float gg = seq_tanh(dot_g);
      float og = seq_sigmoid(dot_o);

      // Read c_prev from device memory (c_init for t=0, out_c for t>0)
      float c_prev;
      if (t == 0u) {
        c_prev = c_init[b_global * H + h_idx];
      } else {
        c_prev = out_c[b_global * seq_len * H + (t - 1u) * H + h_idx];
      }

      // State update
      float c_new = fg * c_prev + ig * gg;
      float h_new = og * seq_tanh(c_new);

      // Write outputs
      uint out_idx = b_global * seq_len * H + t * H + h_idx;
      out_h[out_idx] = h_new;
      out_c[out_idx] = c_new;

      // Update shared hidden state for next timestep
      shared_h[local_b * H + h_idx] = h_new;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// ====================================================================
// v2: transposed Wh layout (Wh_t [H, 4H]) → coalesced reads.
//
// Each thread tid ∈ [0, 4H) reads Wh_t[k, tid] for k = 0..H-1.
// Consecutive threads read consecutive elements → 32× better coalescing
// than v1 where consecutive threads read rows H apart.
//
// Memory access per simdgroup (32 threads) per k iteration:
//   v1: 32 cache lines (stride-H access)
//   v2: 1 cache line  (contiguous access)
// ====================================================================

[[host_name("lstm_sequence_v2_float")]]
[[kernel]] void lstm_sequence_v2_float(
    const device float* input_proj,   // [B, T, 4H] row-major, contiguous
    const device float* Wh_t,         // [H, 4H] row-major (TRANSPOSED from [4H, H])
    const device float* h_init,       // [B, H]
    const device float* c_init,       // [B, H]
    device float* out_h,              // [B, T, H]
    device float* out_c,              // [B, T, H]
    constant uint& batch_size,
    constant uint& seq_len,
    constant uint& hidden_size,
    threadgroup float* shared_h       [[threadgroup(0)]],  // [H]
    threadgroup float* shared_c       [[threadgroup(1)]],  // [H]
    threadgroup float* shared_gates   [[threadgroup(2)]],  // [4H]
    uint tid        [[thread_position_in_threadgroup]],
    uint batch_idx  [[threadgroup_position_in_grid]]) {

  uint H = hidden_size;
  uint H4 = 4u * H;

  if (batch_idx >= batch_size || tid >= H4)
    return;

  // Initialize shared state (first H threads)
  if (tid < H) {
    shared_h[tid] = h_init[batch_idx * H + tid];
    shared_c[tid] = c_init[batch_idx * H + tid];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint t = 0u; t < seq_len; ++t) {
    // Dot product: sum_k( shared_h[k] * Wh_t[k, tid] )
    //
    // Wh_t is [H, 4H] row-major → Wh_t[k, tid] = Wh_t[k * H4 + tid]
    // For a given k, threads 0..31 read addresses k*H4+0, k*H4+1, ..., k*H4+31
    // → perfectly coalesced (one 128-byte cache line per simdgroup).
    //
    // Pairwise summation with unroll-by-8 for O(log H) rounding error.
    float dot = 0.0f;
    {
      uint k = 0u;
      for (; k + 7u < H; k += 8u) {
        uint base = k * H4 + tid;
        float a0 = shared_h[k+0u] * Wh_t[base];
        float a1 = shared_h[k+1u] * Wh_t[base + H4];
        float a2 = shared_h[k+2u] * Wh_t[base + 2u*H4];
        float a3 = shared_h[k+3u] * Wh_t[base + 3u*H4];
        float a4 = shared_h[k+4u] * Wh_t[base + 4u*H4];
        float a5 = shared_h[k+5u] * Wh_t[base + 5u*H4];
        float a6 = shared_h[k+6u] * Wh_t[base + 6u*H4];
        float a7 = shared_h[k+7u] * Wh_t[base + 7u*H4];
        dot += (a0 + a1) + (a2 + a3) + (a4 + a5) + (a6 + a7);
      }
      // Tail (H not divisible by 8)
      for (; k < H; ++k) {
        dot += shared_h[k] * Wh_t[k * H4 + tid];
      }
    }

    // Add input projection (also coalesced: threads read consecutive elements)
    uint x_base = batch_idx * seq_len * H4 + t * H4;
    dot += input_proj[x_base + tid];

    // Write to shared gates
    shared_gates[tid] = dot;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // First H threads: apply gate activations + update state
    if (tid < H) {
      float ig = seq_sigmoid(shared_gates[tid]);
      float fg = seq_sigmoid(shared_gates[H + tid]);
      float gg = seq_tanh(shared_gates[2u * H + tid]);
      float og = seq_sigmoid(shared_gates[3u * H + tid]);

      float c_new = fg * shared_c[tid] + ig * gg;
      float h_new = og * seq_tanh(c_new);

      uint out_idx = batch_idx * seq_len * H + t * H + tid;
      out_h[out_idx] = h_new;
      out_c[out_idx] = c_new;

      shared_h[tid] = h_new;
      shared_c[tid] = c_new;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}
