// Copyright © 2026 Apple Inc.

#include <metal_simdgroup>
#include <metal_stdlib>

using namespace metal;

inline float tq_acc_scalar(
    uint d,
    uint vals_per_word,
    uint bits,
    uint mask,
    uint kp_base,
    const device uint* k_packed,
    const device float* q,
    const device float* centroids,
    uint q_base) {
  uint word = d / vals_per_word;
  uint shift = (d % vals_per_word) * bits;
  uint packed = k_packed[kp_base + word];
  uint idx = (packed >> shift) & mask;
  return q[q_base + d] * centroids[idx];
}

[[host_name("turboquant_qk_decode_scalar")]] [[kernel]] void
turboquant_qk_decode_scalar(
    const device float* q,
    const device uint* k_packed,
    const device float* k_norms,
    const device float* centroids,
    device float* out,
    constant uint& n_queries,
    constant uint& n_keys,
    constant uint& dim,
    constant uint& words_per_key,
    constant uint& vals_per_word,
    constant uint& bits,
    constant uint& mask,
    uint2 tid [[thread_position_in_grid]]) {
  uint n = tid.x;
  uint m = tid.y;
  if (n >= n_keys || m >= n_queries) {
    return;
  }

  float acc = 0.0f;
  uint q_base = m * dim;
  uint kp_base = n * words_per_key;
  for (uint d = 0; d < dim; ++d) {
    acc +=
        tq_acc_scalar(d, vals_per_word, bits, mask, kp_base, k_packed, q, centroids, q_base);
  }
  out[m * n_keys + n] = acc * k_norms[n];
}

template <uint UNROLL>
inline void turboquant_qk_decode_simd_impl(
    const device float* q,
    const device uint* k_packed,
    const device float* k_norms,
    const device float* centroids,
    device float* out,
    constant uint& n_queries,
    constant uint& n_keys,
    constant uint& dim,
    constant uint& words_per_key,
    constant uint& vals_per_word,
    constant uint& bits,
    constant uint& mask,
    uint2 tid [[thread_position_in_grid]]) {
  uint x = tid.x;
  uint m = tid.y;
  uint lane = x & 31u;
  uint n = x >> 5;

  if (n >= n_keys || m >= n_queries) {
    return;
  }

  float acc = 0.0f;
  uint q_base = m * dim;
  uint kp_base = n * words_per_key;

  for (uint d = lane; d < dim; d += 32u * UNROLL) {
    if constexpr (UNROLL >= 1) {
      uint d0 = d;
      if (d0 < dim) {
        acc += tq_acc_scalar(
            d0, vals_per_word, bits, mask, kp_base, k_packed, q, centroids, q_base);
      }
    }
    if constexpr (UNROLL >= 2) {
      uint d1 = d + 32u;
      if (d1 < dim) {
        acc += tq_acc_scalar(
            d1, vals_per_word, bits, mask, kp_base, k_packed, q, centroids, q_base);
      }
    }
    if constexpr (UNROLL >= 4) {
      uint d2 = d + 64u;
      uint d3 = d + 96u;
      if (d2 < dim) {
        acc += tq_acc_scalar(
            d2, vals_per_word, bits, mask, kp_base, k_packed, q, centroids, q_base);
      }
      if (d3 < dim) {
        acc += tq_acc_scalar(
            d3, vals_per_word, bits, mask, kp_base, k_packed, q, centroids, q_base);
      }
    }
  }

  float sum = simd_sum(acc);
  if (lane == 0u) {
    out[m * n_keys + n] = sum * k_norms[n];
  }
}

[[host_name("turboquant_qk_decode_simd_u1")]] [[kernel]] void
turboquant_qk_decode_simd_u1(
    const device float* q,
    const device uint* k_packed,
    const device float* k_norms,
    const device float* centroids,
    device float* out,
    constant uint& n_queries,
    constant uint& n_keys,
    constant uint& dim,
    constant uint& words_per_key,
    constant uint& vals_per_word,
    constant uint& bits,
    constant uint& mask,
    uint2 tid [[thread_position_in_grid]]) {
  turboquant_qk_decode_simd_impl<1>(
      q,
      k_packed,
      k_norms,
      centroids,
      out,
      n_queries,
      n_keys,
      dim,
      words_per_key,
      vals_per_word,
      bits,
      mask,
      tid);
}

[[host_name("turboquant_qk_decode_simd_u2")]] [[kernel]] void
turboquant_qk_decode_simd_u2(
    const device float* q,
    const device uint* k_packed,
    const device float* k_norms,
    const device float* centroids,
    device float* out,
    constant uint& n_queries,
    constant uint& n_keys,
    constant uint& dim,
    constant uint& words_per_key,
    constant uint& vals_per_word,
    constant uint& bits,
    constant uint& mask,
    uint2 tid [[thread_position_in_grid]]) {
  turboquant_qk_decode_simd_impl<2>(
      q,
      k_packed,
      k_norms,
      centroids,
      out,
      n_queries,
      n_keys,
      dim,
      words_per_key,
      vals_per_word,
      bits,
      mask,
      tid);
}

[[host_name("turboquant_qk_decode_simd_u4")]] [[kernel]] void
turboquant_qk_decode_simd_u4(
    const device float* q,
    const device uint* k_packed,
    const device float* k_norms,
    const device float* centroids,
    device float* out,
    constant uint& n_queries,
    constant uint& n_keys,
    constant uint& dim,
    constant uint& words_per_key,
    constant uint& vals_per_word,
    constant uint& bits,
    constant uint& mask,
    uint2 tid [[thread_position_in_grid]]) {
  turboquant_qk_decode_simd_impl<4>(
      q,
      k_packed,
      k_norms,
      centroids,
      out,
      n_queries,
      n_keys,
      dim,
      words_per_key,
      vals_per_word,
      bits,
      mask,
      tid);
}

[[host_name("turboquant_qk_decode_batched_scalar")]] [[kernel]] void
turboquant_qk_decode_batched_scalar(
    const device float* q,
    const device uint* k_packed,
    const device float* k_norms,
    const device float* centroids,
    device float* out,
    constant uint& batch,
    constant uint& n_q_heads,
    constant uint& n_kv_heads,
    constant uint& n_repeats,
    constant uint& n_q_len,
    constant uint& n_keys,
    constant uint& dim,
    constant uint& words_per_key,
    constant uint& vals_per_word,
    constant uint& bits,
    constant uint& mask,
    uint3 tid [[thread_position_in_grid]]) {
  uint n = tid.x; // key index
  uint qlin = tid.y; // flattened [hq, l]
  uint b = tid.z;
  if (n >= n_keys || qlin >= (n_q_heads * n_q_len) || b >= batch) {
    return;
  }

  uint hq = qlin / n_q_len;
  uint l = qlin % n_q_len;
  uint hkv = hq / n_repeats;
  if (hkv >= n_kv_heads) {
    return;
  }

  uint q_base = (((b * n_q_heads + hq) * n_q_len + l) * dim);
  uint kp_base = ((((b * n_kv_heads + hkv) * n_keys + n) * words_per_key));
  uint norm_idx = ((b * n_kv_heads + hkv) * n_keys + n);

  float acc = 0.0f;
  for (uint d = 0; d < dim; ++d) {
    acc += tq_acc_scalar(d, vals_per_word, bits, mask, kp_base, k_packed, q, centroids, q_base);
  }

  uint out_idx = (((b * n_q_heads + hq) * n_q_len + l) * n_keys + n);
  out[out_idx] = acc * k_norms[norm_idx];
}

template <uint UNROLL>
inline void turboquant_qk_decode_batched_simd_impl(
    const device float* q,
    const device uint* k_packed,
    const device float* k_norms,
    const device float* centroids,
    device float* out,
    constant uint& batch,
    constant uint& n_q_heads,
    constant uint& n_kv_heads,
    constant uint& n_repeats,
    constant uint& n_q_len,
    constant uint& n_keys,
    constant uint& dim,
    constant uint& words_per_key,
    constant uint& vals_per_word,
    constant uint& bits,
    constant uint& mask,
    uint3 tid [[thread_position_in_grid]]) {
  uint x = tid.x;
  uint qlin = tid.y;
  uint b = tid.z;

  uint lane = x & 31u;
  uint n = x >> 5;

  if (n >= n_keys || qlin >= (n_q_heads * n_q_len) || b >= batch) {
    return;
  }

  uint hq = qlin / n_q_len;
  uint l = qlin % n_q_len;
  uint hkv = hq / n_repeats;
  if (hkv >= n_kv_heads) {
    return;
  }

  uint q_base = (((b * n_q_heads + hq) * n_q_len + l) * dim);
  uint kp_base = ((((b * n_kv_heads + hkv) * n_keys + n) * words_per_key));
  uint norm_idx = ((b * n_kv_heads + hkv) * n_keys + n);

  float acc = 0.0f;
  for (uint d = lane; d < dim; d += 32u * UNROLL) {
    if constexpr (UNROLL >= 1) {
      uint d0 = d;
      if (d0 < dim) {
        acc += tq_acc_scalar(
            d0, vals_per_word, bits, mask, kp_base, k_packed, q, centroids, q_base);
      }
    }
    if constexpr (UNROLL >= 2) {
      uint d1 = d + 32u;
      if (d1 < dim) {
        acc += tq_acc_scalar(
            d1, vals_per_word, bits, mask, kp_base, k_packed, q, centroids, q_base);
      }
    }
    if constexpr (UNROLL >= 4) {
      uint d2 = d + 64u;
      uint d3 = d + 96u;
      if (d2 < dim) {
        acc += tq_acc_scalar(
            d2, vals_per_word, bits, mask, kp_base, k_packed, q, centroids, q_base);
      }
      if (d3 < dim) {
        acc += tq_acc_scalar(
            d3, vals_per_word, bits, mask, kp_base, k_packed, q, centroids, q_base);
      }
    }
  }

  float sum = simd_sum(acc);
  if (lane == 0u) {
    uint out_idx = (((b * n_q_heads + hq) * n_q_len + l) * n_keys + n);
    out[out_idx] = sum * k_norms[norm_idx];
  }
}

[[host_name("turboquant_qk_decode_batched_simd_u1")]] [[kernel]] void
turboquant_qk_decode_batched_simd_u1(
    const device float* q,
    const device uint* k_packed,
    const device float* k_norms,
    const device float* centroids,
    device float* out,
    constant uint& batch,
    constant uint& n_q_heads,
    constant uint& n_kv_heads,
    constant uint& n_repeats,
    constant uint& n_q_len,
    constant uint& n_keys,
    constant uint& dim,
    constant uint& words_per_key,
    constant uint& vals_per_word,
    constant uint& bits,
    constant uint& mask,
    uint3 tid [[thread_position_in_grid]]) {
  turboquant_qk_decode_batched_simd_impl<1>(
      q,
      k_packed,
      k_norms,
      centroids,
      out,
      batch,
      n_q_heads,
      n_kv_heads,
      n_repeats,
      n_q_len,
      n_keys,
      dim,
      words_per_key,
      vals_per_word,
      bits,
      mask,
      tid);
}

[[host_name("turboquant_qk_decode_batched_simd_u2")]] [[kernel]] void
turboquant_qk_decode_batched_simd_u2(
    const device float* q,
    const device uint* k_packed,
    const device float* k_norms,
    const device float* centroids,
    device float* out,
    constant uint& batch,
    constant uint& n_q_heads,
    constant uint& n_kv_heads,
    constant uint& n_repeats,
    constant uint& n_q_len,
    constant uint& n_keys,
    constant uint& dim,
    constant uint& words_per_key,
    constant uint& vals_per_word,
    constant uint& bits,
    constant uint& mask,
    uint3 tid [[thread_position_in_grid]]) {
  turboquant_qk_decode_batched_simd_impl<2>(
      q,
      k_packed,
      k_norms,
      centroids,
      out,
      batch,
      n_q_heads,
      n_kv_heads,
      n_repeats,
      n_q_len,
      n_keys,
      dim,
      words_per_key,
      vals_per_word,
      bits,
      mask,
      tid);
}

[[host_name("turboquant_qk_decode_batched_simd_u4")]] [[kernel]] void
turboquant_qk_decode_batched_simd_u4(
    const device float* q,
    const device uint* k_packed,
    const device float* k_norms,
    const device float* centroids,
    device float* out,
    constant uint& batch,
    constant uint& n_q_heads,
    constant uint& n_kv_heads,
    constant uint& n_repeats,
    constant uint& n_q_len,
    constant uint& n_keys,
    constant uint& dim,
    constant uint& words_per_key,
    constant uint& vals_per_word,
    constant uint& bits,
    constant uint& mask,
    uint3 tid [[thread_position_in_grid]]) {
  turboquant_qk_decode_batched_simd_impl<4>(
      q,
      k_packed,
      k_norms,
      centroids,
      out,
      batch,
      n_q_heads,
      n_kv_heads,
      n_repeats,
      n_q_len,
      n_keys,
      dim,
      words_per_key,
      vals_per_word,
      bits,
      mask,
      tid);
}
