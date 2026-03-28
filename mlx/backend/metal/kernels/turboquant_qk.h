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



template <uint UNROLL, uint BLOCK>
inline void turboquant_qk_decode_batched_simd_blocked_impl(
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

  uint block_span = 32u * BLOCK;
  uint block_idx = x / block_span;
  uint within_block = x % block_span;
  uint local_key = within_block >> 5;
  uint lane = within_block & 31u;
  uint n = block_idx * BLOCK + local_key;

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

#define DEFINE_TQ_QK_BLOCK_WRAPPER(UNROLL, BLOCK)                               \
  [[host_name("turboquant_qk_decode_batched_simd_u" #UNROLL "_b" #BLOCK)]]     \
  [[kernel]] void turboquant_qk_decode_batched_simd_u##UNROLL##_b##BLOCK(      \
      const device float* q,                                                    \
      const device uint* k_packed,                                              \
      const device float* k_norms,                                              \
      const device float* centroids,                                            \
      device float* out,                                                        \
      constant uint& batch,                                                     \
      constant uint& n_q_heads,                                                 \
      constant uint& n_kv_heads,                                                \
      constant uint& n_repeats,                                                 \
      constant uint& n_q_len,                                                   \
      constant uint& n_keys,                                                    \
      constant uint& dim,                                                       \
      constant uint& words_per_key,                                             \
      constant uint& vals_per_word,                                             \
      constant uint& bits,                                                      \
      constant uint& mask,                                                      \
      uint3 tid [[thread_position_in_grid]]) {                                  \
    turboquant_qk_decode_batched_simd_blocked_impl<UNROLL, BLOCK>(              \
        q,                                                                      \
        k_packed,                                                               \
        k_norms,                                                                \
        centroids,                                                              \
        out,                                                                    \
        batch,                                                                  \
        n_q_heads,                                                              \
        n_kv_heads,                                                             \
        n_repeats,                                                              \
        n_q_len,                                                                \
        n_keys,                                                                 \
        dim,                                                                    \
        words_per_key,                                                          \
        vals_per_word,                                                          \
        bits,                                                                   \
        mask,                                                                   \
        tid);                                                                   \
  }

DEFINE_TQ_QK_BLOCK_WRAPPER(1, 8)
DEFINE_TQ_QK_BLOCK_WRAPPER(1, 16)
DEFINE_TQ_QK_BLOCK_WRAPPER(1, 32)
DEFINE_TQ_QK_BLOCK_WRAPPER(2, 8)
DEFINE_TQ_QK_BLOCK_WRAPPER(2, 16)
DEFINE_TQ_QK_BLOCK_WRAPPER(2, 32)
DEFINE_TQ_QK_BLOCK_WRAPPER(4, 8)
DEFINE_TQ_QK_BLOCK_WRAPPER(4, 16)
DEFINE_TQ_QK_BLOCK_WRAPPER(4, 32)

#undef DEFINE_TQ_QK_BLOCK_WRAPPER
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


inline float tq_av_acc_scalar(
    uint t,
    uint word,
    uint shift,
    uint mask,
    uint probs_base,
    uint vp_head_base,
    uint norms_base,
    uint words_per_key,
    const device float* probs,
    const device uint* v_packed,
    const device float* v_norms,
    const device float* centroids) {
  float p = probs[probs_base + t];
  uint packed = v_packed[vp_head_base + t * words_per_key + word];
  uint idx = (packed >> shift) & mask;
  float v = centroids[idx] * v_norms[norms_base + t];
  return p * v;
}

[[host_name("turboquant_av_decode_batched_scalar")]] [[kernel]] void
turboquant_av_decode_batched_scalar(
    const device float* probs,
    const device uint* v_packed,
    const device float* v_norms,
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
  uint d = tid.x;
  uint qlin = tid.y; // flattened [hq, l]
  uint b = tid.z;
  if (d >= dim || qlin >= (n_q_heads * n_q_len) || b >= batch) {
    return;
  }

  uint hq = qlin / n_q_len;
  uint l = qlin % n_q_len;
  uint hkv = hq / n_repeats;
  if (hkv >= n_kv_heads) {
    return;
  }

  uint word = d / vals_per_word;
  uint shift = (d % vals_per_word) * bits;

  float acc = 0.0f;
  uint probs_base = (((b * n_q_heads + hq) * n_q_len + l) * n_keys);
  uint vp_head_base = ((b * n_kv_heads + hkv) * n_keys * words_per_key);
  uint norms_base = ((b * n_kv_heads + hkv) * n_keys);

  for (uint t = 0; t < n_keys; ++t) {
    acc += tq_av_acc_scalar(
        t,
        word,
        shift,
        mask,
        probs_base,
        vp_head_base,
        norms_base,
        words_per_key,
        probs,
        v_packed,
        v_norms,
        centroids);
  }

  uint out_idx = (((b * n_q_heads + hq) * n_q_len + l) * dim + d);
  out[out_idx] = acc;
}

template <uint UNROLL, uint BLOCK>
inline void turboquant_av_decode_batched_simd_blocked_impl(
    const device float* probs,
    const device uint* v_packed,
    const device float* v_norms,
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

  uint block_span = 32u * BLOCK;
  uint block_idx = x / block_span;
  uint within_block = x % block_span;
  uint local_dim = within_block >> 5;
  uint lane = within_block & 31u;
  uint d = block_idx * BLOCK + local_dim;

  if (d >= dim || qlin >= (n_q_heads * n_q_len) || b >= batch) {
    return;
  }

  uint hq = qlin / n_q_len;
  uint l = qlin % n_q_len;
  uint hkv = hq / n_repeats;
  if (hkv >= n_kv_heads) {
    return;
  }

  uint word = d / vals_per_word;
  uint shift = (d % vals_per_word) * bits;

  float acc = 0.0f;
  uint probs_base = (((b * n_q_heads + hq) * n_q_len + l) * n_keys);
  uint vp_head_base = ((b * n_kv_heads + hkv) * n_keys * words_per_key);
  uint norms_base = ((b * n_kv_heads + hkv) * n_keys);

  for (uint t = lane; t < n_keys; t += 32u * UNROLL) {
    if constexpr (UNROLL >= 1) {
      uint t0 = t;
      if (t0 < n_keys) {
        acc += tq_av_acc_scalar(
            t0,
            word,
            shift,
            mask,
            probs_base,
            vp_head_base,
            norms_base,
            words_per_key,
            probs,
            v_packed,
            v_norms,
            centroids);
      }
    }
    if constexpr (UNROLL >= 2) {
      uint t1 = t + 32u;
      if (t1 < n_keys) {
        acc += tq_av_acc_scalar(
            t1,
            word,
            shift,
            mask,
            probs_base,
            vp_head_base,
            norms_base,
            words_per_key,
            probs,
            v_packed,
            v_norms,
            centroids);
      }
    }
    if constexpr (UNROLL >= 4) {
      uint t2 = t + 64u;
      uint t3 = t + 96u;
      if (t2 < n_keys) {
        acc += tq_av_acc_scalar(
            t2,
            word,
            shift,
            mask,
            probs_base,
            vp_head_base,
            norms_base,
            words_per_key,
            probs,
            v_packed,
            v_norms,
            centroids);
      }
      if (t3 < n_keys) {
        acc += tq_av_acc_scalar(
            t3,
            word,
            shift,
            mask,
            probs_base,
            vp_head_base,
            norms_base,
            words_per_key,
            probs,
            v_packed,
            v_norms,
            centroids);
      }
    }
  }

  float sum = simd_sum(acc);
  if (lane == 0u) {
    uint out_idx = (((b * n_q_heads + hq) * n_q_len + l) * dim + d);
    out[out_idx] = sum;
  }
}

#define DEFINE_TQ_AV_BLOCK_WRAPPER(UNROLL, BLOCK)                               \
  [[host_name("turboquant_av_decode_batched_simd_u" #UNROLL "_b" #BLOCK)]]     \
  [[kernel]] void turboquant_av_decode_batched_simd_u##UNROLL##_b##BLOCK(      \
      const device float* probs,                                                \
      const device uint* v_packed,                                              \
      const device float* v_norms,                                              \
      const device float* centroids,                                            \
      device float* out,                                                        \
      constant uint& batch,                                                     \
      constant uint& n_q_heads,                                                 \
      constant uint& n_kv_heads,                                                \
      constant uint& n_repeats,                                                 \
      constant uint& n_q_len,                                                   \
      constant uint& n_keys,                                                    \
      constant uint& dim,                                                       \
      constant uint& words_per_key,                                             \
      constant uint& vals_per_word,                                             \
      constant uint& bits,                                                      \
      constant uint& mask,                                                      \
      uint3 tid [[thread_position_in_grid]]) {                                  \
    turboquant_av_decode_batched_simd_blocked_impl<UNROLL, BLOCK>(              \
        probs,                                                                  \
        v_packed,                                                               \
        v_norms,                                                                \
        centroids,                                                              \
        out,                                                                    \
        batch,                                                                  \
        n_q_heads,                                                              \
        n_kv_heads,                                                             \
        n_repeats,                                                              \
        n_q_len,                                                                \
        n_keys,                                                                 \
        dim,                                                                    \
        words_per_key,                                                          \
        vals_per_word,                                                          \
        bits,                                                                   \
        mask,                                                                   \
        tid);                                                                   \
  }

DEFINE_TQ_AV_BLOCK_WRAPPER(1, 8)
DEFINE_TQ_AV_BLOCK_WRAPPER(1, 16)
DEFINE_TQ_AV_BLOCK_WRAPPER(1, 32)
DEFINE_TQ_AV_BLOCK_WRAPPER(2, 8)
DEFINE_TQ_AV_BLOCK_WRAPPER(2, 16)
DEFINE_TQ_AV_BLOCK_WRAPPER(2, 32)
DEFINE_TQ_AV_BLOCK_WRAPPER(4, 8)
DEFINE_TQ_AV_BLOCK_WRAPPER(4, 16)
DEFINE_TQ_AV_BLOCK_WRAPPER(4, 32)

#undef DEFINE_TQ_AV_BLOCK_WRAPPER

inline float tq_lookup_centroid(
    uint d,
    uint vals_per_word,
    uint bits,
    uint mask,
    uint row_base,
    const device uint* packed,
    const device float* centroids) {
  uint word = d / vals_per_word;
  uint shift = (d % vals_per_word) * bits;
  uint packed_word = packed[row_base + word];
  uint idx = (packed_word >> shift) & mask;
  return centroids[idx];
}

template <uint UNROLL>
inline void turboquant_decode_attention_batched_impl(
    const device float* q,
    const device uint* k_packed,
    const device float* k_norms,
    const device uint* v_packed,
    const device float* v_norms,
    const device float* centroids,
    device float* out,
    constant uint& batch,
    constant uint& n_q_heads,
    constant uint& n_kv_heads,
    constant uint& n_repeats,
    constant uint& n_q_len,
    constant uint& n_keys,
    constant uint& q_dim,
    constant uint& value_dim,
    constant uint& k_words_per_key,
    constant uint& v_words_per_key,
    constant uint& vals_per_word,
    constant uint& bits,
    constant uint& mask,
    uint3 tid [[thread_position_in_grid]]) {
  uint lane = tid.x;
  uint qlin = tid.y;
  uint b = tid.z;
  if (lane >= 32u || qlin >= (n_q_heads * n_q_len) || b >= batch) {
    return;
  }

  uint hq = qlin / n_q_len;
  uint l = qlin % n_q_len;
  uint hkv = hq / n_repeats;
  if (hkv >= n_kv_heads) {
    return;
  }

  uint q_base = (((b * n_q_heads + hq) * n_q_len + l) * q_dim);
  uint k_head_base = ((b * n_kv_heads + hkv) * n_keys * k_words_per_key);
  uint v_head_base = ((b * n_kv_heads + hkv) * n_keys * v_words_per_key);
  uint norms_base = ((b * n_kv_heads + hkv) * n_keys);
  uint out_base = (((b * n_q_heads + hq) * n_q_len + l) * value_dim);

  float out0 = 0.0f;
  float out1 = 0.0f;
  float out2 = 0.0f;
  float out3 = 0.0f;
  float running_max = -INFINITY;
  float running_norm = 0.0f;

  uint d0 = lane;
  uint d1 = lane + 32u;
  uint d2 = lane + 64u;
  uint d3 = lane + 96u;

  for (uint t = 0; t < n_keys; ++t) {
    uint kp_base = k_head_base + t * k_words_per_key;
    float score_partial = 0.0f;

    if constexpr (UNROLL >= 1) {
      if (d0 < q_dim) {
        score_partial += tq_acc_scalar(
            d0,
            vals_per_word,
            bits,
            mask,
            kp_base,
            k_packed,
            q,
            centroids,
            q_base);
      }
    }
    if constexpr (UNROLL >= 2) {
      if (d1 < q_dim) {
        score_partial += tq_acc_scalar(
            d1,
            vals_per_word,
            bits,
            mask,
            kp_base,
            k_packed,
            q,
            centroids,
            q_base);
      }
    }
    if constexpr (UNROLL >= 4) {
      if (d2 < q_dim) {
        score_partial += tq_acc_scalar(
            d2,
            vals_per_word,
            bits,
            mask,
            kp_base,
            k_packed,
            q,
            centroids,
            q_base);
      }
      if (d3 < q_dim) {
        score_partial += tq_acc_scalar(
            d3,
            vals_per_word,
            bits,
            mask,
            kp_base,
            k_packed,
            q,
            centroids,
            q_base);
      }
    }

    float score = simd_sum(score_partial) * k_norms[norms_base + t];
    float prev_max = running_max;
    float new_max = max(prev_max, score);
    float alpha = isinf(prev_max) ? 0.0f : exp(prev_max - new_max);
    float beta = exp(score - new_max);
    running_norm = running_norm * alpha + beta;

    uint vp_base = v_head_base + t * v_words_per_key;
    float v_norm = v_norms[norms_base + t];

    if constexpr (UNROLL >= 1) {
      if (d0 < value_dim) {
        out0 = out0 * alpha +
               beta * tq_lookup_centroid(
                          d0, vals_per_word, bits, mask, vp_base, v_packed, centroids) *
                   v_norm;
      }
    }
    if constexpr (UNROLL >= 2) {
      if (d1 < value_dim) {
        out1 = out1 * alpha +
               beta * tq_lookup_centroid(
                          d1, vals_per_word, bits, mask, vp_base, v_packed, centroids) *
                   v_norm;
      }
    }
    if constexpr (UNROLL >= 4) {
      if (d2 < value_dim) {
        out2 = out2 * alpha +
               beta * tq_lookup_centroid(
                          d2, vals_per_word, bits, mask, vp_base, v_packed, centroids) *
                   v_norm;
      }
      if (d3 < value_dim) {
        out3 = out3 * alpha +
               beta * tq_lookup_centroid(
                          d3, vals_per_word, bits, mask, vp_base, v_packed, centroids) *
                   v_norm;
      }
    }

    running_max = new_max;
  }

  float inv_norm = 1.0f / max(running_norm, 1e-8f);
  if constexpr (UNROLL >= 1) {
    if (d0 < value_dim) {
      out[out_base + d0] = out0 * inv_norm;
    }
  }
  if constexpr (UNROLL >= 2) {
    if (d1 < value_dim) {
      out[out_base + d1] = out1 * inv_norm;
    }
  }
  if constexpr (UNROLL >= 4) {
    if (d2 < value_dim) {
      out[out_base + d2] = out2 * inv_norm;
    }
    if (d3 < value_dim) {
      out[out_base + d3] = out3 * inv_norm;
    }
  }
}

[[host_name("turboquant_decode_attention_batched_u1")]] [[kernel]] void
turboquant_decode_attention_batched_u1(
    const device float* q,
    const device uint* k_packed,
    const device float* k_norms,
    const device uint* v_packed,
    const device float* v_norms,
    const device float* centroids,
    device float* out,
    constant uint& batch,
    constant uint& n_q_heads,
    constant uint& n_kv_heads,
    constant uint& n_repeats,
    constant uint& n_q_len,
    constant uint& n_keys,
    constant uint& q_dim,
    constant uint& value_dim,
    constant uint& k_words_per_key,
    constant uint& v_words_per_key,
    constant uint& vals_per_word,
    constant uint& bits,
    constant uint& mask,
    uint3 tid [[thread_position_in_grid]]) {
  turboquant_decode_attention_batched_impl<1>(
      q,
      k_packed,
      k_norms,
      v_packed,
      v_norms,
      centroids,
      out,
      batch,
      n_q_heads,
      n_kv_heads,
      n_repeats,
      n_q_len,
      n_keys,
      q_dim,
      value_dim,
      k_words_per_key,
      v_words_per_key,
      vals_per_word,
      bits,
      mask,
      tid);
}

[[host_name("turboquant_decode_attention_batched_u2")]] [[kernel]] void
turboquant_decode_attention_batched_u2(
    const device float* q,
    const device uint* k_packed,
    const device float* k_norms,
    const device uint* v_packed,
    const device float* v_norms,
    const device float* centroids,
    device float* out,
    constant uint& batch,
    constant uint& n_q_heads,
    constant uint& n_kv_heads,
    constant uint& n_repeats,
    constant uint& n_q_len,
    constant uint& n_keys,
    constant uint& q_dim,
    constant uint& value_dim,
    constant uint& k_words_per_key,
    constant uint& v_words_per_key,
    constant uint& vals_per_word,
    constant uint& bits,
    constant uint& mask,
    uint3 tid [[thread_position_in_grid]]) {
  turboquant_decode_attention_batched_impl<2>(
      q,
      k_packed,
      k_norms,
      v_packed,
      v_norms,
      centroids,
      out,
      batch,
      n_q_heads,
      n_kv_heads,
      n_repeats,
      n_q_len,
      n_keys,
      q_dim,
      value_dim,
      k_words_per_key,
      v_words_per_key,
      vals_per_word,
      bits,
      mask,
      tid);
}

[[host_name("turboquant_decode_attention_batched_u4")]] [[kernel]] void
turboquant_decode_attention_batched_u4(
    const device float* q,
    const device uint* k_packed,
    const device float* k_norms,
    const device uint* v_packed,
    const device float* v_norms,
    const device float* centroids,
    device float* out,
    constant uint& batch,
    constant uint& n_q_heads,
    constant uint& n_kv_heads,
    constant uint& n_repeats,
    constant uint& n_q_len,
    constant uint& n_keys,
    constant uint& q_dim,
    constant uint& value_dim,
    constant uint& k_words_per_key,
    constant uint& v_words_per_key,
    constant uint& vals_per_word,
    constant uint& bits,
    constant uint& mask,
    uint3 tid [[thread_position_in_grid]]) {
  turboquant_decode_attention_batched_impl<4>(
      q,
      k_packed,
      k_norms,
      v_packed,
      v_norms,
      centroids,
      out,
      batch,
      n_q_heads,
      n_kv_heads,
      n_repeats,
      n_q_len,
      n_keys,
      q_dim,
      value_dim,
      k_words_per_key,
      v_words_per_key,
      vals_per_word,
      bits,
      mask,
      tid);
}
