"""TurboQuant fused Metal kernels (prototype).

This module provides a fused query-key score path that avoids materializing
full dequantized keys. It computes:

    scores = q_rot @ dequant(k_packed).T

where each key row is reconstructed on-the-fly from packed codebook indices and
a per-row norm.

Notes:
- This is a Python-level prototype using ``mx.fast.metal_kernel``.
- It targets Apple Metal and falls back to a reference implementation if Metal
  is unavailable.
- ``q_rot`` is expected to already be in the rotated basis used by PolarQuant.
"""

from __future__ import annotations

from functools import lru_cache
import os

import numpy as np

import mlx.core as mx


def pack_indices(indices: mx.array, bits: int) -> mx.array:
    """Pack per-dimension quantization indices into uint32 words.

    Args:
      indices: ``(n_keys, dim)`` integer indices in ``[0, 2**bits - 1]``.
      bits: Number of bits per index (2, 3, or 4).

    Returns:
      Packed indices of shape ``(n_keys, ceil(dim / (32//bits)))`` and dtype
      ``mx.uint32``.
    """
    if bits not in (2, 3, 4):
        raise ValueError(f"bits must be one of (2, 3, 4), got {bits}")
    idx_np = np.array(indices, dtype=np.uint32, copy=False)
    if idx_np.ndim != 2:
        raise ValueError(f"indices must be 2D, got shape {idx_np.shape}")
    n_keys, dim = idx_np.shape
    vals_per_word = 32 // bits
    n_words = (dim + vals_per_word - 1) // vals_per_word
    out = np.zeros((n_keys, n_words), dtype=np.uint32)
    mask = (1 << bits) - 1
    for k in range(n_keys):
        for d in range(dim):
            word = d // vals_per_word
            shift = (d % vals_per_word) * bits
            out[k, word] |= (idx_np[k, d] & mask) << shift
    return mx.array(out, dtype=mx.uint32)


@lru_cache(maxsize=4)
def _qk_decode_kernel(unroll: int):
    name = f"turboquant_qk_packed_decode_u{unroll}"
    if unroll == 1:
        unrolled = """
                uint d0 = d;
                uint word0 = d0 / (uint)vals_per_word;
                uint shift0 = (d0 % (uint)vals_per_word) * (uint)bits;
                uint packed0 = k_packed[kp_base + word0];
                uint idx0 = (packed0 >> shift0) & (uint)mask;
                acc += (float)q[q_base + d0] * (float)centroids[idx0];
        """
    elif unroll == 2:
        unrolled = """
                uint d0 = d;
                if (d0 < (uint)dim) {
                    uint word0 = d0 / (uint)vals_per_word;
                    uint shift0 = (d0 % (uint)vals_per_word) * (uint)bits;
                    uint packed0 = k_packed[kp_base + word0];
                    uint idx0 = (packed0 >> shift0) & (uint)mask;
                    acc += (float)q[q_base + d0] * (float)centroids[idx0];
                }
                uint d1 = d + 32u;
                if (d1 < (uint)dim) {
                    uint word1 = d1 / (uint)vals_per_word;
                    uint shift1 = (d1 % (uint)vals_per_word) * (uint)bits;
                    uint packed1 = k_packed[kp_base + word1];
                    uint idx1 = (packed1 >> shift1) & (uint)mask;
                    acc += (float)q[q_base + d1] * (float)centroids[idx1];
                }
        """
    else:
        unrolled = """
                uint d0 = d;
                if (d0 < (uint)dim) {
                    uint word0 = d0 / (uint)vals_per_word;
                    uint shift0 = (d0 % (uint)vals_per_word) * (uint)bits;
                    uint packed0 = k_packed[kp_base + word0];
                    uint idx0 = (packed0 >> shift0) & (uint)mask;
                    acc += (float)q[q_base + d0] * (float)centroids[idx0];
                }
                uint d1 = d + 32u;
                if (d1 < (uint)dim) {
                    uint word1 = d1 / (uint)vals_per_word;
                    uint shift1 = (d1 % (uint)vals_per_word) * (uint)bits;
                    uint packed1 = k_packed[kp_base + word1];
                    uint idx1 = (packed1 >> shift1) & (uint)mask;
                    acc += (float)q[q_base + d1] * (float)centroids[idx1];
                }
                uint d2 = d + 64u;
                if (d2 < (uint)dim) {
                    uint word2 = d2 / (uint)vals_per_word;
                    uint shift2 = (d2 % (uint)vals_per_word) * (uint)bits;
                    uint packed2 = k_packed[kp_base + word2];
                    uint idx2 = (packed2 >> shift2) & (uint)mask;
                    acc += (float)q[q_base + d2] * (float)centroids[idx2];
                }
                uint d3 = d + 96u;
                if (d3 < (uint)dim) {
                    uint word3 = d3 / (uint)vals_per_word;
                    uint shift3 = (d3 % (uint)vals_per_word) * (uint)bits;
                    uint packed3 = k_packed[kp_base + word3];
                    uint idx3 = (packed3 >> shift3) & (uint)mask;
                    acc += (float)q[q_base + d3] * (float)centroids[idx3];
                }
        """

    source = f"""
        uint x = thread_position_in_grid.x;
        uint m = thread_position_in_grid.y;
        uint lane = x & 31u;
        uint n = x >> 5;

        if (n >= (uint)n_keys || m >= (uint)q_shape[0]) return;

        float acc = 0.0f;
        uint q_base = m * (uint)dim;
        uint kp_base = n * (uint)words_per_key;

        for (uint d = lane; d < (uint)dim; d += {32 * unroll}u) {{
{unrolled}
        }}

        float sum = simd_sum(acc);
        if (lane == 0u) {{
            out[m * (uint)n_keys + n] = sum * (float)k_norms[n];
        }}
    """
    return mx.fast.metal_kernel(
        name=name,
        input_names=[
            "q",
            "k_packed",
            "k_norms",
            "centroids",
            "n_keys",
            "dim",
            "words_per_key",
            "vals_per_word",
            "bits",
            "mask",
        ],
        output_names=["out"],
        source=source,
    )


@lru_cache(maxsize=1)
def _qk_prefill_kernel():
    return mx.fast.metal_kernel(
        name="turboquant_qk_packed_prefill",
        input_names=[
            "q",
            "k_packed",
            "k_norms",
            "centroids",
            "n_keys",
            "dim",
            "words_per_key",
            "vals_per_word",
            "bits",
            "mask",
        ],
        output_names=["out"],
        source=r"""
            uint n = thread_position_in_grid.x;
            uint m = thread_position_in_grid.y;
            if (n >= (uint)n_keys || m >= (uint)q_shape[0]) return;

            float acc = 0.0f;
            uint q_base = m * (uint)dim;
            uint kp_base = n * (uint)words_per_key;
            for (uint d = 0; d < (uint)dim; ++d) {
                uint word = d / (uint)vals_per_word;
                uint shift = (d % (uint)vals_per_word) * (uint)bits;
                uint packed = k_packed[kp_base + word];
                uint idx = (packed >> shift) & (uint)mask;
                float z = (float)centroids[idx];
                float qv = (float)q[q_base + d];
                acc += qv * z;
            }
            out[m * (uint)n_keys + n] = acc * (float)k_norms[n];
        """,
    )


def _detect_arch_family() -> str:
    """Best-effort Apple GPU family detection for launch tuning."""
    env = os.environ.get("MLX_TQ_ARCH_HINT", "").strip().lower()
    if env in ("m2", "m3", "m4"):
        return env
    try:
        info = mx.device_info(mx.gpu)
        text = " ".join(str(v) for v in info.values()).lower()
    except Exception:
        return "generic"
    if "m4" in text:
        return "m4"
    if "m3" in text:
        return "m3"
    if "m2" in text:
        return "m2"
    return "generic"


def _decode_unroll_for_arch(dim: int) -> int:
    arch = _detect_arch_family()
    if arch == "m4":
        return 4 if dim >= 256 else 2
    if arch == "m3":
        return 2 if dim >= 192 else 1
    if arch == "m2":
        return 2 if dim >= 256 else 1
    return 1


def _prefill_threadgroup_for_shape(n_keys: int) -> tuple[int, int, int]:
    """Architecture-aware launch policy for prefill path."""
    arch = _detect_arch_family()
    if n_keys >= 4096:
        return (16, 8, 1) if arch in ("m3", "m4") else (8, 8, 1)
    if n_keys >= 2048:
        return (16, 4, 1) if arch in ("m3", "m4") else (8, 8, 1)
    return (8, 8, 1)


def _decode_mode_for_shape(n_keys: int, dim: int) -> str:
    """Select decode kernel mode: scalar or simd."""
    env = os.environ.get("MLX_TQ_DECODE_MODE", "auto").strip().lower()
    if env in ("scalar", "simd"):
        return env

    arch = _detect_arch_family()
    # Current heuristic: SIMD path tends to win only at larger dims / context.
    if arch == "m4" and dim >= 256 and n_keys >= 4096:
        return "simd"
    if arch == "m3" and dim >= 384 and n_keys >= 4096:
        return "simd"
    if arch == "m2" and dim >= 512 and n_keys >= 8192:
        return "simd"
    return "scalar"


def qk_packed_scores(
    q_rot: mx.array,
    k_packed: mx.array,
    k_norms: mx.array,
    centroids: mx.array,
    bits: int,
    *,
    stream=None,
) -> mx.array:
    """Fused TurboQuant-style query-key score matmul.

    Args:
      q_rot: Rotated query matrix with shape ``(n_queries, dim)``.
      k_packed: Packed key indices shape ``(n_keys, words_per_key)``, ``uint32``.
      k_norms: Key norms shape ``(n_keys,)``.
      centroids: Codebook values shape ``(2**bits,)``.
      bits: Bits per code (2, 3, or 4).

    Returns:
      Score matrix shape ``(n_queries, n_keys)`` in ``float32``.
    """
    if bits not in (2, 3, 4):
        raise ValueError(f"bits must be one of (2, 3, 4), got {bits}")
    if q_rot.ndim != 2:
        raise ValueError(f"q_rot must be 2D, got shape {q_rot.shape}")
    if k_packed.ndim != 2:
        raise ValueError(f"k_packed must be 2D, got shape {k_packed.shape}")
    if k_norms.ndim != 1:
        raise ValueError(f"k_norms must be 1D, got shape {k_norms.shape}")
    if centroids.ndim != 1:
        raise ValueError(f"centroids must be 1D, got shape {centroids.shape}")

    n_queries, dim = q_rot.shape
    n_keys = k_packed.shape[0]
    if k_norms.shape[0] != n_keys:
        raise ValueError("k_norms length must match number of keys")
    vals_per_word = 32 // bits
    words_per_key = int(k_packed.shape[1])
    expected_words = (int(dim) + vals_per_word - 1) // vals_per_word
    if words_per_key != expected_words:
        raise ValueError(
            f"k_packed words mismatch: got {words_per_key}, expected {expected_words}"
        )

    if hasattr(mx.fast, "turboquant_qk_packed_scores"):
        return mx.fast.turboquant_qk_packed_scores(
            q_rot.astype(mx.float32),
            k_packed.astype(mx.uint32),
            k_norms.astype(mx.float32),
            centroids.astype(mx.float32),
            int(bits),
            stream=stream if stream is not None else mx.gpu,
        )

    if not mx.metal.is_available():
        # Reference CPU/GPU path: unpack + lookup + matmul.
        mask = (1 << bits) - 1
        kp = np.array(k_packed, dtype=np.uint32, copy=False)
        unpacked = np.zeros((n_keys, dim), dtype=np.uint32)
        for n in range(n_keys):
            for d in range(dim):
                w = d // vals_per_word
                s = (d % vals_per_word) * bits
                unpacked[n, d] = (kp[n, w] >> s) & mask
        z = mx.take(centroids.astype(mx.float32), mx.array(unpacked, dtype=mx.uint32), axis=0)
        k_deq = z * mx.expand_dims(k_norms.astype(mx.float32), axis=-1)
        return q_rot.astype(mx.float32) @ k_deq.T

    inputs = [
        q_rot.astype(mx.float32),
        k_packed.astype(mx.uint32),
        k_norms.astype(mx.float32),
        centroids.astype(mx.float32),
        int(n_keys),
        int(dim),
        int(words_per_key),
        int(vals_per_word),
        int(bits),
        int((1 << bits) - 1),
    ]

    if int(n_queries) == 1 and _decode_mode_for_shape(int(n_keys), int(dim)) == "simd":
        unroll = _decode_unroll_for_arch(int(dim))
        kernel = _qk_decode_kernel(unroll)
        out = kernel(
            inputs=inputs,
            grid=(int(n_keys) * 32, int(n_queries), 1),
            threadgroup=(32, 1, 1),
            output_shapes=[(int(n_queries), int(n_keys))],
            output_dtypes=[mx.float32],
            stream=stream if stream is not None else mx.gpu,
        )[0]
        return out

    kernel = _qk_prefill_kernel()
    out = kernel(
        inputs=inputs,
        grid=(int(n_keys), int(n_queries), 1),
        threadgroup=_prefill_threadgroup_for_shape(int(n_keys)),
        output_shapes=[(int(n_queries), int(n_keys))],
        output_dtypes=[mx.float32],
        stream=stream if stream is not None else mx.gpu,
    )[0]
    return out


def dequant_reference(
    k_indices: mx.array, k_norms: mx.array, centroids: mx.array
) -> mx.array:
    """Reference dequantization for testing."""
    z = mx.take(centroids.astype(mx.float32), k_indices.astype(mx.uint32), axis=0)
    return z * mx.expand_dims(k_norms.astype(mx.float32), axis=-1)
