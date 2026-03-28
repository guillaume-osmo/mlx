import unittest

import mlx.core as mx

import mlx_tests

from mlx.turboquant_fast import dequant_reference, pack_indices


class TestTurboQuantNativeFast(mlx_tests.MLXTestCase):
    @unittest.skipIf(not mx.is_available(mx.gpu), "No GPU available")
    def test_native_fast_matches_reference(self):
        bits = 3
        n_queries = 7
        n_keys = 17
        dim = 64
        levels = 1 << bits

        q_rot = mx.random.normal((n_queries, dim), dtype=mx.float32)
        k_idx = mx.random.randint(0, levels, (n_keys, dim), dtype=mx.uint32)
        k_norms = mx.abs(mx.random.normal((n_keys,), dtype=mx.float32)) + 0.1
        centroids = mx.linspace(-1.0, 1.0, levels, dtype=mx.float32)

        k_packed = pack_indices(k_idx, bits)
        scores_native = mx.fast.turboquant_qk_packed_scores(
            q_rot, k_packed, k_norms, centroids, bits
        )

        k_deq = dequant_reference(k_idx, k_norms, centroids)
        scores_ref = q_rot @ k_deq.T

        self.assertTrue(mx.allclose(scores_native, scores_ref, rtol=1e-4, atol=1e-4))


    @unittest.skipIf(not mx.is_available(mx.gpu), "No GPU available")
    def test_native_av_batched_matches_reference(self):
        bits = 3
        levels = 1 << bits
        B, Hkv, n_repeats, L, T, D = 2, 3, 2, 2, 11, 64
        Hq = Hkv * n_repeats

        probs = mx.random.uniform(shape=(B, Hq, L, T), dtype=mx.float32)
        probs = probs / mx.sum(probs, axis=-1, keepdims=True)

        v_idx = mx.random.randint(0, levels, (B, Hkv, T, D), dtype=mx.uint32)
        v_norms = mx.abs(mx.random.normal((B, Hkv, T), dtype=mx.float32)) + 0.1
        centroids = mx.linspace(-1.0, 1.0, levels, dtype=mx.float32)

        v_packed = pack_indices(v_idx.reshape(-1, D), bits).reshape(B, Hkv, T, -1)
        out_native = mx.fast.turboquant_av_packed_values_batched(
            probs, v_packed, v_norms, centroids, bits, n_repeats, D
        )

        v_deq = mx.take(centroids, v_idx, axis=0) * mx.expand_dims(v_norms, axis=-1)
        probs_g = probs.reshape(B, Hkv, n_repeats, L, T)
        out_ref = (probs_g @ mx.expand_dims(v_deq, axis=2)).reshape(B, Hq, L, D)

        self.assertTrue(mx.allclose(out_native, out_ref, rtol=2e-4, atol=2e-4))

    @unittest.skipIf(not mx.is_available(mx.gpu), "No GPU available")
    def test_native_decode_attention_batched_matches_reference(self):
        bits = 4
        levels = 1 << bits
        B, Hkv, n_repeats, L, T, D = 1, 2, 2, 1, 13, 64
        Hq = Hkv * n_repeats

        q_rot = mx.random.normal((B, Hq, L, D), dtype=mx.float32)
        k_idx = mx.random.randint(0, levels, (B, Hkv, T, D), dtype=mx.uint32)
        v_idx = mx.random.randint(0, levels, (B, Hkv, T, D), dtype=mx.uint32)
        k_norms = mx.abs(mx.random.normal((B, Hkv, T), dtype=mx.float32)) + 0.1
        v_norms = mx.abs(mx.random.normal((B, Hkv, T), dtype=mx.float32)) + 0.1
        centroids = mx.linspace(-1.0, 1.0, levels, dtype=mx.float32)

        k_packed = pack_indices(k_idx.reshape(-1, D), bits).reshape(B, Hkv, T, -1)
        v_packed = pack_indices(v_idx.reshape(-1, D), bits).reshape(B, Hkv, T, -1)

        out_native = mx.fast.turboquant_decode_attention_packed_batched(
            q_rot, k_packed, k_norms, v_packed, v_norms, centroids, bits, n_repeats, D
        )

        k_deq = mx.take(centroids, k_idx, axis=0) * mx.expand_dims(k_norms, axis=-1)
        v_deq = mx.take(centroids, v_idx, axis=0) * mx.expand_dims(v_norms, axis=-1)
        qg = q_rot.reshape(B, Hkv, n_repeats, L, D)
        scores = (qg @ mx.expand_dims(mx.swapaxes(k_deq, -1, -2), axis=2)).reshape(
            B, Hq, L, T
        )
        probs = mx.softmax(scores, axis=-1, precise=True)
        out_ref = (probs.reshape(B, Hkv, n_repeats, L, T) @ mx.expand_dims(v_deq, axis=2)).reshape(
            B, Hq, L, D
        )

        self.assertTrue(mx.allclose(out_native, out_ref, rtol=3e-4, atol=3e-4))

    @unittest.skipIf(not mx.is_available(mx.gpu), "No GPU available")
    def test_native_prod_qk_batched_matches_reference(self):
        bits = 2
        levels = 1 << bits
        B, Hkv, n_repeats, L, T, D = 1, 2, 2, 3, 9, 64
        Hq = Hkv * n_repeats

        q_rot = mx.random.normal((B, Hq, L, D), dtype=mx.float32)
        q_model = mx.random.normal((B, Hq, L, D), dtype=mx.float32)
        k_idx = mx.random.randint(0, levels, (B, Hkv, T, D), dtype=mx.uint32)
        k_norms = mx.abs(mx.random.normal((B, Hkv, T), dtype=mx.float32)) + 0.1
        qjl_idx = mx.random.randint(0, 2, (B, Hkv, T, D), dtype=mx.uint32)
        qjl_gamma = mx.abs(mx.random.normal((B, Hkv, T), dtype=mx.float32)) + 0.05
        centroids = mx.linspace(-1.0, 1.0, levels, dtype=mx.float32)
        projection = mx.random.normal((D, D), dtype=mx.float32)

        k_packed = pack_indices(k_idx.reshape(-1, D), bits).reshape(B, Hkv, T, -1)
        qjl_packed = pack_indices(qjl_idx.reshape(-1, D), 1).reshape(B, Hkv, T, -1)

        scores_native = mx.fast.turboquant_qk_prod_scores_batched(
            q_rot,
            q_model,
            k_packed,
            k_norms,
            centroids,
            bits,
            qjl_packed,
            qjl_gamma,
            projection,
            n_repeats,
        )

        k_deq = mx.take(centroids, k_idx, axis=0) * mx.expand_dims(k_norms, axis=-1)
        qjl_signs = mx.where(qjl_idx > 0, 1.0, -1.0).astype(mx.float32)
        alpha = (mx.pi / 2.0) ** 0.5 / D
        corr_deq = (
            alpha
            * mx.expand_dims(k_norms * qjl_gamma, axis=-1)
            * (qjl_signs @ projection)
        )
        qg_rot = q_rot.reshape(B, Hkv, n_repeats, L, D)
        qg_model = q_model.reshape(B, Hkv, n_repeats, L, D)
        mse_scores = (qg_rot @ mx.expand_dims(mx.swapaxes(k_deq, -1, -2), axis=2)).reshape(
            B, Hq, L, T
        )
        corr_scores = (
            qg_model @ mx.expand_dims(mx.swapaxes(corr_deq, -1, -2), axis=2)
        ).reshape(B, Hq, L, T)
        scores_ref = mse_scores + corr_scores

        self.assertTrue(mx.allclose(scores_native, scores_ref, rtol=3e-4, atol=3e-4))



if __name__ == "__main__":
    mlx_tests.MLXTestRunner()
