import os
import unittest

import mlx.core as mx

import mlx_tests

from mlx.turboquant_fast import dequant_reference, pack_indices


class TestTurboQuantNativeFast(mlx_tests.MLXTestCase):
    def _wht_signs(self, dim):
        raw = mx.random.normal((dim,), dtype=mx.float32)
        return mx.where(raw >= 0, 1.0, -1.0).astype(mx.float32)

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
    def test_native_qjl_score_batched_matches_reference(self):
        B, Hkv, R, T, D = 1, 2, 5, 11, 64

        q_proj = mx.random.normal((B, Hkv, R, D), dtype=mx.float32)
        qjl_idx = mx.random.randint(0, 2, (B, Hkv, T, D), dtype=mx.uint32)
        k_norms = mx.abs(mx.random.normal((B, Hkv, T), dtype=mx.float32)) + 0.1
        qjl_gamma = mx.abs(mx.random.normal((B, Hkv, T), dtype=mx.float32)) + 0.05

        qjl_packed = pack_indices(qjl_idx.reshape(-1, D), 1).reshape(B, Hkv, T, -1)
        corr_native = mx.fast.turboquant_qjl_score_batched(
            q_proj, k_norms, qjl_gamma, qjl_packed
        )

        qjl_signs = mx.where(qjl_idx > 0, 1.0, -1.0).astype(mx.float32)
        alpha = (mx.pi / 2.0) ** 0.5 / D
        corr_ref = (q_proj @ mx.swapaxes(qjl_signs, -1, -2)) * mx.expand_dims(
            alpha * k_norms * qjl_gamma, axis=2
        )

        self.assertTrue(mx.allclose(corr_native, corr_ref, rtol=3e-4, atol=3e-4))

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

    @unittest.skipIf(not mx.is_available(mx.gpu), "No GPU available")
    def test_native_prod_qk_batched_matches_reference_wht(self):
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
        projection_signs = self._wht_signs(D)

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
            projection_signs,
            n_repeats,
        )

        k_deq = mx.take(centroids, k_idx, axis=0) * mx.expand_dims(k_norms, axis=-1)
        qjl_signs = mx.where(qjl_idx > 0, 1.0, -1.0).astype(mx.float32)
        alpha = (mx.pi / 2.0) ** 0.5 / D
        corr_deq = (
            alpha
            * mx.expand_dims(k_norms * qjl_gamma, axis=-1)
            * (mx.hadamard_transform(qjl_signs) * projection_signs)
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

    @unittest.skipIf(not mx.is_available(mx.gpu), "No GPU available")
    def test_native_batched_qk_lut_matches_reference_gqa256(self):
        prev_lut = os.environ.get("MLX_TQ_QK_CENTROID_LUT")
        prev_block = os.environ.get("MLX_TQ_BLOCK")
        os.environ["MLX_TQ_QK_CENTROID_LUT"] = "1"
        os.environ["MLX_TQ_BLOCK"] = "8"
        try:
            bits = 3
            levels = 1 << bits
            B, Hkv, n_repeats, L, T, D = 1, 2, 8, 1, 257, 256
            Hq = Hkv * n_repeats

            q_rot = mx.random.normal((B, Hq, L, D), dtype=mx.float32)
            k_idx = mx.random.randint(0, levels, (B, Hkv, T, D), dtype=mx.uint32)
            k_norms = mx.abs(mx.random.normal((B, Hkv, T), dtype=mx.float32)) + 0.1
            centroids = mx.linspace(-1.0, 1.0, levels, dtype=mx.float32)

            k_packed = pack_indices(k_idx.reshape(-1, D), bits).reshape(B, Hkv, T, -1)
            scores_native = mx.fast.turboquant_qk_packed_scores_batched(
                q_rot, k_packed, k_norms, centroids, bits, n_repeats
            )

            k_deq = mx.take(centroids, k_idx, axis=0) * mx.expand_dims(k_norms, axis=-1)
            qg = q_rot.reshape(B, Hkv, n_repeats, L, D)
            scores_ref = (qg @ mx.expand_dims(mx.swapaxes(k_deq, -1, -2), axis=2)).reshape(
                B, Hq, L, T
            )

            self.assertTrue(mx.allclose(scores_native, scores_ref, rtol=3e-4, atol=3e-4))
        finally:
            if prev_lut is None:
                os.environ.pop("MLX_TQ_QK_CENTROID_LUT", None)
            else:
                os.environ["MLX_TQ_QK_CENTROID_LUT"] = prev_lut
            if prev_block is None:
                os.environ.pop("MLX_TQ_BLOCK", None)
            else:
                os.environ["MLX_TQ_BLOCK"] = prev_block

    @unittest.skipIf(not mx.is_available(mx.gpu), "No GPU available")
    def test_native_prod_decode_attention_batched_matches_reference(self):
        k_bits = 2
        v_bits = 3
        k_levels = 1 << k_bits
        v_levels = 1 << v_bits
        B, Hkv, n_repeats, L, T, D = 1, 2, 2, 1, 11, 64
        Hq = Hkv * n_repeats

        q_rot = mx.random.normal((B, Hq, L, D), dtype=mx.float32)
        q_model = mx.random.normal((B, Hq, L, D), dtype=mx.float32)
        k_idx = mx.random.randint(0, k_levels, (B, Hkv, T, D), dtype=mx.uint32)
        qjl_idx = mx.random.randint(0, 2, (B, Hkv, T, D), dtype=mx.uint32)
        v_idx = mx.random.randint(0, v_levels, (B, Hkv, T, D), dtype=mx.uint32)
        k_norms = mx.abs(mx.random.normal((B, Hkv, T), dtype=mx.float32)) + 0.1
        qjl_gamma = mx.abs(mx.random.normal((B, Hkv, T), dtype=mx.float32)) + 0.05
        v_norms = mx.abs(mx.random.normal((B, Hkv, T), dtype=mx.float32)) + 0.1
        k_centroids = mx.linspace(-1.0, 1.0, k_levels, dtype=mx.float32)
        v_centroids = mx.linspace(-1.0, 1.0, v_levels, dtype=mx.float32)
        projection = mx.random.normal((D, D), dtype=mx.float32)

        k_packed = pack_indices(k_idx.reshape(-1, D), k_bits).reshape(B, Hkv, T, -1)
        qjl_packed = pack_indices(qjl_idx.reshape(-1, D), 1).reshape(B, Hkv, T, -1)
        v_packed = pack_indices(v_idx.reshape(-1, D), v_bits).reshape(B, Hkv, T, -1)

        out_native = mx.fast.turboquant_decode_attention_prod_batched(
            q_rot,
            q_model,
            k_packed,
            k_norms,
            k_centroids,
            k_bits,
            qjl_packed,
            qjl_gamma,
            projection,
            v_packed,
            v_norms,
            v_centroids,
            v_bits,
            n_repeats,
            D,
        )

        k_mse = mx.take(k_centroids, k_idx, axis=0) * mx.expand_dims(k_norms, axis=-1)
        qjl_signs = mx.where(qjl_idx > 0, 1.0, -1.0).astype(mx.float32)
        alpha = (mx.pi / 2.0) ** 0.5 / D
        k_corr = (
            alpha
            * mx.expand_dims(k_norms * qjl_gamma, axis=-1)
            * (qjl_signs @ projection)
        )
        v_deq = mx.take(v_centroids, v_idx, axis=0) * mx.expand_dims(v_norms, axis=-1)

        qg_rot = q_rot.reshape(B, Hkv, n_repeats, L, D)
        qg_model = q_model.reshape(B, Hkv, n_repeats, L, D)
        mse_scores = (qg_rot @ mx.expand_dims(mx.swapaxes(k_mse, -1, -2), axis=2)).reshape(
            B, Hq, L, T
        )
        corr_scores = (
            qg_model @ mx.expand_dims(mx.swapaxes(k_corr, -1, -2), axis=2)
        ).reshape(B, Hq, L, T)
        probs = mx.softmax(mse_scores + corr_scores, axis=-1, precise=True)
        out_ref = (probs.reshape(B, Hkv, n_repeats, L, T) @ mx.expand_dims(v_deq, axis=2)).reshape(
            B, Hq, L, D
        )

        self.assertTrue(mx.allclose(out_native, out_ref, rtol=3e-4, atol=3e-4))

    @unittest.skipIf(not mx.is_available(mx.gpu), "No GPU available")
    def test_native_decode_attention_model_batched_matches_wrapper(self):
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
        rotation = mx.random.normal((D, D), dtype=mx.float32)

        k_packed = pack_indices(k_idx.reshape(-1, D), bits).reshape(B, Hkv, T, -1)
        v_packed = pack_indices(v_idx.reshape(-1, D), bits).reshape(B, Hkv, T, -1)

        out_rot = mx.fast.turboquant_decode_attention_packed_batched(
            q_rot, k_packed, k_norms, v_packed, v_norms, centroids, bits, n_repeats, D
        )
        out_model = mx.fast.turboquant_decode_attention_packed_model_batched(
            q_rot,
            k_packed,
            k_norms,
            v_packed,
            v_norms,
            centroids,
            bits,
            n_repeats,
            D,
            rotation,
        )

        out_ref = out_rot @ rotation
        self.assertTrue(mx.allclose(out_model, out_ref, rtol=3e-4, atol=3e-4))

    @unittest.skipIf(not mx.is_available(mx.gpu), "No GPU available")
    def test_native_prod_decode_attention_model_batched_matches_wrapper(self):
        k_bits = 2
        v_bits = 3
        k_levels = 1 << k_bits
        v_levels = 1 << v_bits
        B, Hkv, n_repeats, L, T, D = 1, 2, 2, 1, 11, 64
        Hq = Hkv * n_repeats

        q_rot = mx.random.normal((B, Hq, L, D), dtype=mx.float32)
        q_model = mx.random.normal((B, Hq, L, D), dtype=mx.float32)
        k_idx = mx.random.randint(0, k_levels, (B, Hkv, T, D), dtype=mx.uint32)
        qjl_idx = mx.random.randint(0, 2, (B, Hkv, T, D), dtype=mx.uint32)
        v_idx = mx.random.randint(0, v_levels, (B, Hkv, T, D), dtype=mx.uint32)
        k_norms = mx.abs(mx.random.normal((B, Hkv, T), dtype=mx.float32)) + 0.1
        qjl_gamma = mx.abs(mx.random.normal((B, Hkv, T), dtype=mx.float32)) + 0.05
        v_norms = mx.abs(mx.random.normal((B, Hkv, T), dtype=mx.float32)) + 0.1
        k_centroids = mx.linspace(-1.0, 1.0, k_levels, dtype=mx.float32)
        v_centroids = mx.linspace(-1.0, 1.0, v_levels, dtype=mx.float32)
        projection = mx.random.normal((D, D), dtype=mx.float32)
        rotation = mx.random.normal((D, D), dtype=mx.float32)

        k_packed = pack_indices(k_idx.reshape(-1, D), k_bits).reshape(B, Hkv, T, -1)
        qjl_packed = pack_indices(qjl_idx.reshape(-1, D), 1).reshape(B, Hkv, T, -1)
        v_packed = pack_indices(v_idx.reshape(-1, D), v_bits).reshape(B, Hkv, T, -1)

        out_rot = mx.fast.turboquant_decode_attention_prod_batched(
            q_rot,
            q_model,
            k_packed,
            k_norms,
            k_centroids,
            k_bits,
            qjl_packed,
            qjl_gamma,
            projection,
            v_packed,
            v_norms,
            v_centroids,
            v_bits,
            n_repeats,
            D,
        )
        out_model = mx.fast.turboquant_decode_attention_prod_model_batched(
            q_rot,
            q_model,
            k_packed,
            k_norms,
            k_centroids,
            k_bits,
            qjl_packed,
            qjl_gamma,
            projection,
            v_packed,
            v_norms,
            v_centroids,
            v_bits,
            n_repeats,
            D,
            rotation,
        )

        out_ref = out_rot @ rotation
        self.assertTrue(mx.allclose(out_model, out_ref, rtol=3e-4, atol=3e-4))

    @unittest.skipIf(not mx.is_available(mx.gpu), "No GPU available")
    def test_native_prod_decode_attention_batched_matches_reference_wht(self):
        k_bits = 2
        v_bits = 3
        k_levels = 1 << k_bits
        v_levels = 1 << v_bits
        B, Hkv, n_repeats, L, T, D = 1, 2, 2, 1, 11, 64
        Hq = Hkv * n_repeats

        q_rot = mx.random.normal((B, Hq, L, D), dtype=mx.float32)
        q_model = mx.random.normal((B, Hq, L, D), dtype=mx.float32)
        k_idx = mx.random.randint(0, k_levels, (B, Hkv, T, D), dtype=mx.uint32)
        qjl_idx = mx.random.randint(0, 2, (B, Hkv, T, D), dtype=mx.uint32)
        v_idx = mx.random.randint(0, v_levels, (B, Hkv, T, D), dtype=mx.uint32)
        k_norms = mx.abs(mx.random.normal((B, Hkv, T), dtype=mx.float32)) + 0.1
        qjl_gamma = mx.abs(mx.random.normal((B, Hkv, T), dtype=mx.float32)) + 0.05
        v_norms = mx.abs(mx.random.normal((B, Hkv, T), dtype=mx.float32)) + 0.1
        k_centroids = mx.linspace(-1.0, 1.0, k_levels, dtype=mx.float32)
        v_centroids = mx.linspace(-1.0, 1.0, v_levels, dtype=mx.float32)
        projection_signs = self._wht_signs(D)

        k_packed = pack_indices(k_idx.reshape(-1, D), k_bits).reshape(B, Hkv, T, -1)
        qjl_packed = pack_indices(qjl_idx.reshape(-1, D), 1).reshape(B, Hkv, T, -1)
        v_packed = pack_indices(v_idx.reshape(-1, D), v_bits).reshape(B, Hkv, T, -1)

        out_native = mx.fast.turboquant_decode_attention_prod_batched(
            q_rot,
            q_model,
            k_packed,
            k_norms,
            k_centroids,
            k_bits,
            qjl_packed,
            qjl_gamma,
            projection_signs,
            v_packed,
            v_norms,
            v_centroids,
            v_bits,
            n_repeats,
            D,
        )

        k_mse = mx.take(k_centroids, k_idx, axis=0) * mx.expand_dims(k_norms, axis=-1)
        qjl_signs = mx.where(qjl_idx > 0, 1.0, -1.0).astype(mx.float32)
        alpha = (mx.pi / 2.0) ** 0.5 / D
        k_corr = (
            alpha
            * mx.expand_dims(k_norms * qjl_gamma, axis=-1)
            * (mx.hadamard_transform(qjl_signs) * projection_signs)
        )
        v_deq = mx.take(v_centroids, v_idx, axis=0) * mx.expand_dims(v_norms, axis=-1)

        qg_rot = q_rot.reshape(B, Hkv, n_repeats, L, D)
        qg_model = q_model.reshape(B, Hkv, n_repeats, L, D)
        mse_scores = (qg_rot @ mx.expand_dims(mx.swapaxes(k_mse, -1, -2), axis=2)).reshape(
            B, Hq, L, T
        )
        corr_scores = (
            qg_model @ mx.expand_dims(mx.swapaxes(k_corr, -1, -2), axis=2)
        ).reshape(B, Hq, L, T)
        probs = mx.softmax(mse_scores + corr_scores, axis=-1, precise=True)
        out_ref = (
            probs.reshape(B, Hkv, n_repeats, L, T) @ mx.expand_dims(v_deq, axis=2)
        ).reshape(B, Hq, L, D)

        self.assertTrue(mx.allclose(out_native, out_ref, rtol=3e-4, atol=3e-4))



if __name__ == "__main__":
    mlx_tests.MLXTestRunner()
