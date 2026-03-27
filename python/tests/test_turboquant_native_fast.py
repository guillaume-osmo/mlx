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


if __name__ == "__main__":
    mlx_tests.MLXTestRunner()
