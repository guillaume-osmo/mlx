#!/usr/bin/env python3
"""Auto-tune native TurboQuant decode settings for a target decode shape.

Example:
  /Users/guillaume-osmo/miniconda3/envs/osmo/bin/python benchmarks/python/turboquant_autotune.py \
    --shape 1x8192x128 --bits 3
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from typing import Dict, List, Tuple

import mlx.core as mx

from mlx.turboquant_fast import dequant_reference, pack_indices, qk_packed_scores


TUNABLE_KEYS = [
    "MLX_TQ_NATIVE_MODE",
    "MLX_TQ_SIMD_UNROLL",
    "MLX_TQ_SCALAR_TGX",
    "MLX_TQ_SCALAR_TGY",
    "MLX_TQ_AUTO_MIN_KEYS",
    "MLX_TQ_AUTO_MIN_DIM",
]


def parse_shape(shape: str) -> Tuple[int, int, int]:
    try:
        nq, nk, dim = shape.lower().split("x")
        return int(nq), int(nk), int(dim)
    except Exception as exc:
        raise ValueError(f"Invalid shape '{shape}'. Use format nqxnkxdim.") from exc


def apply_env(cfg: Dict[str, str]) -> None:
    for k in TUNABLE_KEYS:
        os.environ.pop(k, None)
    for k, v in cfg.items():
        os.environ[k] = str(v)


def bench_case(
    q: mx.array,
    kref: mx.array,
    kp: mx.array,
    norms: mx.array,
    centroids: mx.array,
    bits: int,
    warmup: int,
    iters: int,
) -> Tuple[float, float]:
    for _ in range(warmup):
        y_ref = q @ kref.T
        y_native = qk_packed_scores(q, kp, norms, centroids, bits)
        mx.eval(y_ref, y_native)

    t0 = time.perf_counter()
    for _ in range(iters):
        y_ref = q @ kref.T
        mx.eval(y_ref)
    ref_ms = (time.perf_counter() - t0) * 1000.0 / iters

    t0 = time.perf_counter()
    for _ in range(iters):
        y_native = qk_packed_scores(q, kp, norms, centroids, bits)
        mx.eval(y_native)
    native_ms = (time.perf_counter() - t0) * 1000.0 / iters

    return ref_ms, native_ms


def bench_ref_only(q: mx.array, kref: mx.array, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        y = q @ kref.T
        mx.eval(y)
    t0 = time.perf_counter()
    for _ in range(iters):
        y = q @ kref.T
        mx.eval(y)
    return (time.perf_counter() - t0) * 1000.0 / iters


def median(xs: List[float]) -> float:
    ys = sorted(xs)
    n = len(ys)
    if n % 2 == 1:
        return ys[n // 2]
    return 0.5 * (ys[n // 2 - 1] + ys[n // 2])


def make_candidates(nk: int, dim: int) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []

    # Auto mode threshold sweeps.
    for min_keys, min_dim in itertools.product(
        [512, 1024, 2048, 4096, 8192], [64, 128, 192, 256, 384, 512]
    ):
        out.append(
            {
                "MLX_TQ_NATIVE_MODE": "auto",
                "MLX_TQ_AUTO_MIN_KEYS": str(min_keys),
                "MLX_TQ_AUTO_MIN_DIM": str(min_dim),
            }
        )

    # Forced simd sweeps.
    for unroll in [1, 2, 4]:
        out.append(
            {
                "MLX_TQ_NATIVE_MODE": "simd",
                "MLX_TQ_SIMD_UNROLL": str(unroll),
            }
        )

    # Forced scalar sweeps.
    for tgx, tgy in itertools.product([8, 16, 32, 64], [1, 2, 4, 8, 16]):
        out.append(
            {
                "MLX_TQ_NATIVE_MODE": "scalar",
                "MLX_TQ_SCALAR_TGX": str(tgx),
                "MLX_TQ_SCALAR_TGY": str(tgy),
            }
        )

    # A tiny guard candidate.
    out.append({"MLX_TQ_NATIVE_MODE": "auto"})
    return out


def config_to_export(cfg: Dict[str, str]) -> str:
    if not cfg:
        return "# no env vars required"
    parts = [f"{k}={v}" for k, v in sorted(cfg.items())]
    return " ".join(parts)


def main() -> None:
    p = argparse.ArgumentParser("TurboQuant native auto-tuner")
    p.add_argument("--shape", type=str, default="1x8192x128", help="nqxnkxdim")
    p.add_argument("--bits", type=int, default=3, choices=[2, 3, 4])
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--output-json", type=str, default="")
    args = p.parse_args()

    nq, nk, dim = parse_shape(args.shape)
    bits = args.bits
    levels = 1 << bits

    mx.random.seed(args.seed)
    q = mx.random.normal((nq, dim), dtype=mx.float32)
    idx = mx.random.randint(0, levels, (nk, dim), dtype=mx.uint32)
    norms = mx.abs(mx.random.normal((nk,), dtype=mx.float32)) + 0.1
    centroids = mx.linspace(-1.0, 1.0, levels, dtype=mx.float32)
    kp = pack_indices(idx, bits)
    kref = dequant_reference(idx, norms, centroids)

    candidates = make_candidates(nk, dim)
    rows = []
    best = None

    ref_samples = [
        bench_ref_only(q, kref, args.warmup, args.iters) for _ in range(args.repeats)
    ]
    ref_ms_fixed = median(ref_samples)

    for i, cfg in enumerate(candidates, start=1):
        apply_env(cfg)
        native_samples = []
        for _ in range(args.repeats):
            # Use bench_case to reuse warmup path for the native call.
            _, native_ms = bench_case(
                q, kref, kp, norms, centroids, bits, args.warmup, args.iters
            )
            native_samples.append(native_ms)
        native_ms = median(native_samples)
        speedup = ref_ms_fixed / native_ms
        row = {
            "rank_hint": i,
            "config": dict(cfg),
            "ref_ms": ref_ms_fixed,
            "native_ms": native_ms,
            "speedup": speedup,
        }
        rows.append(row)
        if best is None or speedup > best["speedup"]:
            best = row

    rows_sorted = sorted(rows, key=lambda r: r["speedup"], reverse=True)
    top = rows_sorted[: max(1, args.topk)]

    print(f"Shape: nq={nq} nk={nk} dim={dim} bits={bits}")
    print(f"Candidates: {len(rows_sorted)}")
    print(f"Reference (fixed median): {ref_ms_fixed:.4f}ms")
    print("")
    print("Top configs:")
    for r in top:
        print(
            f"speedup={r['speedup']:.3f}x native={r['native_ms']:.4f}ms ref={r['ref_ms']:.4f}ms  {config_to_export(r['config'])}"
        )

    assert best is not None
    print("")
    print("Best export:")
    print(config_to_export(best["config"]))

    if args.output_json:
        payload = {
            "shape": {"nq": nq, "nk": nk, "dim": dim, "bits": bits},
            "best": best,
            "top": top,
            "all": rows_sorted,
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote: {args.output_json}")


if __name__ == "__main__":
    main()
