from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from mosaic.serving.engine import MosaicSearchEngine


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("reports/serving_benchmark_v1.json"))
    args = parser.parse_args()
    engine = MosaicSearchEngine(args.root, device="cpu")
    if not engine.ready:
        raise RuntimeError(engine.health())
    rng = np.random.default_rng(20260726)
    vectors = rng.normal(size=(args.requests, engine._vectors.shape[1])).astype(np.float32)
    latencies: list[float] = []
    # Warm up cache and FAISS dynamic library.
    engine.search_vector(vectors[0], args.top_k)
    for vector in vectors:
        started = time.perf_counter()
        engine.search_vector(vector, args.top_k)
        latencies.append((time.perf_counter() - started) * 1000.0)
    report = {
        "schema_version": "mosaic.serving_benchmark.v1",
        "scope": "in_process_vector_search_not_http_text_encoding",
        "backend": engine.health()["index_backend"],
        "items": int(engine._vectors.shape[0]),
        "dimension": int(engine._vectors.shape[1]),
        "requests": int(args.requests),
        "top_k": int(args.top_k),
        "latency_ms": {
            "mean": float(np.mean(latencies)),
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "p99": percentile(latencies, 99),
            "max": max(latencies),
        },
        "throughput_qps_sequential": 1000.0 / float(np.mean(latencies)),
        "claim_boundary": "single-process local exact-index microbenchmark; no network/concurrency/SLA claim",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

