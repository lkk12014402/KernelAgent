"""Benchmark verifier."""

import importlib.util
import sys
import time
from pathlib import Path

import torch

from .base import Verifier, VerifyResult


class BenchmarkVerifier(Verifier):
    """Verify kernel performance."""

    def verify(self, kernel_path: Path) -> VerifyResult:
        """Run benchmarks."""
        try:
            # Import benchmark module
            spec = importlib.util.spec_from_file_location(
                "bench", kernel_path.parent / "bench.py"
            )
            bench = importlib.util.module_from_spec(spec)
            sys.modules["bench"] = bench
            spec.loader.exec_module(bench)

            # Run benchmark
            bench_fn = getattr(bench, f"bench_{kernel_path.parent.parent.name}")

            # Warmup
            for _ in range(self.cfg["warmup"]):
                bench_fn()

            # Benchmark
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(self.cfg["repetitions"]):
                bench_fn()
            torch.cuda.synchronize()
            end = time.time()

            # Calculate metrics
            avg_time = (end - start) / self.cfg["repetitions"] * 1000  # ms

            return VerifyResult(
                ok=True,
                message=f"Benchmark verification passed: {avg_time:.2f} ms",
                profile_data={"avg_time_ms": avg_time},
            )

        except Exception as e:
            return VerifyResult(
                ok=False, message=f"Benchmark verification failed: {str(e)}"
            )
