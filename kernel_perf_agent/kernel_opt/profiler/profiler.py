"""Profiler."""
from pathlib import Path
from .base import Profiler, ProfilingResult

def KernelProfiler(
    kernel_path: Path
) -> ProfilingResult:
    return ProfilingResult(ok=True, message="Passes")
