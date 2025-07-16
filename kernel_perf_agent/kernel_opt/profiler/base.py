"""Base profiler classes."""
from pathlib import Path
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class ProfilingResult:
    """Profiling result."""
    ok: bool
    message: str


class Profiler(ABC):
    """Abstract base class for profilers."""

    @abstractmethod
    def profile(self, kernel_path: Path) -> ProfilingResult:
        """Profile kernel execution."""
        pass
