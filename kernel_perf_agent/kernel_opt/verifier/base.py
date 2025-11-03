"""Base verification classes."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass
class VerifyResult:
    """Verification result."""

    ok: bool
    message: str
    profile_data: Optional[Dict] = None


class Verifier(ABC):
    """Abstract base class for verifiers."""

    @abstractmethod
    def verify(self, kernel_path: Path, kernel_code: str, test_code: bool) -> VerifyResult:
        """Verify kernel implementation."""
        pass
