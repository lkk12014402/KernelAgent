"""Verification loop."""

from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass
import py_compile

from .base import Verifier, VerifyResult
from .numeric import NumericVerifier
from kernel_perf_agent.kernel_opt.utils.parser_util import get_unwrapped_source

@dataclass
class VerifyResult:
    """Verification result."""

    ok: bool
    message: str

def KernelFileVerifier(
    module_path: Path,
) -> VerifyResult:
    """Run verification loop.

    Args:
        module_path: Path to the kernel file
    """

    try:
        py_compile.compile(module_path)
    except py_compile.PyCompileError as e:
        return VerifyResult(ok=False, message=f"Input compilation failed with syntax error: {e}")

    return VerifyResult(ok=True, message="Syntax verification passed.")

def KernelCodeVerifier(
    code: str,
) -> VerifyResult:
    """Run verification loop.

    Args:
        code: Code to be verified
    """

    try:
        compiled_code = compile(code, '<string>', 'exec')
    except SyntaxError as e:
        return VerifyResult(ok=False, message=f"Input compilation failed with syntax error: {e}")
    except Exception as e:
        return VerifyResult(ok=False, message=f"Input compilation failed with other errors: {e}")

    return VerifyResult(ok=True, message="Syntax verification passed.")
