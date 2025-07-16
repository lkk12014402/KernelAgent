"""Utility functions for file I/O operations."""

from pathlib import Path
from typing import Union


def write_file(path: Union[str, Path], content: str) -> None:
    """Write content to a file.

    Args:
        path: Path to the file
        content: Content to write
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
