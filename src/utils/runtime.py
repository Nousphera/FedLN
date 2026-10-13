"""Nousphera maintenance: isolate disposable simulation files from user data."""

from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory


@contextmanager
def temporary_cache_directory(parent):
    """Remove only the new directory owned by this simulation, even on failure."""
    parent = Path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="fedln-", dir=parent) as directory:
        yield directory
