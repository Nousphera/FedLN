#!/usr/bin/env python3
"""Prepare the datasets used by FedLN."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fedln.preparation import main

if __name__ == "__main__":
    main()
