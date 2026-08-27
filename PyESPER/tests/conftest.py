"""Shared pytest configuration for the PyESPER test suite.

Ensures the repository root is importable (so ``PyESPER`` and the ``NeuralNetworks``
weight modules both resolve) regardless of where pytest is invoked from.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
