"""Locate PyESPER's data directories without the caller having to know them.

PyESPER needs three things at runtime that live *next to* the ``PyESPER`` package
rather than inside it: ``NeuralNetworks/`` (the hard-coded net weights, needed by the
``nn``/``mixed`` methods), ``Mat_fullgrid/`` (288 MB of pre-trained LIR coefficient
grids, needed by ``lir``/``mixed``), and ``Uncertainty_Polys/`` (uncertainty paths
only). Historically every entry point took a ``Path`` argument pointing at the
repository checkout that holds them, and callers had to thread that path everywhere.

:func:`data_root` removes that requirement for the common cases. Resolution order:

1. An explicit, non-empty ``path`` argument -- always wins, unchanged behaviour.
2. The ``PYESPER_DATA_DIR`` environment variable.
3. Auto-detection: the parent directory of the installed ``PyESPER`` package, if it
   contains ``Mat_fullgrid/``. This covers both a repository checkout on ``sys.path``
   and a ``pip install -e`` (editable) install -- an editable install maps the package
   straight back to its source tree, so the data directories sit exactly there.

A plain (non-editable) ``pip install`` copies only the Python packages into
site-packages, not the 288 MB of ``.mat`` grids, so auto-detection deliberately
*verifies* that ``Mat_fullgrid`` is actually present rather than guessing -- with a
wheel-style install, set ``PYESPER_DATA_DIR`` to a checkout of the data directories
(or install editable, which is the recommended way to use this fork).
"""

from __future__ import annotations

import os
from pathlib import Path

# The directory that marks a valid data root. NeuralNetworks/ is found via a normal
# import (it is a package), so the .mat grids are the only thing to locate on disk.
_MARKER = "Mat_fullgrid"


def data_root(path=None) -> str:
    """Resolve the directory holding ``Mat_fullgrid/`` etc. See module docstring."""
    if path is not None and str(path).strip():
        return str(path)

    env = os.environ.get("PYESPER_DATA_DIR", "").strip()
    if env:
        if not (Path(env) / _MARKER).is_dir():
            raise FileNotFoundError(
                f"PYESPER_DATA_DIR={env!r} does not contain a {_MARKER}/ directory."
            )
        return env

    package_parent = Path(__file__).resolve().parent.parent
    if (package_parent / _MARKER).is_dir():
        return str(package_parent)

    raise FileNotFoundError(
        "Cannot locate PyESPER's data directories (Mat_fullgrid/ etc.). They live "
        "next to the PyESPER package in its repository, and are found automatically "
        "for a repository checkout or an editable install (pip install -e). For any "
        "other install, either pass an explicit path to the estimation call or set "
        f"PYESPER_DATA_DIR to a directory containing {_MARKER}/. "
        f"(Looked in: {package_parent})"
    )
