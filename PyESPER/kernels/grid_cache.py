"""Process-wide, thread-safe caches for the point-independent LIR data.

The LIR path recomputes, on **every call**, a large amount of work that does not depend
on the caller's points at all:

* :func:`PyESPER.fetch_data.fetch_data` reads four ``Mat_fullgrid/*.mat`` files per
  requested variable through ``scipy.io.loadmat`` -- about 82 MB of coefficients per
  variable -- with no memoisation anywhere.
* :func:`PyESPER.interpolate.interpolate` builds two
  ``scipy.interpolate.NearestNDInterpolator`` cKD-trees over the ~106,400-node ESPER
  grid, evaluates each at ~106,400 query points to fill its region's coefficient cube,
  and constructs two ``RegularGridInterpolator``s over the result.

None of that is a function of longitude/latitude/depth; it is a function of the data
files and which (variable, equation) combinations were asked for. Under
``xr_methods`` the whole lot was therefore being rebuilt once per dask chunk -- roughly
2.8 s of pure repetition per chunk, paid a dozen or more times for a production domain.

Both caches are keyed on the absolute path of the data directory (so two different ESPER
archives never collide) plus the request, and guarded by one module-level lock with
double-checked locking: the fast path is a plain dict lookup with no lock at all, and the
lock is only taken on a miss, to stop two threads from redundantly doing the same
expensive build. One lock rather than one per key is deliberate -- builds are seconds
apart at worst, and contention is irrelevant next to the work being avoided.

Memory: caching every variable's coefficients costs ~82 MB each (~490 MB for the six
production variables) and the two interpolants a further ~70 MB for a single-equation
request. That is bounded, paid once, and far cheaper than the repeated rebuild it
replaces.
"""

from __future__ import annotations

import os
import threading

_LOCK = threading.Lock()
_MAT_CACHE: dict[tuple, dict] = {}
_INTERP_CACHE: dict[tuple, tuple] = {}
_TABLE_CACHE: dict[tuple, tuple] = {}


def _normalise(path) -> str:
    """Canonical form of the data directory, for use as a cache key and for loads.

    Resolves an empty/None ``path`` through :func:`PyESPER.paths.data_root` (env var,
    then auto-detection next to the installed package), then makes it absolute and
    symlink-free so two spellings of the same directory share one cache entry.
    """
    from PyESPER.paths import data_root

    return os.path.realpath(data_root(path))


def clear() -> None:
    """Drop everything. Intended for tests and for reclaiming the memory."""
    with _LOCK:
        _MAT_CACHE.clear()
        _INTERP_CACHE.clear()
        _TABLE_CACHE.clear()


def cache_info() -> dict:
    """Cache occupancy, for tests and diagnostics."""
    return {
        "mat_entries": len(_MAT_CACHE),
        "interpolant_entries": len(_INTERP_CACHE),
        "table_entries": len(_TABLE_CACHE),
    }


def variable_grids(variable: str, path) -> dict:
    """Load (and memoise) one variable's ``Mat_fullgrid`` coefficients.

    Returns ``{"grid_coords", "cs", "aainds", "uncgrid"}`` where ``cs`` is the list of 16
    per-equation coefficient arrays. Cached per *single* variable rather than per
    request, so a six-variable call and a later one-variable call share the loads.
    """
    path = _normalise(path)
    key = (path, variable)
    entry = _MAT_CACHE.get(key)
    if entry is not None:
        return entry
    with _LOCK:
        entry = _MAT_CACHE.get(key)
        if entry is None:
            entry = _load_variable(variable, path)
            _MAT_CACHE[key] = entry
    return entry


def _load_variable(variable: str, path) -> dict:
    import numpy as np
    from scipy.io import loadmat

    def _mat(suffix):
        return os.path.join(path, f"Mat_fullgrid/LIR_files_{variable}_{suffix}.mat")

    cs1 = loadmat(_mat("fullCs1"), squeeze_me=True)
    cs2 = loadmat(_mat("fullCs2"), squeeze_me=True)
    cs3 = loadmat(_mat("fullCs3"), squeeze_me=True)
    grid = loadmat(_mat("fullGrids"))

    combined = np.concatenate((cs1["Cs1"], cs2["Cs2"], cs3["Cs3"]), axis=1)
    # Freeze the base *before* slicing: numpy does not retroactively propagate a
    # writeable=False flag to views that already exist, so views taken first would stay
    # mutable and the read-only guarantee below would be a lie.
    combined.flags.writeable = False
    entry = {
        "grid_coords": grid["GridCoords"],
        "cs": [combined[:, :, i] for i in range(combined.shape[2])],
        "aainds": grid["AAIndsM"],
        "uncgrid": grid["UncGrid"][0][0],
        # Keep the concatenated block alive explicitly: `cs` holds views into it, so it
        # can never be freed anyway, and naming it makes that obvious rather than a
        # surprise to whoever next reads the memory profile.
        "_combined": combined,
    }
    # Shared across threads and calls -- make accidental mutation raise rather than
    # silently poison every later call.
    for value in (entry["grid_coords"], entry["aainds"]):
        value.flags.writeable = False
    return entry


def interpolants(path, gdf, build):
    """Memoise the two region interpolants built from ``gdf``.

    The key is the data directory plus ``tuple(gdf)`` -- the combination *names* in
    order. That is sufficient and exact: given a fixed data directory, the ordered set of
    ``f"{variable}{equation}"`` keys determines every coefficient that goes into the
    interpolants, and the order determines which axis of the result belongs to which
    combination (``organize_data`` indexes the output positionally).

    ``build`` is a zero-argument callable invoked only on a miss.
    """
    key = (_normalise(path), tuple(gdf))
    entry = _INTERP_CACHE.get(key)
    if entry is not None:
        return entry
    with _LOCK:
        entry = _INTERP_CACHE.get(key)
        if entry is None:
            entry = build()
            _INTERP_CACHE[key] = entry
    return entry


def stacked_table(path, gdf, build):
    """Memoise the kernel-ready coefficient table derived from the interpolants.

    Same key and lock discipline as :func:`interpolants`. Kept as a separate entry
    because it is a different representation of the same data -- one contiguous
    ``(2, nx, ny, nz, n_combo, n_coef)`` array, region-major, so the kernel can select a
    point's region by indexing rather than by branching between two objects.
    """
    key = (_normalise(path), tuple(gdf))
    entry = _TABLE_CACHE.get(key)
    if entry is not None:
        return entry
    with _LOCK:
        entry = _TABLE_CACHE.get(key)
        if entry is None:
            entry = build()
            _TABLE_CACHE[key] = entry
    return entry
