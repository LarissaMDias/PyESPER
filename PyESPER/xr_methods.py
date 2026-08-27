"""Dask-lazy, xarray-native wrappers around the PyESPER estimation routines.

The stock :func:`PyESPER.lir.lir` / :func:`PyESPER.nn.nn` / :func:`PyESPER.mixed.mixed`
entry points take dict-of-1-D-list inputs and return dict-of-1-D-array estimates for a
flat set of points. They are fully vectorised over the points and (with
``compute_uncertainties=False``) have modest, roughly fixed overhead per call.

Seawater property estimation is **point-independent**: each location's estimate depends
only on its own longitude/latitude/depth/salinity/temperature(/date). That makes it a
natural fit for :func:`xarray.apply_ufunc` with ``dask="parallelized"`` -- dask invokes
the kernel on one chunk at a time, so peak memory is bounded by the chunk size (not the
whole domain or the number of time records). **Chunks are never processed concurrently**
-- :func:`PyESPER.concurrency.kernel_lock` enforces that from inside this module, under
whatever scheduler the caller happens to use (see the neural-net thread-safety note
below). So unlike a typical dask array, "peak memory bounded by chunk size" is this
module's only real parallelism lever: make each chunk as large as memory affords (see
:func:`_max_points_per_chunk`) rather than relying on many small chunks running at once,
since they never do.

**"Bounded by the chunk size" assumes the chunks are actually small.** This used to be
acute: ``run_nets`` cost roughly 10 KB *per point*, because it carried a ``(stacked
nets, hidden width, Q points)`` activation array from layer to layer. The fused tiled
kernel that replaced it (see ``PyESPER/kernels/nn_forward.py``) holds no array sized by
both net count and point count, and measures ~450 bytes per point; the per-chunk
pipeline around it costs more than the kernel does now. The cap below is consequently
far more conservative than it needs to be -- it is left large-but-finite as a backstop
against the OOM described below, not because the old per-point cost still applies. A caller's existing dask chunking
(e.g. a physics grid regridded with the entire vertical dimension in one chunk) can
easily produce chunks of tens of millions of points -- fine for a plain regrid, but
enough to demand 100+ GB for a single ``run_nets`` call. This module defensively
caps the points per chunk before dispatching, independent of whatever chunking the
caller's arrays already carry -- this is not a hypothetical: an unchunked-enough 4 km /
100-level production grid OOM-killed a 251 GB machine twice at the default chunking
before this fix. The cap comes from :func:`_max_points_per_chunk`, which divides a memory
budget by the measured per-point cost of the specific method and variable count
requested. It only has to bound *one* chunk (see the thread-safety note below -- chunks
run strictly one at a time), not several concurrent workers' worth, so it is sized
against the whole process rather than divided by a worker count. Callers who know better
can override it with ``max_points_per_chunk``.

Public functions
----------------
``lir_xr`` / ``nn_xr`` / ``mixed_xr`` accept :class:`xarray.DataArray` inputs and return a
dict ``{variable: DataArray}`` of estimates (µmol/kg, ESPER's native units) with the same
dims/coords as the broadcast inputs. They compute **estimates only** (no uncertainties),
so downstream code can assign the results lazily to a dataset and only materialise them at
write time.

Notes
-----
* NaN inputs (e.g. land cells on a model grid) are handled: non-finite points are skipped
  and returned as NaN, so the underlying routines only ever see valid points.
* Only one chunk may be in flight at once. ``run_nets``' ``_tansig``, and the
  ``eos80_jit`` seawater routines that *both* the LIR and NN paths call, are numba
  ``@njit(parallel=True)`` kernels that each claim every core they can see
  (``numba.get_num_threads()``) *per call*. Several chunks entering them concurrently
  (dask's default threaded scheduler, several worker threads at once) was observed to
  reliably deadlock the whole process -- every thread parked in a futex wait, 0% CPU,
  stuck at a fixed completed-chunk count -- not merely slow.

  This is now enforced here rather than asked of the caller. ``_estimate_block`` runs
  inside :func:`PyESPER.concurrency.kernel_lock`, a module-level semaphore, so exactly
  one chunk is ever inside a numba parallel region no matter which scheduler the caller
  chose. The earlier advice -- "run under a synchronous scheduler" -- was not
  enforceable, because these functions return *lazy* arrays and never see the
  ``.compute()`` call; it also was not in fact being followed by
  ``roms_tools.setup.esper``, which is why production saw both the deadlock and the
  associated n_workers-times-chunk memory blowup. See ``PyESPER/concurrency.py`` for the
  policy and its escape hatches, and ``run_nets.py``'s ``batched_forward``/``_tansig``
  docstrings for the kernel-level mechanism.
"""

from __future__ import annotations

import os

import numpy as np
import xarray as xr

from PyESPER.concurrency import kernel_lock

# ESPER's estimable variables (PyESPER naming).
VALID_VARIABLES = ("TA", "DIC", "pH", "phosphate", "nitrate", "silicate", "oxygen")

# The predictor set these wrappers support: coordinates + salinity + temperature only,
# i.e. Equation 8 (S, T) or 16 (S). Extending to nutrient/oxygen predictors would mean
# threading those DataArrays through as additional PredictorMeasurements.
_METHODS = {"lir", "nn", "mixed"}

# Memory budget for one chunk, in bytes. The points-per-chunk cap is derived from this
# and the measured per-point cost of the requested method and variable count, rather
# than being a single hard-coded point count -- the per-point cost varies by more than
# 4x across the supported requests (see _bytes_per_point), so one number is either
# wasteful for small requests or unsafe for large ones.
#
# 24 GiB is deliberately conservative: chunks are processed strictly one at a time (see
# PyESPER.concurrency), so this bounds the whole process, not a per-worker share. Raise
# it via PYESPER_CHUNK_MEMORY (bytes) on a large node to cut the number of chunks and
# with it the fixed per-chunk overhead.
_DEFAULT_CHUNK_MEMORY = 24 * 1024**3

# Never fragment below this, however tight the budget: each chunk pays a fixed setup
# cost (defaults/iterations/polygon classification), so very small chunks are pure loss.
_MIN_POINTS_PER_CHUNK = 250_000

# Measured peak bytes/point for one _estimate_block call on this machine, six variables
# and one variable, equation 8, fitted as base + per_variable * n_variables:
#
#     method   1 variable   6 variables    fit
#     nn         460 B/pt     1185 B/pt    315 + 145*n
#     lir        606 B/pt     1968 B/pt    334 + 272*n
#
# The constants below round those up with roughly 1.4x headroom, because the fit is from
# one machine and one equation. `mixed` runs both paths, so it is charged for both.
_BYTES_PER_POINT = {
    "nn": (450, 200),
    "lir": (450, 380),
    "mixed": (900, 580),
}


def _chunk_memory_budget():
    """Per-chunk memory budget in bytes, overridable by ``PYESPER_CHUNK_MEMORY``."""
    raw = os.environ.get("PYESPER_CHUNK_MEMORY", "").strip()
    if not raw:
        return _DEFAULT_CHUNK_MEMORY
    try:
        value = int(float(raw))
    except ValueError:
        raise ValueError(
            f"PYESPER_CHUNK_MEMORY must be a number of bytes, got {raw!r}."
        ) from None
    if value <= 0:
        raise ValueError(f"PYESPER_CHUNK_MEMORY must be positive, got {value}.")
    return value


def _bytes_per_point(method, n_variables):
    base, per_variable = _BYTES_PER_POINT[method]
    return base + per_variable * max(int(n_variables), 1)


def _max_points_per_chunk(method, n_variables):
    """How many points one chunk may hold before it risks the memory budget."""
    budget = _chunk_memory_budget() // _bytes_per_point(method, n_variables)
    return max(int(budget), _MIN_POINTS_PER_CHUNK)


def _method_fn(method):
    method = str(method).lower()
    if method not in _METHODS:
        raise ValueError(
            f"method must be one of {sorted(_METHODS)}, got {method!r}."
        )
    if method == "lir":
        from PyESPER.lir import lir

        def _call(variables, path, coords, preds, dates, equation):
            est, _coef, _unc = lir(
                variables, path, coords, preds,
                EstDates=dates, Equations=[equation],
                verbose=False, compute_uncertainties=False,
                # Coefficients are 6 float64 per point per variable and nothing here
                # looks at them.
                want_coefficients=False,
            )
            return est

        return _call
    if method == "nn":
        from PyESPER.nn import nn

        def _call(variables, path, coords, preds, dates, equation):
            est, _unc = nn(
                variables, path, coords, preds,
                EstDates=dates, Equations=[equation],
                verbose=False, compute_uncertainties=False,
            )
            return est

        return _call

    from PyESPER.mixed import mixed

    def _call(variables, path, coords, preds, dates, equation):
        est, _unc = mixed(
            variables, path, coords, preds,
            EstDates=dates, Equations=[equation],
            verbose=False, compute_uncertainties=False,
            want_coefficients=False,
        )
        return est

    return _call


def _estimate_block(sal, temp, lon, lat, depth, dates, *, variables, path, method, equation):
    """Estimate ``variables`` for one (numpy) block of points.

    All inputs are numpy arrays of an identical arbitrary shape (the dask block). Returns
    a tuple of arrays (one per variable, in ``variables`` order), each the same shape as
    the inputs, in µmol/kg. Non-finite points are returned as NaN.
    """
    shape = sal.shape
    n_out = len(variables)
    outs = [np.full(shape, np.nan, dtype="float64") for _ in range(n_out)]

    sal_f = np.asarray(sal, dtype="float64").ravel()
    temp_f = np.asarray(temp, dtype="float64").ravel()
    lon_f = np.asarray(lon, dtype="float64").ravel()
    lat_f = np.asarray(lat, dtype="float64").ravel()
    depth_f = np.asarray(depth, dtype="float64").ravel()
    dates_f = np.asarray(dates, dtype="float64").ravel()

    valid = (
        np.isfinite(sal_f) & np.isfinite(temp_f) & np.isfinite(lon_f)
        & np.isfinite(lat_f) & np.isfinite(depth_f) & np.isfinite(dates_f)
    )
    if not valid.any():
        return tuple(outs)

    idx = np.flatnonzero(valid)

    # Everything from here on is serialised, not just the kernel call itself.
    #
    # Two reasons. (1) Safety: PyESPER's numba kernels are ``@njit(parallel=True)`` and
    # each claims the whole thread pool per call, so several dask worker threads
    # entering them at once deadlocks the process -- see ``PyESPER.concurrency`` for the
    # mechanism and the escape hatches. (2) Memory: the bulk of a block's footprint is
    # the ``.tolist()`` conversions below and the estimation call's own temporaries, so
    # the lock has to be held across those too for peak memory to be a deterministic
    # one-chunk bound rather than one-chunk-times-workers. Acquiring it only around the
    # estimation call would leave every waiting worker holding a full set of
    # point-length Python lists.
    with kernel_lock():
        # numpy arrays, not Python lists. The estimation routines accept either --
        # everything downstream immediately calls np.array/np.asarray on these -- but a
        # list of n Python floats costs ~40 bytes/point against 8, and building six of
        # them (then converting them straight back) was a measurable share of each
        # chunk. `defaults()` copies longitude before modifying it, so passing borrowed
        # arrays here does not mutate the caller's data.
        coords = {
            "longitude": lon_f[idx],
            "latitude": lat_f[idx],
            "depth": depth_f[idx],
        }
        preds = {
            "salinity": sal_f[idx],
            "temperature": temp_f[idx],
        }
        est = _method_fn(method)(
            list(variables), path, coords, preds, dates_f[idx], equation
        )
        del coords, preds

        flat_outs = [
            np.full(sal_f.shape, np.nan, dtype="float64") for _ in range(n_out)
        ]
        for i, var in enumerate(variables):
            key = f"{var}{equation}"
            flat_outs[i][idx] = np.asarray(est[key], dtype="float64").ravel()
            outs[i] = flat_outs[i].reshape(shape)
    return tuple(outs)


def _estimate_xr(salinity, temperature, longitude, latitude, depth, *,
                 variables, path, method, equation, est_dates,
                 max_points_per_chunk=None):
    """Shared implementation for ``lir_xr``/``nn_xr``/``mixed_xr``."""
    if isinstance(variables, str):
        variables = [variables]
    variables = list(variables)
    unknown = [v for v in variables if v not in VALID_VARIABLES]
    if unknown:
        raise ValueError(
            f"Unknown variable(s) {unknown}; valid options: {list(VALID_VARIABLES)}."
        )
    if equation not in (8, 16):
        raise ValueError(
            "equation must be 8 (salinity+temperature) or 16 (salinity only); "
            f"got {equation!r}. These are the S/T-only ESPER equations."
        )

    # Dates as a DataArray so it broadcasts alongside the fields (decimal year).
    if est_dates is None:
        est_dates = 2002.0
    if not isinstance(est_dates, xr.DataArray):
        est_dates = xr.DataArray(est_dates)

    # Align/broadcast every input to a common set of dims (lazy for dask inputs).
    sal, temp, lon, lat, dep, dates = xr.broadcast(
        salinity, temperature, longitude, latitude, depth, est_dates
    )

    # Defensively cap the points per chunk before dispatching, regardless of whatever
    # chunking the broadcast inputs already carry: a caller's chunking is chosen for
    # regridding and IO, never for this pipeline's per-point cost, and an unbounded
    # chunk here is a real OOM (it killed a 251 GB machine twice on a 4 km / 100-level
    # grid). The cap is derived from a memory budget and the measured per-point cost of
    # this particular request -- see _max_points_per_chunk.
    if sal.chunks is not None:
        budget_points = (
            _max_points_per_chunk(method, len(variables))
            if max_points_per_chunk is None
            else max(int(max_points_per_chunk), 1)
        )
        # Largest block the caller's own chunking would produce. Only shrink: if the
        # caller already chose something smaller, respect it. Rechunking anyway would
        # add a graph layer for nothing, and "auto" would happily *grow* their chunks
        # up to the budget, which is not this function's business.
        largest_block = 1
        for per_dim in sal.chunks:
            largest_block *= max(per_dim)

        if largest_block > budget_points:
            from dask.array.core import normalize_chunks

            target_chunks = normalize_chunks(
                "auto",
                shape=sal.shape,
                limit=budget_points * 8,  # bytes-equivalent at 8 B/element
                dtype=np.dtype("float64"),
                previous_chunks=sal.chunks,
            )
            rechunk_kwargs = dict(zip(sal.dims, target_chunks, strict=True))
            sal, temp, lon, lat, dep, dates = (
                arr.chunk(rechunk_kwargs) for arr in (sal, temp, lon, lat, dep, dates)
            )

    results = xr.apply_ufunc(
        _estimate_block,
        sal, temp, lon, lat, dep, dates,
        kwargs=dict(
            variables=variables, path=str(path), method=method, equation=equation
        ),
        output_core_dims=[[]] * len(variables),
        dask="parallelized",
        output_dtypes=[np.float64] * len(variables),
    )
    if len(variables) == 1:
        results = (results,)
    return {var: results[i] for i, var in enumerate(variables)}


def lir_xr(salinity, temperature, longitude, latitude, depth, *,
           variables, path="", equation=8, est_dates=None,
           max_points_per_chunk=None):
    """Dask-lazy LIR estimates as xarray DataArrays. See module docstring.

    Parameters
    ----------
    salinity, temperature, longitude, latitude, depth : xarray.DataArray
        Predictors/coordinates on the target grid (any shared/broadcastable dims). Salinity
        is PSS-78, temperature in-situ °C, depth metres (positive down), longitude °E,
        latitude °N.
    variables : str or list of str
        ESPER variable name(s) to estimate: subset of ``TA``, ``DIC``, ``phosphate``,
        ``nitrate``, ``silicate``, ``oxygen``, ``pH``.
    path : str, optional
        Directory containing the ESPER data (``Mat_fullgrid/`` etc.). Optional:
        when omitted (or empty), it is resolved via :func:`PyESPER.paths.data_root`
        -- the ``PYESPER_DATA_DIR`` environment variable, else auto-detected next
        to the installed package (works for a repository checkout on ``sys.path``
        and for an editable ``pip install -e``).
    equation : int
        8 (salinity + temperature) or 16 (salinity only).
    est_dates : float or xarray.DataArray, optional
        Decimal year(s); only affects DIC/pH. Defaults to 2002.0.
    max_points_per_chunk : int, optional
        Override the automatic points-per-chunk cap. By default it is derived from a
        24 GiB budget (``PYESPER_CHUNK_MEMORY``) and this request's measured per-point
        cost. Chunks smaller than the cap are never grown.

    Returns
    -------
    dict[str, xarray.DataArray]
        ``{variable: estimate}`` in µmol/kg, dims/coords matching the broadcast inputs.
    """
    return _estimate_xr(
        salinity, temperature, longitude, latitude, depth,
        variables=variables, path=path, method="lir", equation=equation,
        est_dates=est_dates, max_points_per_chunk=max_points_per_chunk,
    )


def nn_xr(salinity, temperature, longitude, latitude, depth, *,
          variables, path="", equation=8, est_dates=None,
          max_points_per_chunk=None):
    """Dask-lazy neural-network estimates as xarray DataArrays. See :func:`lir_xr`."""
    return _estimate_xr(
        salinity, temperature, longitude, latitude, depth,
        variables=variables, path=path, method="nn", equation=equation,
        est_dates=est_dates, max_points_per_chunk=max_points_per_chunk,
    )


def mixed_xr(salinity, temperature, longitude, latitude, depth, *,
             variables, path="", equation=8, est_dates=None,
             max_points_per_chunk=None):
    """Dask-lazy LIR+NN ensemble-mean estimates as xarray DataArrays. See :func:`lir_xr`."""
    return _estimate_xr(
        salinity, temperature, longitude, latitude, depth,
        variables=variables, path=path, method="mixed", equation=equation,
        est_dates=est_dates, max_points_per_chunk=max_points_per_chunk,
    )
