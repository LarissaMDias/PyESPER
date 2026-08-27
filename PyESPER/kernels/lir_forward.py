"""Fused trilinear-interpolation + linear-model kernel for the LIR path.

What this replaces
------------------
An LIR estimate is, underneath all the bookkeeping, one line::

    estimate = C0 + C_S*S + C_T*T + C_A*A + C_B*B + C_C*C

where the six coefficients come from a trilinear lookup in a pre-trained
(longitude, latitude, depth) grid, using one of two coefficient sets depending on
whether the point is in the Atlantic/Arctic or not. Getting to that line used to cost
three separate O(n_points) passes, and they dominated the path:

* ``scipy``'s :class:`~scipy.interpolate.RegularGridInterpolator` took its slow Python
  ``_evaluate_linear`` branch, because the coefficient array is 5-D. That path allocates
  a full fancy-indexed copy of the values, a multiply and a fresh add for each of the 8
  hypercube corners, with no in-place accumulation -- about 200 bytes/point *per
  equation* in temporaries. Measured 55% of an LIR call.
* ``input_AAinds`` split the inputs into two dicts of 14 full-length arrays each, per
  variable-equation combination, in order to evaluate the two regions separately -- 25%.
* ``organize_data`` then concatenated the halves back together, argsorted an index
  column, and fancy-indexed seven arrays to undo the split -- 11%.

This kernel does all of it in one pass with no heap temporaries: per point it brackets
the three axes, accumulates the eight corners into a small thread-local buffer, picks
its region by a boolean mask (so the split/merge/argsort simply never happen, and points
never leave input order), and applies the linear model immediately.

Numerical fidelity
------------------
Bit-for-bit identical to the scipy path, deliberately. The table handed to the kernel is
the very array scipy was interpolating -- lifted straight off the cached interpolant
objects rather than rebuilt -- and the bracket search, the weight products and the
accumulation order all reproduce ``RegularGridInterpolator``'s exactly, including its
``searchsorted``-then-clip index convention and its corner ordering. Clamp-to-edge would
have been within tolerance, but reproducing scipy costs nothing here and makes the
regression test an equality check rather than a tolerance check.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

# Points per tile. The per-tile state is one small (n_combo, n_coef) accumulator, so this
# only needs to be large enough to amortise the loop setup.
_TILE = 4096

# fastmath is off, deliberately, and it costs nothing measurable: this kernel is bound by
# the eight scattered corner reads, not by arithmetic throughput, so FMA contraction and
# reassociation buy no speed here (measured 0.057 s against 0.058 s at 20k points x 6
# combinations). What they do buy is a few ulps of drift away from the scipy path -- 2.6e-15
# relative, harmless in itself, but it turns the regression test from an equality check
# into a tolerance check. Exactness is worth more than a 2% arithmetic margin that does
# not exist. It also keeps NaN propagation well-defined: coefficient grids legitimately
# contain NaN for equations that do not use a predictor.
_FASTMATH = False


@njit(inline="always")
def _bracket(grid, x):
    """scipy's ``RegularGridInterpolator._find_indices``, for one scalar.

    ``np.searchsorted(grid, x) - 1`` clipped to ``[0, len(grid) - 2]``, then the
    normalised distance within that cell. Reproduced exactly rather than reinvented:
    the ``side='left'`` convention decides which cell a point lying *on* a knot falls
    into, and differing there would change results at every grid node.
    """
    i = np.searchsorted(grid, x) - 1
    if i < 0:
        i = 0
    elif i > grid.shape[0] - 2:
        i = grid.shape[0] - 2
    return i, (x - grid[i]) / (grid[i + 1] - grid[i])


@njit(parallel=True, fastmath=_FASTMATH, nogil=True, cache=True, boundscheck=False)
def _lir_kernel(
    lon, lat, d2d, in_atlantic, table, gx, gy, gz,
    predictors, sel, estimates, coefficients, want_coefficients, tile,
):
    """Interpolate coefficients and apply the linear model, for every point.

    ``table`` is ``(2, nx, ny, nz, n_combo, n_coef)`` with region 0 = Atlantic/Arctic.
    ``sel[c, q]`` is the column of ``predictors`` multiplying coefficient ``q`` of
    combination ``c``, or -1 if that term is not used by the combination's equation.
    Coefficient 0 is the intercept and has no predictor.
    """
    n_points = lon.shape[0]
    n_combo = table.shape[4]
    n_coef = table.shape[5]
    n_tiles = (n_points + tile - 1) // tile

    for t in prange(n_tiles):
        p0 = t * tile
        p1 = min(p0 + tile, n_points)
        acc = np.empty((n_combo, n_coef), dtype=np.float64)

        for p in range(p0, p1):
            ix, tx = _bracket(gx, lon[p])
            iy, ty = _bracket(gy, lat[p])
            iz, tz = _bracket(gz, d2d[p])
            region = 0 if in_atlantic[p] else 1

            for c in range(n_combo):
                for q in range(n_coef):
                    acc[c, q] = 0.0

            # The eight hypercube corners, in scipy's own order (last axis varies
            # fastest) and with its own weight association: the lower edge carries
            # (1 - t), the upper carries t. Weight is formed as (wx*wy)*wz to match its
            # left-to-right accumulation.
            for cx in range(2):
                wx = tx if cx == 1 else 1.0 - tx
                for cy in range(2):
                    wy = ty if cy == 1 else 1.0 - ty
                    for cz in range(2):
                        wz = tz if cz == 1 else 1.0 - tz
                        w = (wx * wy) * wz
                        block = table[region, ix + cx, iy + cy, iz + cz]
                        for c in range(n_combo):
                            for q in range(n_coef):
                                acc[c, q] += block[c, q] * w

            for c in range(n_combo):
                value = acc[c, 0]  # intercept
                for q in range(1, n_coef):
                    column = sel[c, q]
                    if column >= 0:
                        value += acc[c, q] * predictors[column, p]
                estimates[p, c] = value
                if want_coefficients:
                    for q in range(n_coef):
                        column = sel[c, q]
                        # Unused terms are reported as exactly 0.0, matching the
                        # zero-filled columns organize_data used to emit.
                        coefficients[p, c, q] = (
                            acc[c, q] if (q == 0 or column >= 0) else 0.0
                        )


# Which coefficient slots each equation actually uses. organize_data derived this from
# ``mask = 16 - equation`` and four bit tests; this is the same thing expressed directly,
# in coefficient order (intercept, S, T, A, B, C). Verified equal for all 16 equations.
def _active_coefficients(equation):
    mask = 16 - int(equation)
    return (
        True,               # intercept
        True,               # S: always used
        bool(mask & 8),     # T
        bool(mask & 2),     # A
        bool(mask & 1),     # B
        bool(mask & 4),     # C
    )


_COEF_PREDICTOR = (None, "S", "T", "A", "B", "C")


def _stack_regions(interp_aa, interp_else):
    """One contiguous ``(2, nx, ny, nz, n_combo, n_coef)`` table plus the shared axes.

    The values are taken straight off the ``RegularGridInterpolator`` objects the
    existing build path produced, so the kernel is interpolating byte-identical data to
    what scipy was given. If a region has no grid points at all its half is filled with
    NaN, which is what scipy's ``fill_value`` would have produced for it.
    """
    reference = interp_aa if interp_aa is not None else interp_else
    if reference is None:
        raise ValueError("neither region produced an interpolant")

    grid = tuple(np.ascontiguousarray(axis, dtype=np.float64) for axis in reference.grid)
    shape = (2,) + tuple(len(axis) for axis in grid) + reference.values.shape[3:]
    table = np.empty(shape, dtype=np.float64)
    for index, interp in enumerate((interp_aa, interp_else)):
        if interp is None:
            table[index] = np.nan
        else:
            table[index] = interp.values
    return table, grid


def lir_estimates(path, gdf, code, longitude, latitude, depth, in_atlantic,
                  want_coefficients=True, tile=_TILE):
    """Estimates (and optionally coefficients) for every ``gdf`` combination.

    Parameters
    ----------
    path, gdf
        As passed to :func:`PyESPER.interpolate.interpolate`; used to look up the cached
        interpolants, which are built at most once per process.
    code
        ``iterations()``-shaped dict, for the S/T/A/B/C predictor columns.
    longitude, latitude, depth
        Point coordinates. ``depth`` is in metres; the grid's third axis is depth/25.
    in_atlantic
        Boolean array selecting the Atlantic/Arctic coefficient set per point.

    Returns
    -------
    (Estimate, CoefficientsUsed)
        Dicts keyed by combination name, in ``gdf`` order. ``CoefficientsUsed`` is empty
        when ``want_coefficients`` is False.
    """
    from PyESPER.interpolate import build_interpolants
    from PyESPER.kernels import grid_cache

    names = list(gdf)
    interp_aa, interp_else = grid_cache.interpolants(
        path, gdf, lambda: build_interpolants(gdf)
    )
    table, (gx, gy, gz) = grid_cache.stacked_table(
        path, gdf, lambda: _stack_regions(interp_aa, interp_else)
    )

    n_points = len(longitude)
    n_combo = table.shape[4]
    n_coef = table.shape[5]
    if n_combo != len(names):
        raise ValueError(
            f"coefficient table holds {n_combo} combinations but {len(names)} were "
            "requested; the interpolant cache key is out of step with `gdf`"
        )

    # De-duplicated predictor columns, plus a (combination, coefficient) -> column map.
    columns: list[np.ndarray] = []
    by_id: dict[int, int] = {}

    def column_index(array):
        array = np.ascontiguousarray(array, dtype=np.float64)
        index = by_id.get(id(array))
        if index is None:
            index = len(columns)
            by_id[id(array)] = index
            columns.append(array)
        return index

    sel = np.full((len(names), n_coef), -1, dtype=np.int64)
    for c, name in enumerate(names):
        equation = int("".join(ch for ch in name if ch.isdigit()))
        active = _active_coefficients(equation)
        entry = code[name]
        for q in range(1, n_coef):
            if active[q]:
                sel[c, q] = column_index(entry[_COEF_PREDICTOR[q]])

    predictors = (
        np.ascontiguousarray(np.stack(columns))
        if columns
        else np.zeros((1, n_points), dtype=np.float64)
    )

    estimates = np.empty((n_points, n_combo), dtype=np.float64)
    coefficients = (
        np.empty((n_points, n_combo, n_coef), dtype=np.float64)
        if want_coefficients
        else np.empty((1, 1, 1), dtype=np.float64)
    )

    if n_points:
        _lir_kernel(
            np.ascontiguousarray(longitude, dtype=np.float64),
            np.ascontiguousarray(latitude, dtype=np.float64),
            np.ascontiguousarray(np.asarray(depth, dtype=np.float64) / 25.0),
            np.ascontiguousarray(in_atlantic, dtype=np.bool_),
            table, gx, gy, gz, predictors, sel,
            estimates, coefficients, bool(want_coefficients), int(tile),
        )

    Estimate = {name: estimates[:, c] for c, name in enumerate(names)}
    CoefficientsUsed = {}
    if want_coefficients:
        labels = ("Intercept", "Coef S", "Coef T", "Coef A", "Coef B", "Coef C")
        for c, name in enumerate(names):
            CoefficientsUsed[name] = {
                label: coefficients[:, c, q] for q, label in enumerate(labels)
            }
    return Estimate, CoefficientsUsed
