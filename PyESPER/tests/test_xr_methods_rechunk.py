"""Regression tests for the chunk-size policy in ``xr_methods._estimate_xr``.

Context: a caller's own dask chunking (e.g. a physics grid regridded with the entire
vertical dimension in one chunk) can produce chunks of tens of millions of points. That
chunking is chosen for regridding and IO, never for this pipeline's per-point memory
cost, so an unbounded chunk here is a real OOM risk -- this exact scenario OOM-killed a
251 GB machine twice on a 4 km / 100-level production grid.

The cap is no longer a single hard-coded point count. It is derived from a memory budget
(``PYESPER_CHUNK_MEMORY``, default 24 GiB) divided by the measured per-point cost of the
specific method and variable count requested, because that cost varies by more than 4x
across supported requests. These tests do not run the actual estimation (expensive);
they check the policy: never exceed the budget, never *grow* a caller's chunks, and
never fragment below a floor.
"""

from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from PyESPER.xr_methods import (
    _MIN_POINTS_PER_CHUNK,
    _bytes_per_point,
    _max_points_per_chunk,
    nn_xr,
)


def _fake_estimate_block(sal, temp, lon, lat, depth, dates, *, variables, path, method, equation):
    """Records the chunk's point count instead of calling any real net."""
    _fake_estimate_block.chunk_sizes.append(sal.size)
    return tuple(np.zeros(sal.shape, dtype="float64") for _ in variables)


_fake_estimate_block.chunk_sizes = []


def _grid(shape, chunks, seed=0):
    rng = np.random.default_rng(seed)
    dims = ("s", "y", "x")[: len(shape)]
    return [
        xr.DataArray(rng.uniform(lo, hi, shape), dims=dims).chunk(dict(zip(dims, chunks)))
        for lo, hi in [(32, 36), (2, 20), (-180, -110), (20, 55), (0, 500)]
    ]


def _run(arrays, **kwargs):
    _fake_estimate_block.chunk_sizes = []
    with patch("PyESPER.xr_methods._estimate_block", _fake_estimate_block):
        out = nn_xr(
            *arrays,
            variables=["nitrate", "oxygen"], path="/nonexistent", equation=8,
            est_dates=2013.0, **kwargs,
        )
        if hasattr(out["nitrate"].data, "dask"):
            out["nitrate"].compute(scheduler="synchronous")
    return out


def test_oversized_chunks_are_split_below_the_cap():
    """A caller-supplied chunk bigger than the cap must be split before the kernel.

    Driven through the explicit ``max_points_per_chunk`` override so the test stays
    cheap: the automatic cap is tens of millions of points, and allocating a grid that
    large just to watch it be divided would cost gigabytes for no extra confidence --
    the code path taken is identical either way.
    """
    # 20 x 200 x 200 in chunks of 20 x 100 x 100: 200,000 points per caller chunk.
    # Same shape of problem as the 100 x 800 x 800 / 16M-point production grid that
    # caused the original crash, scaled down to keep the test cheap.
    arrays = _grid((20, 200, 200), (20, 100, 100))
    cap = 50_000  # each caller chunk is 4x over
    _run(arrays, max_points_per_chunk=cap)

    assert _fake_estimate_block.chunk_sizes, "kernel was never invoked"
    assert max(_fake_estimate_block.chunk_sizes) <= cap, (
        f"a chunk of {max(_fake_estimate_block.chunk_sizes):,} points reached the "
        f"kernel -- over the {cap:,}-point cap"
    )
    assert len(_fake_estimate_block.chunk_sizes) > 4, "array was not actually split"


def test_caller_chunks_under_the_cap_are_not_grown():
    """Chunks smaller than the budget must be left exactly as the caller made them.

    dask's "auto" rechunking will happily *grow* chunks up to a limit. That is not this
    function's business -- the cap is a ceiling, not a target -- and rechunking anyway
    would add a graph layer for nothing.
    """
    arrays = _grid((8, 40, 40), (2, 20, 20))  # 800-point chunks, far under any budget
    before = arrays[0].chunks
    out = _run(arrays)
    assert out["nitrate"].chunks == before, "chunks were altered despite being in budget"
    assert set(_fake_estimate_block.chunk_sizes) == {800}


@pytest.mark.parametrize("method", ["nn", "lir", "mixed"])
def test_budget_shrinks_as_variable_count_grows(method):
    """More variables means more bytes per point, so fewer points per chunk."""
    one = _max_points_per_chunk(method, 1)
    six = _max_points_per_chunk(method, 6)
    assert one > six > 0
    assert _bytes_per_point(method, 6) > _bytes_per_point(method, 1)


def test_budget_respects_the_memory_env_override(monkeypatch):
    baseline = _max_points_per_chunk("nn", 6)
    monkeypatch.setenv("PYESPER_CHUNK_MEMORY", str(48 * 1024**3))
    assert _max_points_per_chunk("nn", 6) > baseline
    monkeypatch.setenv("PYESPER_CHUNK_MEMORY", "not-a-number")
    with pytest.raises(ValueError):
        _max_points_per_chunk("nn", 6)


def test_budget_never_fragments_below_the_floor(monkeypatch):
    """A tiny budget must not produce pathologically small chunks.

    Each chunk pays a fixed setup cost (defaults/iterations/polygon classification), so
    fragmenting without limit trades one problem for another.
    """
    monkeypatch.setenv("PYESPER_CHUNK_MEMORY", "1")
    assert _max_points_per_chunk("nn", 6) == _MIN_POINTS_PER_CHUNK


def test_already_small_chunks_are_left_alone_in_count():
    """Inputs already within budget should not be needlessly fragmented further."""
    _fake_estimate_block.chunk_sizes = []

    shape = (10, 50, 50)  # 25,000 points total, well under budget
    rng = np.random.default_rng(0)
    sal = xr.DataArray(rng.uniform(32, 36, shape), dims=("s", "y", "x")).chunk({"s": shape[0], "y": shape[1], "x": shape[2]})
    temp = xr.DataArray(rng.uniform(2, 20, shape), dims=("s", "y", "x")).chunk({"s": shape[0], "y": shape[1], "x": shape[2]})
    lon = xr.DataArray(rng.uniform(-180, -110, shape), dims=("s", "y", "x")).chunk({"s": shape[0], "y": shape[1], "x": shape[2]})
    lat = xr.DataArray(rng.uniform(20, 55, shape), dims=("s", "y", "x")).chunk({"s": shape[0], "y": shape[1], "x": shape[2]})
    depth = xr.DataArray(rng.uniform(0, 500, shape), dims=("s", "y", "x")).chunk({"s": shape[0], "y": shape[1], "x": shape[2]})

    with patch("PyESPER.xr_methods._estimate_block", _fake_estimate_block):
        out = nn_xr(
            sal, temp, lon, lat, depth,
            variables=["nitrate", "oxygen"], path="/nonexistent", equation=8, est_dates=2013.0,
        )
        out["nitrate"].compute(scheduler="synchronous")

    assert _fake_estimate_block.chunk_sizes == [shape[0] * shape[1] * shape[2]]


def test_numpy_backed_inputs_are_unaffected():
    """Non-dask (eager numpy) inputs must not be forced into dask by this path."""
    n = 100
    rng = np.random.default_rng(0)
    sal = xr.DataArray(rng.uniform(32, 36, n), dims=("p",))
    temp = xr.DataArray(rng.uniform(2, 20, n), dims=("p",))
    lon = xr.DataArray(rng.uniform(-180, -110, n), dims=("p",))
    lat = xr.DataArray(rng.uniform(20, 55, n), dims=("p",))
    depth = xr.DataArray(rng.uniform(0, 500, n), dims=("p",))

    with patch("PyESPER.xr_methods._estimate_block", _fake_estimate_block):
        out = nn_xr(
            sal, temp, lon, lat, depth,
            variables=["nitrate", "oxygen"], path="/nonexistent", equation=8, est_dates=2013.0,
        )

    assert not hasattr(out["nitrate"].data, "dask")


def test_estimate_block_does_not_mutate_caller_arrays():
    """``_estimate_block`` passes borrowed numpy arrays into the estimation routines.

    That is only safe because ``defaults()`` copies longitude (``np.array(...)``) before
    normalising it into [0, 360). If that ever became ``np.asarray``, this path would
    silently rewrite the caller's dask block in place -- and with a shared, cached block
    that corruption would be extremely hard to trace. Pin it.
    """
    import numpy as np

    from PyESPER.xr_methods import _estimate_block

    n = 32
    rng = np.random.default_rng(0)
    # Longitudes deliberately spanning the branches defaults() rewrites: <0 and >360.
    lon = np.linspace(-170.0, 400.0, n)
    lat = rng.uniform(-70.0, 70.0, n)
    depth = rng.uniform(0.0, 4000.0, n)
    sal = rng.uniform(32.0, 36.0, n)
    temp = rng.uniform(-1.0, 25.0, n)
    dates = np.full(n, 2002.0)
    originals = [a.copy() for a in (sal, temp, lon, lat, depth, dates)]

    _estimate_block(
        sal, temp, lon, lat, depth, dates,
        variables=["TA"], path="", method="nn", equation=8,
    )

    for name, before, after in zip(
        ("salinity", "temperature", "longitude", "latitude", "depth", "dates"),
        originals,
        (sal, temp, lon, lat, depth, dates),
    ):
        np.testing.assert_array_equal(
            after, before, err_msg=f"_estimate_block mutated the caller's {name}"
        )
