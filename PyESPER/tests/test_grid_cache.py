"""Tests for the process-wide LIR data cache (:mod:`PyESPER.kernels.grid_cache`).

The LIR path used to reload ~82 MB of ``Mat_fullgrid`` coefficients per variable and
rebuild two cKD-tree-backed interpolants on *every* call -- work that depends only on the
data files and the requested (variable, equation) combinations, never on the caller's
points. Under ``xr_methods`` that meant repeating it once per dask chunk.

These tests pin the two properties that matter: the expensive work happens exactly once,
and the answers do not change because of it.
"""

import threading

import numpy as np
import pytest

from PyESPER.kernels import grid_cache

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(
    not (REPO_ROOT / "Mat_fullgrid").is_dir(),
    reason="Mat_fullgrid/ data directory not present",
)


@pytest.fixture(autouse=True)
def _clear_cache():
    grid_cache.clear()
    yield
    grid_cache.clear()


def test_mat_files_are_read_once_per_variable(monkeypatch):
    """A repeated request must not touch ``scipy.io.loadmat`` again."""
    import scipy.io

    calls = []
    real_loadmat = scipy.io.loadmat

    def counting_loadmat(*args, **kwargs):
        calls.append(args[0])
        return real_loadmat(*args, **kwargs)

    monkeypatch.setattr(scipy.io, "loadmat", counting_loadmat)

    from PyESPER.fetch_data import fetch_data

    fetch_data(["TA"], str(REPO_ROOT))
    assert len(calls) == 4, f"expected 4 .mat reads for one variable, got {len(calls)}"

    for _ in range(5):
        fetch_data(["TA"], str(REPO_ROOT))
    assert len(calls) == 4, f"cache missed: {len(calls)} reads after 6 calls"


def test_cache_is_shared_across_variables_and_requests(monkeypatch):
    """A one-variable request reuses what a six-variable request already loaded."""
    import scipy.io

    calls = []
    real_loadmat = scipy.io.loadmat
    monkeypatch.setattr(
        scipy.io, "loadmat",
        lambda *a, **k: (calls.append(a[0]), real_loadmat(*a, **k))[1],
    )

    from PyESPER.fetch_data import fetch_data

    fetch_data(["TA", "DIC"], str(REPO_ROOT))
    assert len(calls) == 8
    fetch_data(["DIC"], str(REPO_ROOT))
    assert len(calls) == 8, "second request re-read an already-cached variable"


def test_concurrent_first_calls_build_once_and_agree():
    """Two threads racing on a cold cache must not both build, and must agree.

    This is why the cache uses an explicit lock with double-checked locking rather than
    ``functools.lru_cache``: the latter is safe but has a non-atomic miss window, so a
    race duplicates the (expensive) build.
    """
    from PyESPER.fetch_data import fetch_data

    results = {}
    barrier = threading.Barrier(8)

    def worker(i):
        barrier.wait()
        results[i] = fetch_data(["TA"], str(REPO_ROOT))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=300)
    assert not any(t.is_alive() for t in threads)
    assert len(results) == 8
    assert grid_cache.cache_info()["mat_entries"] == 1

    # Every thread must have received the *same* cached arrays, not private copies.
    first = results[0]
    for other in results.values():
        assert other[0]["TA"] is first[0]["TA"]
        assert other[1]["TA"][0] is first[1]["TA"][0]


def test_cached_arrays_are_read_only():
    """Cached data is shared; accidental mutation must raise, not corrupt later calls."""
    from PyESPER.fetch_data import fetch_data

    grid_coords, cs, aainds, _unc = fetch_data(["TA"], str(REPO_ROOT))
    with pytest.raises(ValueError):
        grid_coords["TA"][0, 0] = 1.0
    with pytest.raises(ValueError):
        aainds["TA"][0, 0] = 1.0
    with pytest.raises(ValueError):
        cs["TA"][0][0, 0] = 1.0


def test_interpolants_are_built_once_and_results_are_unchanged():
    """Repeated ``lir()`` calls reuse the interpolants and return identical numbers."""
    from PyESPER.lir import lir

    n = 500
    rng = np.random.default_rng(3)
    coords = {
        "longitude": rng.uniform(0, 360, n).tolist(),
        "latitude": rng.uniform(-78, 80, n).tolist(),
        "depth": rng.uniform(0, 5500, n).tolist(),
    }
    preds = {
        "salinity": rng.uniform(31, 37, n).tolist(),
        "temperature": rng.uniform(-2, 30, n).tolist(),
    }
    kwargs = dict(
        EstDates=[2002.0] * n, Equations=[8], verbose=False,
        compute_uncertainties=False,
    )

    first, _coef, _unc = lir(["TA"], str(REPO_ROOT), coords, preds, **kwargs)
    assert grid_cache.cache_info()["interpolant_entries"] == 1
    second, _coef, _unc = lir(["TA"], str(REPO_ROOT), coords, preds, **kwargs)
    assert grid_cache.cache_info()["interpolant_entries"] == 1, "key is not stable"

    # Bit-for-bit: the cache must not perturb the numbers at all.
    np.testing.assert_array_equal(
        np.asarray(first["TA8"], dtype=float),
        np.asarray(second["TA8"], dtype=float),
    )


def test_distinct_requests_get_distinct_interpolants():
    """The cache key must include which combinations were asked for."""
    from PyESPER.coefs_AAinds import coefs_AAinds
    from PyESPER.fetch_data import fetch_data
    from PyESPER.interpolate import interpolate

    lir_data = fetch_data(["TA", "DIC"], str(REPO_ROOT))
    gdf_one, _ = coefs_AAinds([8], _single(lir_data, "TA"))
    gdf_two, _ = coefs_AAinds([8], lir_data)
    interpolate(str(REPO_ROOT), gdf_one, {}, {}, verbose=False)
    interpolate(str(REPO_ROOT), gdf_two, {}, {}, verbose=False)
    assert grid_cache.cache_info()["interpolant_entries"] == 2


def _single(lir_data, variable):
    """Narrow a fetch_data result down to one variable, preserving its structure."""
    grid_coords, cs, aainds, unc = lir_data
    return [
        {variable: grid_coords[variable]},
        {variable: cs[variable]},
        {variable: aainds[variable]},
        unc,
    ]
