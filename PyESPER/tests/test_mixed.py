"""Regression tests for ``mixed``/``mixed_xr``.

``mixed()`` averages the LIR and NN estimates. It used to index its two uncertainty
dicts unconditionally, but the gridded entry points always call it with
``compute_uncertainties=False``, which makes both dicts ``None`` -- so ``mixed_xr``
raised ``TypeError`` on every call and the method was effectively unusable from
``xr_methods``.
"""

import numpy as np
import pytest

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(
    not (REPO_ROOT / "Mat_fullgrid").is_dir(),
    reason="Mat_fullgrid/ data directory not present",
)


def _inputs(n=150, seed=2):
    xr = pytest.importorskip("xarray")
    rng = np.random.default_rng(seed)
    bounds = [(32, 36), (-1, 25), (0, 360), (-70, 70), (0, 4000)]
    return [
        xr.DataArray(rng.uniform(lo, hi, n), dims=("p",)).chunk({"p": n // 2})
        for lo, hi in bounds
    ]


def test_mixed_xr_is_the_mean_of_lir_and_nn():
    dask = pytest.importorskip("dask")
    from PyESPER.xr_methods import lir_xr, mixed_xr, nn_xr

    args = _inputs()
    kwargs = dict(
        variables=["TA", "DIC"], path=str(REPO_ROOT), equation=8, est_dates=2002.0
    )
    with dask.config.set(scheduler="synchronous"):
        mixed = {k: v.compute().values for k, v in mixed_xr(*args, **kwargs).items()}
        lir = {k: v.compute().values for k, v in lir_xr(*args, **kwargs).items()}
        nn = {k: v.compute().values for k, v in nn_xr(*args, **kwargs).items()}

    assert set(mixed) == {"TA", "DIC"}
    for variable in mixed:
        np.testing.assert_array_equal(
            mixed[variable],
            0.5 * (lir[variable] + nn[variable]),
            err_msg=f"{variable}: mixed is not the mean of lir and nn",
        )


def test_mixed_returns_none_uncertainties_when_not_computed():
    """Estimates must still come back; only the uncertainties are absent."""
    from PyESPER.mixed import mixed

    n = 64
    rng = np.random.default_rng(5)
    coords = {
        "longitude": rng.uniform(0, 360, n),
        "latitude": rng.uniform(-70, 70, n),
        "depth": rng.uniform(0, 4000, n),
    }
    preds = {
        "salinity": rng.uniform(32, 36, n),
        "temperature": rng.uniform(-1, 25, n),
    }
    estimates, uncertainties = mixed(
        ["TA"], str(REPO_ROOT), coords, preds,
        EstDates=np.full(n, 2002.0), Equations=[8], verbose=False,
        compute_uncertainties=False,
    )
    assert uncertainties is None
    assert np.isfinite(np.asarray(estimates["TA8"], dtype=float)).all()
