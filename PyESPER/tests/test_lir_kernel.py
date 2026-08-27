"""Tests for the fused LIR kernel (:mod:`PyESPER.kernels.lir_forward`).

The kernel replaces three passes -- ``input_AAinds`` splitting every input column by
ocean basin, ``interpolate`` evaluating scipy's RegularGridInterpolator on each half, and
``organize_data`` concatenating/argsorting/reindexing them back -- with one pass that
picks each point's coefficient set by mask.

Those three functions are still present, so they serve as a live oracle here rather than
a frozen fixture. The kernel is written to reproduce scipy's arithmetic exactly (its
searchsorted-then-clip bracket convention, its corner ordering, its weight products) and
runs with fastmath off, so these are equality checks, not tolerance checks.
"""

import numpy as np
import pytest

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(
    not (REPO_ROOT / "Mat_fullgrid").is_dir(),
    reason="Mat_fullgrid/ data directory not present",
)

_COEF_LABELS = ("Intercept", "Coef S", "Coef T", "Coef A", "Coef B", "Coef C")


def _setup(variables, equations, n_points, seed=0):
    """Everything lir() assembles before estimating, for both the old and new paths."""
    from PyESPER.defaults import defaults
    from PyESPER.inputdata_organize import inputdata_organize
    from PyESPER.iterations import iterations
    from PyESPER.lir_uncertainties import measurement_uncertainty_defaults

    rng = np.random.default_rng(seed)
    coords = {
        "longitude": rng.uniform(0, 360, n_points),
        "latitude": rng.uniform(-78, 80, n_points),
        "depth": rng.uniform(0, 5500, n_points),
    }
    preds = {
        "salinity": rng.uniform(31, 37, n_points),
        "temperature": rng.uniform(-2, 30, n_points),
        "phosphate": rng.uniform(0, 4, n_points),
        "nitrate": rng.uniform(0, 45, n_points),
        "silicate": rng.uniform(0, 170, n_points),
        "oxygen": rng.uniform(0, 350, n_points),
    }
    eqs, n, verbose, dates, C, per_kg, uncerts = defaults(
        variables, preds, coords, Equations=equations, verbose=False
    )
    u_pre, du_pre = measurement_uncertainty_defaults(n, preds, uncerts)
    input_all = inputdata_organize(dates, C, preds, u_pre)
    code, _u, _du = iterations(
        variables, equations, per_kg, C, preds, input_all, u_pre, du_pre
    )
    return C, code


def _old_path(variables, equations, C, code):
    """The pre-kernel sequence: split by basin, interpolate each half, merge back."""
    from PyESPER.coefs_AAinds import coefs_AAinds
    from PyESPER.fetch_data import fetch_data
    from PyESPER.input_AAinds import input_AAinds
    from PyESPER.interpolate import interpolate
    from PyESPER.organize_data import organize_data

    lir_data = fetch_data(variables, str(REPO_ROOT))
    aa, other = input_AAinds(C, code)
    gdf, _cs = coefs_AAinds(equations, lir_data)
    aa_lcs, aa_interp, el_lcs, el_interp = interpolate(
        str(REPO_ROOT), gdf, aa, other, verbose=False
    )
    return organize_data(aa_lcs, el_lcs, aa_interp, el_interp, gdf, aa, other)


def _new_path(variables, equations, C, code, **kwargs):
    from PyESPER.coefs_AAinds import coefs_AAinds
    from PyESPER.fetch_data import fetch_data
    from PyESPER.input_AAinds import atlantic_mask
    from PyESPER.kernels.lir_forward import lir_estimates

    lir_data = fetch_data(variables, str(REPO_ROOT))
    gdf, _cs = coefs_AAinds(equations, lir_data)
    return lir_estimates(
        str(REPO_ROOT), gdf, code,
        C["longitude"], C["latitude"], C["depth"],
        atlantic_mask(C["longitude"], C["latitude"]), **kwargs,
    )


@pytest.mark.parametrize("equation", list(range(1, 17)))
def test_matches_the_split_and_merge_path_for_every_equation(equation):
    """All 16 equations, estimates and coefficients, exactly.

    This is the test that matters most for this rewrite. ``organize_data`` decided which
    coefficient slots an equation uses via ``mask = 16 - equation`` and four bit tests;
    the kernel uses an explicit table. A mismatch would silently multiply the wrong
    predictor by the wrong coefficient for some equations and not others.
    """
    variables = ["TA", "oxygen"]
    C, code = _setup(variables, [equation], 3000, seed=equation)

    old_est, old_coef = _old_path(variables, [equation], C, code)
    new_est, new_coef = _new_path(variables, [equation], C, code)

    assert set(new_est) == set(old_est)
    for name in old_est:
        np.testing.assert_array_equal(
            np.asarray(new_est[name], dtype=float),
            np.asarray(old_est[name], dtype=float),
            err_msg=f"estimates differ for {name}",
        )
        for label in _COEF_LABELS:
            np.testing.assert_array_equal(
                np.asarray(new_coef[name][label], dtype=float),
                np.asarray(old_coef[name][label], dtype=float),
                err_msg=f"{label} differs for {name}",
            )


def test_matches_for_a_multi_variable_multi_equation_request():
    """Several variables and equations at once, exercising the combination ordering.

    ``organize_data`` indexed the interpolant's output axis positionally, so the kernel
    has to agree not just on values but on which axis belongs to which combination.
    """
    variables = ["TA", "DIC", "phosphate", "nitrate", "silicate", "oxygen"]
    equations = [8, 16, 1]
    C, code = _setup(variables, equations, 2000, seed=99)

    old_est, old_coef = _old_path(variables, equations, C, code)
    new_est, new_coef = _new_path(variables, equations, C, code)

    assert set(new_est) == set(old_est)
    for name in old_est:
        np.testing.assert_array_equal(
            np.asarray(new_est[name], dtype=float),
            np.asarray(old_est[name], dtype=float),
            err_msg=f"estimates differ for {name}",
        )
        for label in _COEF_LABELS:
            np.testing.assert_array_equal(
                np.asarray(new_coef[name][label], dtype=float),
                np.asarray(old_coef[name][label], dtype=float),
            )


@pytest.mark.parametrize("n_points", [1, 2, 4095, 4096, 4097, 8192, 12295])
def test_tile_boundaries(n_points):
    """Results must not depend on where points fall across the kernel's tiles."""
    variables = ["TA"]
    C, code = _setup(variables, [8], n_points, seed=n_points)
    tiled, _ = _new_path(variables, [8], C, code)
    single, _ = _new_path(variables, [8], C, code, tile=max(n_points, 1))
    np.testing.assert_array_equal(tiled["TA8"], single["TA8"])


def test_points_stay_in_input_order():
    """The old path split by basin, then argsorted an index column to undo it.

    The kernel never reorders, so this checks the two agree point-for-point on a set
    deliberately arranged to interleave basins.
    """
    variables = ["TA"]
    C, code = _setup(variables, [8], 500, seed=7)
    # Alternate Atlantic and Pacific longitudes so a basin split would badly scramble
    # the order if it were not undone correctly.
    C["longitude"][:] = np.where(np.arange(500) % 2 == 0, 330.0, 200.0)
    for entry in code.values():
        entry["Longitude"] = C["longitude"]

    old_est, _ = _old_path(variables, [8], C, code)
    new_est, _ = _new_path(variables, [8], C, code)
    np.testing.assert_array_equal(
        np.asarray(new_est["TA8"], dtype=float),
        np.asarray(old_est["TA8"], dtype=float),
    )


def test_coefficients_can_be_skipped():
    variables = ["TA"]
    C, code = _setup(variables, [8], 256, seed=3)
    with_coef, coef = _new_path(variables, [8], C, code, want_coefficients=True)
    without, empty = _new_path(variables, [8], C, code, want_coefficients=False)
    assert empty == {}
    assert coef["TA8"]["Intercept"].shape == (256,)
    np.testing.assert_array_equal(with_coef["TA8"], without["TA8"])
