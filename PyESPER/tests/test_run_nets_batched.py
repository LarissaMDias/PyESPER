"""Numerical-equivalence regression test for the batched ``run_nets`` rewrite.

The rewrite (see ``PyESPER/run_nets.py``'s module docstring) changes *how* the
neural-net estimates are computed (grouped, vectorized batches instead of one
Python-level call per net) but must produce the same numbers as before. This
compares the current ``run_nets`` against ``_legacy_run_nets_reference.py`` (a
frozen, untouched copy of the pre-rewrite implementation) on synthetic-but-
realistic ocean data, across every ESPER target variable and both supported
equations, run through the exact same code path a real ``nn()`` call uses.
"""

import numpy as np
import pytest

from PyESPER.run_nets import run_nets
from PyESPER.tests._legacy_run_nets_reference import run_nets_legacy

# All six ROMS/MARBL-mapped ESPER target variables (see
# roms_tools.setup.esper.ROMS_TO_ESPER in the roms-tools repo) -- the actual set
# a real cstar-forge/roms-tools run requests together in one ``nn()`` call.
ALL_VARIABLES = ["TA", "DIC", "phosphate", "nitrate", "silicate", "oxygen"]


def _make_code(n_points, equation, seed=0, coords=None):
    """Build a ``code`` dict shaped exactly like ``iterations()``'s output: one
    entry per ``{variable}{equation}`` name, each a dict of numpy arrays over
    realistic ocean ranges. ``A``/``B``/``C`` are populated even though equation
    8/16 don't use them -- ``run_nets`` unconditionally reads all five columns
    for every entry (see ``iterations.py``'s ``data_cols`` tuple), so a
    realistic ``code`` dict always carries them regardless of which equation is
    requested.

    ``coords`` lets a caller reuse one ``(lon, lat, depth)`` triple across several
    calls. That is not a convenience -- it is what ``iterations()`` actually does:
    it builds ``metadata_dict`` once and ``update()``s the *same* Longitude/
    Latitude/Depth arrays into every entry (see ``iterations.py``), so in real use
    every entry of ``code`` shares one set of coordinates.
    """
    rng = np.random.default_rng(seed)
    if coords is None:
        coords = (
            rng.uniform(-180, 180, n_points),
            rng.uniform(-75, 75, n_points),
            rng.uniform(0, 5000, n_points),
        )
    lon, lat, depth = coords
    salinity = rng.uniform(32, 36, n_points)
    temperature = rng.uniform(-2, 25, n_points)
    phosphate = rng.uniform(0, 3, n_points)
    nitrate = rng.uniform(0, 40, n_points)
    silicate = rng.uniform(0, 150, n_points)

    code = {}
    for v in ALL_VARIABLES:
        name = f"{v}{equation}"
        code[name] = {
            "Longitude": lon,
            "Latitude": lat,
            "Depth": depth,
            "S": salinity,
            "T": temperature,
            "A": phosphate,
            "B": nitrate,
            "C": silicate,
        }
    return code, coords


@pytest.mark.parametrize("equation", [8, 16])
@pytest.mark.parametrize("n_points", [1, 2, 37, 5_000])
def test_run_nets_matches_legacy_reference(equation, n_points):
    """The batched implementation must reproduce the original per-net
    implementation's output for every requested variable and both regions, at
    point counts spanning "degenerate" (1-2 points -- exercises no hidden
    broadcasting edge cases) through "realistic-ish" (5,000).
    """
    code, _ = _make_code(
        n_points, equation, seed=hash((equation, n_points)) % (2**32)
    )

    est_atl_new, est_other_new = run_nets(ALL_VARIABLES, [equation], code)
    est_atl_old, est_other_old = run_nets_legacy(ALL_VARIABLES, [equation], code)

    assert set(est_atl_new) == set(est_atl_old)
    assert set(est_other_new) == set(est_other_old)

    for name in est_atl_new:
        # rtol/atol: same arithmetic, reorganized into batched BLAS calls --
        # differences should only ever be floating-point-non-associativity-level
        # (a different summation order inside matmul), not a real numerical
        # divergence. 1e-9 relative is generously tight for that.
        np.testing.assert_allclose(
            est_atl_new[name],
            est_atl_old[name],
            rtol=1e-9,
            atol=1e-12,
            err_msg=f"Atlantic/Arctic mismatch for {name!r}",
        )
        np.testing.assert_allclose(
            est_other_new[name],
            est_other_old[name],
            rtol=1e-9,
            atol=1e-12,
            err_msg=f"Other-region mismatch for {name!r}",
        )
        assert est_atl_new[name].shape == (n_points, 4)
        assert est_other_new[name].shape == (n_points, 4)


def test_run_nets_single_variable_matches_legacy():
    """Also check the single-variable call shape (not just the full 6-variable
    batch) -- a smaller, degenerate case of the same grouping logic.
    """
    code, _ = _make_code(500, 8, seed=1)
    new_atl, new_other = run_nets(["oxygen"], [8], code)
    old_atl, old_other = run_nets_legacy(["oxygen"], [8], code)
    np.testing.assert_allclose(
        new_atl["oxygen8"], old_atl["oxygen8"], rtol=1e-9, atol=1e-12
    )
    np.testing.assert_allclose(
        new_other["oxygen8"], old_other["oxygen8"], rtol=1e-9, atol=1e-12
    )


def test_run_nets_both_equations_together_matches_legacy():
    """Equations 8 and 16 have different input widths (6 vs 5 predictor columns,
    see ``net_weights.parse_net_weights``) -- requesting both together in one
    call exercises that the per-equation grouping in ``run_nets`` doesn't mix
    them up.
    """
    # Both equations must share one set of coordinates, because that is the only
    # thing ``iterations()`` ever produces (see ``_make_code``). Giving them
    # *different* coordinates would not test anything real: the legacy reference
    # reads Longitude/Latitude/Depth from whichever entry it happens to iterate
    # first and silently applies them to every other entry
    # (``_legacy_run_nets_reference.py``, the ``if cosd is None`` guard), so such a
    # ``code`` dict makes the oracle disagree with any implementation that honours
    # each entry's own coordinates -- including the current one. See
    # ``test_per_entry_coordinates_are_honoured`` below.
    code_8, coords = _make_code(200, 8, seed=2)
    code_16, _ = _make_code(200, 16, seed=3, coords=coords)
    code = {**code_8, **code_16}
    new_atl, new_other = run_nets(ALL_VARIABLES, [8, 16], code)
    old_atl, old_other = run_nets_legacy(ALL_VARIABLES, [8, 16], code)
    for name in old_atl:
        np.testing.assert_allclose(new_atl[name], old_atl[name], rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(
            new_other[name], old_other[name], rtol=1e-9, atol=1e-12
        )


def test_per_entry_coordinates_are_honoured():
    """Each ``code`` entry's own Longitude/Latitude/Depth must drive its own nets.

    The legacy reference (and the batched numpy implementation that preceded the
    current kernel) computed the geographic predictor columns *once*, from whichever
    entry came first in ``code``, and reused them for every other entry -- see the
    ``if cosd is None`` guard in ``_legacy_run_nets_reference.py``. That is latent:
    ``iterations()`` always hands every entry the same coordinate arrays, so the two
    behaviours are indistinguishable in production. It is still wrong, and the
    current implementation does not do it.

    This pins the corrected behaviour directly, without reference to the oracle:
    perturbing one entry's coordinates must change that entry's estimates and leave
    the others alone.
    """
    code, coords = _make_code(64, 8, seed=11)
    baseline_atl, baseline_other = run_nets(ALL_VARIABLES, [8], code)

    moved = {name: dict(entry) for name, entry in code.items()}
    moved["oxygen8"]["Latitude"] = code["oxygen8"]["Latitude"] - 25.0
    moved_atl, moved_other = run_nets(ALL_VARIABLES, [8], moved)

    assert not np.allclose(moved_atl["oxygen8"], baseline_atl["oxygen8"]), (
        "moving oxygen8's latitude did not change its estimate -- the entry's own "
        "coordinates are being ignored"
    )
    for name in baseline_atl:
        if name == "oxygen8":
            continue
        np.testing.assert_array_equal(
            moved_atl[name], baseline_atl[name],
            err_msg=f"{name} changed when only oxygen8's coordinates moved",
        )
        np.testing.assert_array_equal(moved_other[name], baseline_other[name])
