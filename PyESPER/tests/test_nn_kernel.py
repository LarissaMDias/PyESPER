"""Tests for the fused tiled neural-net kernel (:mod:`PyESPER.kernels`).

``test_run_nets_batched.py`` already checks the end-to-end numbers against the frozen
per-module reference. These tests cover the things a whole-pipeline comparison cannot
localise: the transfer function's accuracy on its own, the tile decomposition, the
weight packing, and the properties that a ``prange`` rewrite specifically puts at risk.
"""

import numpy as np
import pytest

from PyESPER.kernels import evaluate_nets
from PyESPER.kernels._fastmath import tansig
from PyESPER.kernels.nn_forward import _TILE
from PyESPER.kernels.nn_packing import build_predictor_table, packed_nets
from PyESPER.net_weights import parse_net_weights

ALL_VARIABLES = ["TA", "DIC", "phosphate", "nitrate", "silicate", "oxygen"]


def _code(n_points, equation=8, seed=0, coords=None):
    rng = np.random.default_rng(seed)
    if coords is None:
        coords = (
            rng.uniform(0, 360, n_points),
            rng.uniform(-78, 80, n_points),
            rng.uniform(0, 5500, n_points),
        )
    lon, lat, depth = coords
    shared = {
        "Longitude": lon, "Latitude": lat, "Depth": depth,
        "S": rng.uniform(31, 37, n_points), "T": rng.uniform(-2, 30, n_points),
        "A": rng.uniform(0, 3, n_points), "B": rng.uniform(0, 45, n_points),
        "C": rng.uniform(0, 160, n_points),
    }
    return {f"{v}{equation}": shared for v in ALL_VARIABLES}


# --------------------------------------------------------------------- tansig

@np.vectorize
def _tansig_py(x):
    return tansig(x)


def test_tansig_matches_the_reference_formula():
    """The kernel's transfer function is a hand-rolled exp + Newton reciprocal.

    It must agree with ``2/(1+exp(-2x))-1`` -- what every ESPER net module actually
    computes -- across the whole range the hidden pre-activations can reach.
    """
    x = np.concatenate([
        np.linspace(-25.0, 25.0, 200_001),
        np.array([0.0, 1e-12, -1e-12, 0.5, -0.5, 19.06, -19.06, 20.0, -20.0]),
    ])
    # The kernel clamps at +-20; tanh is saturated to float64 precision well before that.
    reference = 2.0 / (1.0 + np.exp(-2.0 * np.clip(x, -20.0, 20.0))) - 1.0
    got = _tansig_py(x)
    assert np.abs(got - reference).max() < 5e-15


def test_tansig_saturates_and_is_bounded():
    """Far outside the clamp the result must stay a well-behaved +-1, not overflow.

    The clamp is load-bearing: it bounds the exponent so the ``(k + 1023) << 52``
    construction of ``2**k`` cannot overflow into a garbage float.
    """
    extreme = np.array([-1e300, -1e6, -100.0, 100.0, 1e6, 1e300])
    got = _tansig_py(extreme)
    assert np.all(np.isfinite(got))
    assert np.all(np.abs(got) <= 1.0 + 1e-15)
    assert np.all(got[:3] < -0.999999)
    assert np.all(got[3:] > 0.999999)


def test_tansig_is_odd_and_zero_at_zero():
    assert abs(tansig(0.0)) < 1e-15
    for x in (0.1, 1.0, 3.0, 7.5):
        assert abs(tansig(x) + tansig(-x)) < 1e-15


# ------------------------------------------------------------------- tiling

@pytest.mark.parametrize(
    "n_points",
    [1, 2, 7, _TILE - 1, _TILE, _TILE + 1, 2 * _TILE, 3 * _TILE + 7],
)
def test_tile_boundaries(n_points):
    """Results must not depend on how points fall across tile boundaries.

    The kernel processes points in tiles of ``_TILE`` with a padded scratch stride; a
    partial final tile, or an exact multiple, are the classic places for an off-by-one
    to hide. Compared against a single-tile evaluation of the same points.
    """
    code = _code(n_points, seed=n_points)
    tiled = evaluate_nets(code, ALL_VARIABLES, 8)
    single = evaluate_nets(code, ALL_VARIABLES, 8, tile=max(n_points, 1))
    np.testing.assert_array_equal(tiled, single)


def test_zero_points():
    out = evaluate_nets(_code(0), ALL_VARIABLES, 8)
    assert out.shape == (len(ALL_VARIABLES) * 2, 0, 4)


def test_results_are_independent_of_thread_count():
    """Same numbers at 1 thread and at many.

    The direct regression test for the ``prange`` rewrite: each tile writes a disjoint
    slice of the output and owns private scratch, so the thread count must not be
    observable in the results at all -- not even at floating-point level.
    """
    import numba

    code = _code(4 * _TILE + 13, seed=99)
    original = numba.get_num_threads()
    try:
        numba.set_num_threads(1)
        serial = evaluate_nets(code, ALL_VARIABLES, 8)
        numba.set_num_threads(min(16, numba.config.NUMBA_NUM_THREADS))
        parallel = evaluate_nets(code, ALL_VARIABLES, 8)
    finally:
        numba.set_num_threads(original)
    np.testing.assert_array_equal(serial, parallel)


# ------------------------------------------------------------------ packing

@pytest.mark.parametrize("equation", [8, 16])
def test_packed_weights_match_the_source_modules(equation):
    """The flat blobs must be a faithful, bit-exact repacking of each net's weights."""
    packed = packed_nets(tuple(ALL_VARIABLES), equation)
    net = 0
    for name_index, variable in enumerate(ALL_VARIABLES):
        for region_index, region in enumerate(("Atl", "Other")):
            for member_index, member in enumerate((1, 2, 3, 4)):
                reference = parse_net_weights(
                    f"NeuralNetworks.ESPER_{variable}_{equation}_{region}_{member}"
                )
                assert packed.net_row[net] == name_index * 2 + region_index
                assert packed.net_member[net] == member_index
                assert packed.net_nlay[net] == len(reference["layers"])

                first = packed.net_lay0[net]
                for layer, (W, b) in enumerate(reference["layers"]):
                    li = first + layer
                    assert (packed.lay_nout[li], packed.lay_nin[li]) == W.shape
                    w_start = packed.lay_w_off[li]
                    np.testing.assert_array_equal(
                        packed.W_blob[w_start:w_start + W.size].reshape(W.shape), W
                    )
                    b_start = packed.lay_b_off[li]
                    np.testing.assert_array_equal(
                        packed.b_blob[b_start:b_start + b.size], b
                    )
                np.testing.assert_array_equal(packed.x_off[net], reference["x1_xoffset"])
                np.testing.assert_array_equal(packed.x_gain[net], reference["x1_gain"])
                assert packed.y_inv_gain[net] == 1.0 / reference["y1_gain"]
                net += 1
    assert net == packed.n_nets


def test_packed_nets_are_cached_and_read_only():
    first = packed_nets(("TA",), 8)
    assert packed_nets(("TA",), 8) is first
    with pytest.raises(ValueError):
        first.W_blob[0] = 0.0


def test_predictor_columns_are_deduplicated():
    """All six variables share one predictor set at equation 8, so must share columns.

    This is what keeps the input at ~48 bytes/point instead of one ``(n_in, n_points)``
    slab per variable. Correctness does not depend on it -- a missed de-duplication just
    costs memory -- but a *wrong* de-duplication would silently mix variables up, so the
    mapping is checked too.
    """
    code = _code(32, equation=8, seed=5)
    table, src = build_predictor_table(code, tuple(ALL_VARIABLES), 8)
    assert table.shape == (6, 32), f"expected 6 shared columns, got {table.shape[0]}"
    for row in src:
        np.testing.assert_array_equal(row, src[0])

    entry = code["TA8"]
    radians = np.deg2rad(np.asarray(entry["Longitude"]) - 20.0)
    np.testing.assert_allclose(table[src[0, 0]], np.cos(radians))
    np.testing.assert_allclose(table[src[0, 1]], np.sin(radians))
    np.testing.assert_array_equal(table[src[0, 2]], entry["Latitude"])
    np.testing.assert_array_equal(table[src[0, 3]], entry["Depth"])
    np.testing.assert_array_equal(table[src[0, 4]], entry["S"])
    np.testing.assert_array_equal(table[src[0, 5]], entry["T"])


def test_distinct_predictor_arrays_are_not_merged():
    """Two variables given genuinely different inputs must get different columns."""
    code = {k: dict(v) for k, v in _code(16, equation=8, seed=6).items()}
    code["oxygen8"]["S"] = code["oxygen8"]["S"] + 1.0  # a fresh, distinct array
    table, src = build_predictor_table(code, tuple(ALL_VARIABLES), 8)
    oxygen = ALL_VARIABLES.index("oxygen")
    assert src[oxygen, 4] != src[0, 4]
    np.testing.assert_array_equal(table[src[oxygen, 4]], code["oxygen8"]["S"])


def test_nan_inputs_propagate_to_nan_estimates():
    """A NaN predictor must yield NaN, not a silently plausible number.

    ``xr_methods`` filters non-finite points before calling, but equations other than
    8/16 legitimately hand the nets NaN columns when an optional predictor is omitted,
    and the reference implementation propagates those. This is why the kernel's
    ``fastmath`` set deliberately omits ``nnan``/``ninf``.
    """
    code = {k: dict(v) for k, v in _code(8, equation=8, seed=7).items()}
    for entry in code.values():
        entry["T"] = np.asarray(entry["T"], dtype=float).copy()
        entry["T"][3] = np.nan
    out = evaluate_nets(code, ALL_VARIABLES, 8)
    assert np.all(np.isnan(out[:, 3, :])), "NaN input did not propagate"
    assert np.all(np.isfinite(np.delete(out, 3, axis=1))), "NaN leaked to other points"
