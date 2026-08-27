"""Flatten the ESPER neural nets into contiguous arrays a numba kernel can walk.

:func:`PyESPER.net_weights.parse_net_weights` already turns one
``NeuralNetworks.ESPER_*`` module into a dict of numpy arrays. That is the right shape
for numpy code and the wrong shape for a jitted kernel: a Python list of per-layer
``(W, b)`` tuples cannot be indexed from nopython code, and dispatching per net means
re-walking the point array once per net.

This module packs *all* the nets a call needs -- every requested variable x both regions
x all four ensemble members -- into one flat, ragged-but-indexable set of arrays, so the
kernel can loop over nets internally and touch the point data exactly once.

"Ragged" matters: the four ensemble members do not share an architecture. For equation 8
they are 6->40->1, 6->20->20->1, 6->25->15->1 and 6->30->10->1, so a rectangular
``(G, layers, ...)`` layout would not fit. Instead there is a flat layer table indexed by
``net_lay0[g] .. net_lay0[g] + net_nlay[g]``, and one contiguous blob holding every
layer's weights end to end.

Predictor de-duplication
------------------------
``iterations.py`` hands each ``{variable}{equation}`` entry the *same* underlying numpy
arrays for its S/T/A/B/C slots when the mapping works out that way -- and for equations 8
and 16 it always does, for every variable. So instead of materialising a
``(n_nets, n_in, n_points)`` predictor stack (which is what the previous implementation
did, duplicating one array up to 12 times per ensemble member), the distinct columns are
collected once by object identity into a ``(n_columns, n_points)`` table and each net
gets an index vector into it. For the production case that is 6 columns total --
cos(lon-20), sin(lon-20), lat, depth, S, T -- shared by all 48 nets.
"""

from __future__ import annotations

import threading

import numpy as np

from PyESPER.net_weights import parse_net_weights

# The regions and ensemble members present in the NeuralNetworks archive, in the order
# the estimates are assembled in.
REGIONS = ("Atl", "Other")
MEMBERS = (1, 2, 3, 4)

# Which of S/T/A/B/C each equation feeds to its nets, after the four geographic columns.
# Mirrors run_nets.equation_map; kept here so the packing is self-contained.
EQUATION_PREDICTORS = {
    1: ("S", "T", "A", "B", "C"),
    2: ("S", "T", "A", "C"),
    3: ("S", "T", "B", "C"),
    4: ("S", "T", "C"),
    5: ("S", "T", "A", "B"),
    6: ("S", "T", "A"),
    7: ("S", "T", "B"),
    8: ("S", "T"),
    9: ("S", "A", "B", "C"),
    10: ("S", "A", "C"),
    11: ("S", "B", "C"),
    12: ("S", "C"),
    13: ("S", "A", "B"),
    14: ("S", "A"),
    15: ("S", "B"),
    16: ("S",),
}


class PackedNets:
    """Flat arrays describing every net for one ``(variables, equation)`` request."""

    __slots__ = (
        "variables", "equation", "n_in", "n_nets", "max_units",
        "W_blob", "b_blob", "lay_w_off", "lay_b_off", "lay_nout", "lay_nin",
        "net_lay0", "net_nlay", "x_off", "x_gain", "x_ymin",
        "y_xoffset", "y_inv_gain", "y_ymin", "net_row", "net_member",
    )

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        # Cached objects are shared across threads and calls; make accidental mutation
        # an error rather than silent corruption of every later call's weights.
        for name in self.__slots__:
            value = getattr(self, name)
            if isinstance(value, np.ndarray):
                value.flags.writeable = False


_CACHE: dict[tuple, PackedNets] = {}
_LOCK = threading.Lock()


def packed_nets(variables, equation) -> PackedNets:
    """Return (and memoise) the packed weights for ``variables`` x ``equation``.

    Double-checked locking. ``functools.lru_cache`` would be *correct* here -- the GIL
    makes the dict store atomic -- but it has a non-atomic miss window, so two threads
    arriving together would both run the build, and the build parses ~48 source files
    with ``ast`` and allocates the blobs. One lock avoids the duplicated work.
    """
    key = (tuple(variables), int(equation))
    packed = _CACHE.get(key)
    if packed is not None:
        return packed
    with _LOCK:
        packed = _CACHE.get(key)
        if packed is None:
            packed = _build(key[0], key[1])
            _CACHE[key] = packed
    return packed


def _build(variables, equation) -> PackedNets:
    n_names = len(variables)
    weights = []
    net_row = []      # output row: which (variable, region) pair this net contributes to
    net_member = []   # ensemble member index 0..3

    for name_index, variable in enumerate(variables):
        for region_index, region in enumerate(REGIONS):
            for member_index, member in enumerate(MEMBERS):
                module = f"NeuralNetworks.ESPER_{variable}_{equation}_{region}_{member}"
                weights.append(parse_net_weights(module))
                net_row.append(name_index * len(REGIONS) + region_index)
                net_member.append(member_index)

    n_nets = len(weights)
    n_in = int(weights[0]["x1_gain"].shape[0])
    for index, w in enumerate(weights):
        if int(w["x1_gain"].shape[0]) != n_in:
            raise ValueError(
                f"net {index} for equation {equation} takes {w['x1_gain'].shape[0]} "
                f"inputs but the first takes {n_in}; nets for one equation must agree"
            )

    lay_nout, lay_nin, lay_w_off, lay_b_off = [], [], [], []
    w_parts, b_parts = [], []
    net_lay0, net_nlay = [], []
    w_cursor = b_cursor = 0

    for w in weights:
        net_lay0.append(len(lay_nout))
        for (W, b) in w["layers"]:
            lay_nout.append(W.shape[0])
            lay_nin.append(W.shape[1])
            lay_w_off.append(w_cursor)
            lay_b_off.append(b_cursor)
            w_parts.append(np.ascontiguousarray(W, dtype=np.float64).ravel())
            b_parts.append(np.ascontiguousarray(b, dtype=np.float64).ravel())
            w_cursor += W.size
            b_cursor += b.size
        net_nlay.append(len(w["layers"]))

    # The kernel's two scratch buffers must hold the widest thing they ever carry: the
    # input row count on the way in, or any layer's output width thereafter.
    max_units = max(n_in, max(lay_nout))

    return PackedNets(
        variables=tuple(variables),
        equation=int(equation),
        n_in=n_in,
        n_nets=n_nets,
        max_units=int(max_units),
        W_blob=np.concatenate(w_parts),
        b_blob=np.concatenate(b_parts),
        lay_w_off=np.array(lay_w_off, dtype=np.int64),
        lay_b_off=np.array(lay_b_off, dtype=np.int64),
        lay_nout=np.array(lay_nout, dtype=np.int64),
        lay_nin=np.array(lay_nin, dtype=np.int64),
        net_lay0=np.array(net_lay0, dtype=np.int64),
        net_nlay=np.array(net_nlay, dtype=np.int64),
        x_off=np.ascontiguousarray(
            np.stack([w["x1_xoffset"] for w in weights]), dtype=np.float64
        ),
        x_gain=np.ascontiguousarray(
            np.stack([w["x1_gain"] for w in weights]), dtype=np.float64
        ),
        x_ymin=np.array([w["x1_ymin"] for w in weights], dtype=np.float64),
        y_xoffset=np.array([w["y1_xoffset"] for w in weights], dtype=np.float64),
        # Precomputed reciprocal: the output mapminmax divides by gain, and a division
        # in the kernel's inner tail would scalarise the surrounding loop (see
        # _fastmath's module docstring).
        y_inv_gain=np.array([1.0 / w["y1_gain"] for w in weights], dtype=np.float64),
        y_ymin=np.array([w["y1_ymin"] for w in weights], dtype=np.float64),
        net_row=np.array(net_row, dtype=np.int64),
        net_member=np.array(net_member, dtype=np.int64),
    )


def build_predictor_table(code, variables, equation):
    """Collect the distinct predictor columns for one equation into ``(n_cols, Q)``.

    Returns ``(table, src)`` where ``table[c]`` is one predictor column over all points
    and ``src[name_index, j]`` is the column index feeding input ``j`` of the nets for
    ``variables[name_index]``.

    Columns are de-duplicated by object identity, which is exactly right here: two
    entries of ``code`` that use the same physical measurement hold the *same* ndarray
    (see ``iterations.py``, which assigns from a shared ``data_cols`` tuple). If a caller
    ever passes equal-but-distinct arrays the result is still correct, just with a larger
    table -- identity is an optimisation, never a correctness assumption.
    """
    columns: list[np.ndarray] = []
    by_id: dict[int, int] = {}

    def column_index(array):
        array = np.ascontiguousarray(array, dtype=np.float64)
        key = id(array)
        index = by_id.get(key)
        if index is None:
            index = len(columns)
            by_id[key] = index
            columns.append(array)
        return index

    # Geographic inputs are derived, so they are keyed on the identity of the array they
    # are derived *from* rather than on their own (freshly allocated) identity.
    derived: dict[tuple[int, str], int] = {}

    def derived_index(source, kind):
        key = (id(source), kind)
        index = derived.get(key)
        if index is None:
            radians = np.deg2rad(np.asarray(source, dtype=np.float64) - 20.0)
            values = np.cos(radians) if kind == "cos" else np.sin(radians)
            index = len(columns)
            columns.append(np.ascontiguousarray(values))
            derived[key] = index
        return index

    predictors = EQUATION_PREDICTORS[int(equation)]
    src = np.empty((len(variables), 4 + len(predictors)), dtype=np.int64)

    for name_index, variable in enumerate(variables):
        entry = code[f"{variable}{equation}"]
        longitude = entry["Longitude"]
        src[name_index, 0] = derived_index(longitude, "cos")
        src[name_index, 1] = derived_index(longitude, "sin")
        src[name_index, 2] = column_index(entry["Latitude"])
        src[name_index, 3] = column_index(entry["Depth"])
        for offset, key in enumerate(predictors):
            src[name_index, 4 + offset] = column_index(entry[key])

    table = np.ascontiguousarray(np.stack(columns)) if columns else np.zeros((0, 0))
    return table, src
