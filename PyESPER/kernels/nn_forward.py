"""The fused, tiled ESPER neural-net forward pass.

What changed and why
--------------------
The previous implementation (``run_nets.batched_forward``) evaluated the nets as a
sequence of whole-array numpy/BLAS operations: for each ensemble member it stacked every
net's predictors into a ``(G, n_in, Q)`` array and carried a ``(G, hidden, Q)``
activation array from layer to layer. At production point counts that activation array
alone is tens of gigabytes -- measured at roughly **10 KB per point** -- and every one of
those bytes is written to DRAM by one operation and read back by the next. The
arithmetic itself (about 22,200 multiply-adds per point) is nowhere near enough work to
justify that traffic, so the whole thing ran at memory speed rather than compute speed.

This module inverts the loop order. Points are walked in **tiles** of a few hundred, and
for each tile *every* net is evaluated before moving on. The activations for a tile are
two small scratch buffers that stay resident in L2, so no intermediate ever reaches DRAM
and no array is sized by the point count except the inputs and the outputs.

Consequences:

* Memory drops from ~10 KB/point to the inputs plus outputs -- ~430 bytes/point for the
  production six-variable request, and independent of net count or hidden width.
* No BLAS. That is a feature here, not a sacrifice: callers commonly pin BLAS to one
  thread to avoid oversubscription (cstar-forge does), which was silently
  single-threading the old ``np.matmul`` anyway.
* One ``prange`` over tiles is the only parallelism, which is what
  :mod:`PyESPER.concurrency` needs in order to be able to guarantee it is the only
  parallelism in the process.

Loop-order notes (these are load-bearing, not stylistic)
--------------------------------------------------------
* The point index is the **innermost** loop everywhere, over contiguous memory, with the
  weight held as a loop-invariant scalar. That is the canonical ``axpy`` shape LLVM turns
  into packed FMAs. Making the net or unit index innermost instead costs roughly an order
  of magnitude.
* The scratch buffers are allocated **inside** the ``prange`` body rather than passed in.
  Besides giving each thread its own copy for free, it is what lets LLVM prove they do
  not alias the inputs -- numba has no ``restrict``, and the identical loop measured
  1.79 ns/element with locally-allocated output versus 6.4 ns/element when both source
  and destination were arguments.
* The buffer row stride is ``tile + _PAD``, deliberately not a multiple of 4 KB. With a
  bare power-of-two stride the two scratch buffers' rows collide in the 8-way 32 KB L1,
  which measured 14% slower at tile 256 and 31% slower at tile 2048.
"""

from __future__ import annotations

import numpy as np
from numba import get_num_threads, njit, prange

from PyESPER.kernels._fastmath import tansig
from PyESPER.kernels.nn_packing import build_predictor_table, packed_nets

# Points per tile. Sized so that one tile's scratch (2 buffers x max_units x stride x 8 B
# ~= 383 KB at max_units 40) fits the 512 KB private L2 of a Zen 2 core, while the slice
# actually live across the innermost loop (one output unit's row, 4 KB) sits in L1.
_TILE = 512

# Row padding, in elements. Breaks the 4 KB aliasing between the two scratch buffers --
# see the module docstring.
_PAD = 8

# fastmath, minus `nnan`/`ninf`. The speed comes from `contract` (FMA fusion) and
# `reassoc`; `nnan`/`ninf` would additionally promise the kernel never sees a NaN, which
# is not something this module can honour. Equations other than 8/16 legitimately feed
# NaN columns to the nets when a caller omits an optional predictor, and the reference
# implementation propagates those to NaN estimates. Keeping NaN semantics costs nothing
# measurable here and preserves that behaviour.
_FASTMATH = {"nsz", "arcp", "contract", "afn", "reassoc"}


@njit(parallel=True, fastmath=_FASTMATH, nogil=True, cache=True)
def _forward(
    table, src, n_in,
    W_blob, b_blob, lay_w_off, lay_b_off, lay_nout, lay_nin,
    net_lay0, net_nlay, x_off, x_gain, x_ymin,
    y_xoffset, y_inv_gain, y_ymin,
    net_row, net_member, out, tile, max_units,
):
    """Evaluate every packed net over every point, writing ``out[row, point, member]``.

    ``table`` is ``(n_columns, Q)`` of de-duplicated predictor columns; ``src`` is
    ``(n_names, n_in)`` mapping each variable's net inputs to those columns. ``out`` is
    ``(n_names * 2, Q, 4)`` and is written, not accumulated, so it need not be zeroed.
    """
    n_points = table.shape[1]
    n_nets = net_lay0.shape[0]
    n_tiles = (n_points + tile - 1) // tile
    stride = tile + 8

    for t in prange(n_tiles):
        p0 = t * tile
        n = n_points - p0
        if n > tile:
            n = tile

        # Thread-private, and local so LLVM can see they do not alias `table`/`out`.
        a = np.empty((max_units, stride), dtype=np.float64)
        b = np.empty((max_units, stride), dtype=np.float64)

        for g in range(n_nets):
            name = net_row[g] >> 1  # two regions per variable, region is the low bit
            ymin = x_ymin[g]

            # mapminmax_apply, straight into the layer-input buffer.
            for j in range(n_in):
                column = src[name, j]
                offset = x_off[g, j]
                gain = x_gain[g, j]
                for p in range(n):
                    a[j, p] = (table[column, p0 + p] - offset) * gain + ymin

            first = net_lay0[g]
            n_layers = net_nlay[g]
            for layer in range(n_layers):
                li = first + layer
                n_out = lay_nout[li]
                n_inp = lay_nin[li]
                w_base = lay_w_off[li]
                b_base = lay_b_off[li]

                for o in range(n_out):
                    bias = b_blob[b_base + o]
                    for p in range(n):
                        b[o, p] = bias
                    row = w_base + o * n_inp
                    for k in range(n_inp):
                        weight = W_blob[row + k]
                        for p in range(n):
                            b[o, p] += weight * a[k, p]
                    if layer < n_layers - 1:
                        for p in range(n):
                            b[o, p] = tansig(b[o, p])

                # Swap rather than copy; the next layer reads `a` and writes `b`.
                a, b = b, a

            # The output layer is always a single unit, so the estimate is row 0.
            # mapminmax_reverse, folded into the store.
            out_row = net_row[g]
            member = net_member[g]
            xoffset = y_xoffset[g]
            inv_gain = y_inv_gain[g]
            ymin_out = y_ymin[g]
            for p in range(n):
                out[out_row, p0 + p, member] = (
                    a[0, p] - ymin_out
                ) * inv_gain + xoffset


def evaluate_nets(code, variables, equation, tile=_TILE):
    """Evaluate all ESPER nets for ``variables`` at one ``equation``.

    Parameters
    ----------
    code : dict
        ``iterations()``-shaped input: one entry per ``f"{variable}{equation}"`` name,
        each holding ``Longitude``/``Latitude``/``Depth`` plus the S/T/A/B/C columns.
    variables, equation
        The requested target variables and the equation number.

    Returns
    -------
    ndarray
        ``(len(variables) * 2, n_points, 4)``: for variable ``i``, row ``2*i`` is the
        Atlantic/Arctic net's estimates and row ``2*i + 1`` the other-region net's, with
        the four ensemble members on the last axis. Rows are contiguous, so a caller
        wanting the historical ``(n_points, 4)`` per-region array can take a view rather
        than a copy.
    """
    variables = tuple(variables)
    packed = packed_nets(variables, equation)
    table, src = build_predictor_table(code, variables, equation)

    if src.shape[1] != packed.n_in:
        raise ValueError(
            f"equation {equation} supplies {src.shape[1]} predictor columns but its "
            f"nets take {packed.n_in} inputs"
        )

    n_points = table.shape[1]
    out = np.empty((len(variables) * 2, n_points, 4), dtype=np.float64)
    if n_points == 0:
        return out

    _forward(
        table, src, packed.n_in,
        packed.W_blob, packed.b_blob, packed.lay_w_off, packed.lay_b_off,
        packed.lay_nout, packed.lay_nin, packed.net_lay0, packed.net_nlay,
        packed.x_off, packed.x_gain, packed.x_ymin,
        packed.y_xoffset, packed.y_inv_gain, packed.y_ymin,
        packed.net_row, packed.net_member, out, int(tile), packed.max_units,
    )
    return out


def warmup_nn(variables=("TA",), equation=8):
    """Compile the kernel on a tiny input, so the first real call does not pay for it."""
    n = 4
    entry = {
        "Longitude": np.linspace(0.0, 300.0, n),
        "Latitude": np.linspace(-60.0, 60.0, n),
        "Depth": np.linspace(0.0, 3000.0, n),
        "S": np.full(n, 34.5),
        "T": np.full(n, 10.0),
        "A": np.full(n, 1.0),
        "B": np.full(n, 20.0),
        "C": np.full(n, 30.0),
    }
    code = {f"{v}{equation}": entry for v in variables}
    evaluate_nets(code, variables, equation)
