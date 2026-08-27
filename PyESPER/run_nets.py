"""Run the ESPER neural nets over a set of points.

Every ``NeuralNetworks.ESPER_{variable}_{equation}_{Atl,Other}_{1..4}`` module is a
MATLAB ``genFunction()`` export: a small 2- or 3-layer feedforward net with its weights
hard-coded as module-level literals. A request for six variables at one equation means
48 of them (6 variables x 2 regions x 4 ensemble members), all evaluated at the same
points.

This function is now a thin adapter. The arithmetic lives in
:func:`PyESPER.kernels.evaluate_nets`, a fused tiled numba kernel; everything here does
is unpack the caller's ``code`` dict and reshape the kernel's output into the
``EstAtl``/``EstOther`` contract the rest of the package expects.

History, because the shape of the code no longer shows it
---------------------------------------------------------
The original implementation imported each net's module in turn and called its
``PyESPER_NN(X)``, so the same point array was walked 48 times and each net paid its own
``mapminmax``/``tansig``/matmul overhead. That was replaced by a batched numpy version
(``batched_forward``) which grouped nets of identical architecture and evaluated them
with one ``np.matmul`` per layer -- about 8x faster, but it carried a
``(n_nets, hidden_width, n_points)`` activation array from layer to layer, which is
**~10 KB per point** and lands in DRAM. At production point counts that both dominated
runtime (the arithmetic is only ~22,200 multiply-adds per point; the memory traffic was
roughly 63 KB per point) and made memory the binding constraint.

The current kernel walks points in cache-resident tiles and evaluates all 48 nets per
tile, so no intermediate is ever sized by the point count. Measured on the target
machine, six variables at equation 8, against the batched numpy version it replaced:

=========== ============ ============ ==========
n_points    batched      fused        speedup
=========== ============ ============ ==========
10,000      0.194 s      0.018 s      10.8x
100,000     1.341 s      0.040 s      33.8x
1,000,000   8.979 s      0.140 s      63.9x
4,000,000   39.812 s     0.548 s      72.6x
=========== ============ ============ ==========

with peak memory going from 10,538 to 448 bytes per point (23.5x). Agreement with the
original per-module implementation (still present, and still the test oracle, at
``PyESPER/tests/_legacy_run_nets_reference.py``) is ~1e-14 relative to each variable's
scale; worst observed absolute difference is 3e-11 on estimates of order 3,000.
"""

import numpy as np

from PyESPER.concurrency import kernel_lock
from PyESPER.kernels import evaluate_nets


def run_nets(DesiredVariables, Equations, code={}):
    """
    Running neural nets

    Inputs:
        DesiredVariables: List of variables for estimates
        Equations: List of desired equations
        code: Dictionary of preprocessed measurements

    Outputs:
        EstAtl: Dictionary of estimates for the Atlantic and Arctic
            Oceans
        EstOther: Dictionary of estimates for not Atlantic/Arctic

    Each value is ``(n_points, 4)`` -- one column per ensemble member -- keyed by
    ``f"{variable}{equation}"``, unchanged from previous versions.
    """
    variables = tuple(DesiredVariables)
    EstAtl, EstOther = {}, {}

    # Reentrant, so this is a no-op when the caller already holds it (the usual case
    # via ``xr_methods._estimate_block``). It matters for direct callers such as
    # ``pH_DIC_nn_adjustment``: the kernel below is ``prange``-parallel and claims the
    # whole numba pool, so two threads entering it at once is the deadlock this guards
    # against. See :mod:`PyESPER.concurrency`.
    with kernel_lock():
        for equation in Equations:
            # (n_variables * 2, n_points, 4); rows alternate Atlantic, Other.
            estimates = evaluate_nets(code, variables, equation)
            for index, variable in enumerate(variables):
                name = f"{variable}{equation}"
                # Views, not copies -- rows of `estimates` are contiguous.
                EstAtl[name] = estimates[index * 2]
                EstOther[name] = estimates[index * 2 + 1]

    return EstAtl, EstOther
