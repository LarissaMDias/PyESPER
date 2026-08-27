"""Static extraction of ESPER neural-net weights, for batched (vectorized) inference.

Background
----------
Each ``NeuralNetworks.ESPER_{variable}_{equation}_{Atl,Other}_{1..4}`` module is a
MATLAB Neural Network Toolbox ``genFunction()`` export, transliterated to Python:
the forward pass (``mapminmax_apply`` -> N tansig hidden layers -> a final linear
layer -> ``mapminmax_reverse``) is boilerplate, and the only thing that actually
varies between modules is a handful of literal weight/bias arrays and the two
input/output normalization dictionaries (``x1_step1``/``y1_step1``), all assigned
as local variables at the top of each module's ``PyESPER_NN`` function body.

Historically each module was imported and called one at a time (``run_nets.py``'s
original per-ensemble-member loop), so this per-call overhead -- and the fact each
call runs its own independent ``tansig``/``mapminmax`` numpy ops over the full point
count -- dominated runtime once point counts reached the millions (see the
``run_nets``/``batched_forward`` docstrings for the actual numbers and how batching
fixes it).

Getting from "one function per net" to "one batched call across many nets" requires
the raw weight *arrays*, not the ready-to-call function -- and the arrays are function
*locals*, not module-level names, so a plain ``import`` doesn't expose them. Two ways
to get them: (a) call the function once with dummy input and monkey-patch/trace to
capture its locals, or (b) statically parse the (deterministically-shaped, literal-
only) assignments straight out of the source. (b) is what this module does: it is
read-only (never executes the target module), trivially cacheable, and -- because it
only ever calls ``ast.literal_eval`` on assignments whose right-hand side is a plain
literal (numbers/lists/dicts of numbers) -- cannot execute arbitrary code, unlike
``importlib.import_module`` on a module of unknown provenance.

This module has ONE job: turn a module name into structured numpy arrays. It has no
opinion about batching/grouping -- see ``run_nets.py`` for that.
"""

from __future__ import annotations

import ast
import functools
import importlib.util
from pathlib import Path

import numpy as np

# The exact local-variable names genFunction() emits, in the order a forward pass
# needs them. Layer 1's weight is always named "IW1_1" (input weight); every
# subsequent layer k>=2's weight is named "LW{k}_{k-1}" (layer weight, this layer
# from the previous one) -- both are un-ambiguous, deterministic MATLAB-export
# conventions, not something this fork invented.
_INPUT_NORM_NAME = "x1_step1"
_OUTPUT_NORM_NAME = "y1_step1"


def _layer_weight_name(layer_index: int) -> str:
    """MATLAB genFunction()'s deterministic weight-variable name for layer
    ``layer_index`` (1-based): ``"IW1_1"`` for the first layer, ``"LW{k}_{k-1}"``
    for every later one.
    """
    if layer_index == 1:
        return "IW1_1"
    return f"LW{layer_index}_{layer_index - 1}"


@functools.lru_cache(maxsize=None)
def parse_net_weights(module_name: str) -> dict:
    """Statically extract one ``ESPER_*`` module's weights, without importing it.

    Parameters
    ----------
    module_name : str
        Dotted module name, e.g. ``"NeuralNetworks.ESPER_oxygen_8_Atl_1"``.

    Returns
    -------
    dict
        ``{"x1_xoffset": (n_in,) float64, "x1_gain": (n_in,) float64,
        "x1_ymin": float, "y1_xoffset": float, "y1_gain": float, "y1_ymin": float,
        "layers": [(W, b), ...]}`` where ``layers`` is in forward-pass order, each
        ``W`` has shape ``(n_out, n_in)`` and each ``b`` has shape ``(n_out,)``. The
        last layer is always the linear output layer (no activation applied to it);
        every earlier layer is followed by ``tansig`` -- see ``batched_forward``.

    Notes
    -----
    Cached (``lru_cache``): every module is only ever parsed once per process, same
    as the old ``_load_net``'s import-once-and-cache behavior in ``run_nets.py``.
    """
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        raise ImportError(f"cannot locate source file for module {module_name!r}")
    source = Path(spec.origin).read_text()
    tree = ast.parse(source, filename=spec.origin)

    func = next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "PyESPER_NN"
        ),
        None,
    )
    if func is None:
        raise ValueError(f"{module_name!r} has no top-level 'PyESPER_NN' function")

    # Every weight/bias/normalization-dict assignment in these modules is a bare
    # `name = <literal>` at the top of the function body -- collect them by name.
    # Non-literal assignments (e.g. `TS = len(X[0])`, further down the function,
    # once the actual math starts) simply fail `ast.literal_eval` and are skipped;
    # we only want the constant weight data, not the forward-pass logic itself.
    values: dict[str, object] = {}
    for node in func.body:
        if not (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            continue
        name = node.targets[0].id
        try:
            values[name] = ast.literal_eval(node.value)
        except (ValueError, SyntaxError):
            continue

    missing = {_INPUT_NORM_NAME, _OUTPUT_NORM_NAME, "b1", "IW1_1"} - values.keys()
    if missing:
        raise ValueError(
            f"{module_name!r}: expected genFunction()-style locals not found: "
            f"{sorted(missing)}"
        )

    x1 = values[_INPUT_NORM_NAME]
    y1 = values[_OUTPUT_NORM_NAME]

    layers: list[tuple[np.ndarray, np.ndarray]] = []
    layer_index = 1
    while f"b{layer_index}" in values:
        w_name = _layer_weight_name(layer_index)
        if w_name not in values:
            raise ValueError(
                f"{module_name!r}: found 'b{layer_index}' but no '{w_name}' -- "
                "inconsistent genFunction() export"
            )
        # b1/b2/... are always a column (or, for a 1-unit layer, a bare scalar) --
        # np.ravel handles both uniformly. W is always (n_out, n_in) already,
        # except a final 1-unit output layer, which genFunction() sometimes emits
        # as a flat (n_in,) vector rather than (1, n_in) -- reshape defensively.
        W = np.atleast_2d(np.asarray(values[w_name], dtype=np.float64))
        b = np.ravel(np.asarray(values[f"b{layer_index}"], dtype=np.float64))
        if W.shape[0] != b.shape[0]:
            # atleast_2d can pick the wrong orientation for a (n,) 1-D weight
            # vector (e.g. a 1-unit layer's LW stored flat) -- (1, n) is what
            # `tansig(W @ x + b)` needs when b has 1 element.
            W = W.reshape(1, -1) if b.shape[0] == 1 else W.T
        layers.append((W, b))
        layer_index += 1

    if not layers:
        raise ValueError(f"{module_name!r}: no layers found (no 'b1')")

    return {
        "x1_xoffset": np.ravel(np.asarray(x1["xoffset"], dtype=np.float64)),
        "x1_gain": np.ravel(np.asarray(x1["gain"], dtype=np.float64)),
        "x1_ymin": float(x1["ymin"]),
        "y1_xoffset": float(y1["xoffset"]),
        "y1_gain": float(y1["gain"]),
        "y1_ymin": float(y1["ymin"]),
        "layers": layers,
    }


def architecture_signature(weights: dict) -> tuple:
    """A hashable key identifying a net's *shape* (layer count + each layer's
    (n_out, n_in)) -- two nets with the same signature can be evaluated together
    in one batched call (see ``run_nets.batched_forward``), regardless of their
    actual weight values or which variable/region/ensemble-member they come from.
    """
    return tuple(W.shape for W, _b in weights["layers"])
