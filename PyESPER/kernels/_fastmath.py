"""A branch-free, division-free ``tansig`` that numba can actually vectorise.

``tansig(x) = 2 / (1 + exp(-2x)) - 1`` (mathematically ``tanh``) is applied to every
hidden unit of every ESPER net -- roughly 1,900 evaluations per estimated point -- and
was measured to dominate the neural-net path's runtime. Getting it fast is most of the
kernel's speedup, and it is not as simple as "call ``np.exp``".

Three separate things stop the obvious implementations from vectorising, all measured on
the target machine (2 x EPYC 7742, AVX2, numba 0.66, **no SVML installed**):

1. ``np.exp``/``np.tanh`` compile to a scalar ``call`` into libm. Without SVML numba has
   no vector version to dispatch to, so the whole loop scalarises. (~11.7 ns/element.)
2. **Division scalarises the loop too**, even under ``fastmath``: writing
   ``2.0 / (1.0 + e) - 1.0`` emits ``divsd``, and one scalar op is enough to sink the
   surrounding vectorisation. The reciprocal here is therefore computed with a
   magic-constant seed plus Newton-Raphson refinement -- multiplies only.
3. **The float-to-integer conversion needed for ``2**k`` scalarises it as well.**
   ``vcvttpd2qq`` (packed f64 -> i64) is AVX-512; on AVX2 there is no such instruction,
   so ``np.int64(fk)`` forces LLVM back to scalar code. The exponent bits are instead
   extracted by adding a magic constant that parks the integer in the low mantissa bits,
   which needs only packed float add and packed integer shift (``vpaddq``/``vpsllq``).

There is one more trap worth spelling out, because it fails *silently*: the usual way to
write the round-to-integer step is ``y = t*LOG2E + MAGIC; fk = y - MAGIC``. Under
``fastmath`` LLVM reassociates that straight back to ``fk = t*LOG2E``, the rounding never
happens, and the result is quietly wrong (measured max error 0.17 -- not a rounding-level
discrepancy, a broken function). ``np.floor`` is used for the rounding instead, which
lowers to a packed ``vroundpd`` on AVX2 and cannot be reassociated away; the magic
constant is used *only* to extract the bits.

Accuracy, measured against ``2/(1+np.exp(-2x))-1`` over ``x`` in [-25, 25] (4M points)
plus the interesting values (0, +-19.06, +-20, tiny, saturated):

* max absolute error   7.8e-16
* max relative error   2.9e-13 (over ``|reference| > 1e-3``)

which is comfortably inside the 1e-9 agreed tolerance, and small enough that it stays
invisible after the output layer's ``mapminmax`` rescaling.
"""

from __future__ import annotations

import llvmlite.ir as llir
import numpy as np
from numba import njit, types
from numba.extending import intrinsic


@intrinsic
def i64_as_f64(typingctx, x):
    """Reinterpret an int64's bits as a float64. Lowers to an LLVM ``bitcast``.

    A bitcast costs nothing (the value is already in a register) and, unlike a
    *conversion*, is legal to vectorise on AVX2. ``np.int64(...).view(...)`` does not
    work on scalars in nopython mode, which is why this exists.
    """
    if x != types.int64:
        return None

    def codegen(context, builder, sig, args):
        return builder.bitcast(args[0], llir.DoubleType())

    return types.float64(types.int64), codegen


@intrinsic
def f64_as_i64(typingctx, x):
    """Reinterpret a float64's bits as an int64. Lowers to an LLVM ``bitcast``."""
    if x != types.float64:
        return None

    def codegen(context, builder, sig, args):
        return builder.bitcast(args[0], llir.IntType(64))

    return types.int64(types.float64), codegen


_LOG2E = 1.4426950408889634
_LN2HI = 6.93147180369123816490e-01  # Cody-Waite split of ln 2, so that
_LN2LO = 1.90821492927058770002e-10  # fk*LN2HI is exact for the |k| we allow
_MAGIC = 6755399441055744.0  # 1.5 * 2**52
_RCP_SEED = np.int64(0x7FDE623822FC16E6)  # ~5%-accurate 1/x seed by bit subtraction

# exp(r) Taylor coefficients, |r| <= ln2/2 = 0.347; degree 12 leaves a truncation term
# r**13/13! <= 2.2e-16, i.e. below one ulp.
_C2 = 1.0 / 2.0
_C3 = 1.0 / 6.0
_C4 = 1.0 / 24.0
_C5 = 1.0 / 120.0
_C6 = 1.0 / 720.0
_C7 = 1.0 / 5040.0
_C8 = 1.0 / 40320.0
_C9 = 1.0 / 362880.0
_C10 = 1.0 / 3628800.0
_C11 = 1.0 / 39916800.0
_C12 = 1.0 / 479001600.0
_C13 = 1.0 / 6227020800.0

# tanh saturates to +-1 in float64 well before |x| = 19.07. Clamping is not cosmetic:
# it bounds k to [-58, 58], which is what makes the (k + 1023) << 52 exponent
# construction below safe from overflow. Do not remove it without re-deriving that.
_CLAMP = 20.0


@njit(inline="always", fastmath=True, cache=True)
def tansig(x):
    """MATLAB's ``tansig`` transfer function. See the module docstring."""
    t = -2.0 * min(max(x, -_CLAMP), _CLAMP)

    # Range reduction: t = fk*ln2 + r with |r| <= ln2/2.
    # np.floor -> packed vroundpd. Recovering fk as (t*LOG2E + MAGIC) - MAGIC would be
    # reassociated back to t*LOG2E by fastmath and silently break; see module docstring.
    fk = np.floor(t * _LOG2E + 0.5)
    r = (t - fk * _LN2HI) - fk * _LN2LO

    # Estrin evaluation: a 4-deep dependency chain instead of Horner's 12-deep one.
    r2 = r * r
    r4 = r2 * r2
    r8 = r4 * r4
    pz = (
        ((1.0 + r) + r2 * (_C2 + r * _C3))
        + r4 * ((_C4 + r * _C5) + r2 * (_C6 + r * _C7))
        + r8 * (((_C8 + r * _C9) + r2 * (_C10 + r * _C11)) + r4 * (_C12 + r * _C13))
    )

    # exp(t) = pz * 2**fk. The magic add parks fk in the low mantissa bits; << 52
    # discards everything above bit 11, so the high bits of the sum do not matter.
    e = pz * i64_as_f64((f64_as_i64(fk + _MAGIC) + 1023) << 52)

    # 1/(1+e) by seed + 4 Newton steps (5% -> 2.5e-3 -> 6.4e-6 -> 4e-11 -> ~1e-21).
    # Four steps are needed: three leave ~4e-11, far coarser than float64.
    d = 1.0 + e
    q = i64_as_f64(_RCP_SEED - f64_as_i64(d))
    q = q * (2.0 - d * q)
    q = q * (2.0 - d * q)
    q = q * (2.0 - d * q)
    q = q * (2.0 - d * q)

    # Kept in the reference's own algebraic form (2/(1+e) - 1 rather than (1-e)/(1+e))
    # so the cancellation behaviour near x = 0 matches what the ESPER nets were
    # validated against.
    return 2.0 * q - 1.0
