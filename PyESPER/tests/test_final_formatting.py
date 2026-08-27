"""Regression test for the ``("pH" or "DIC") in DesiredVariables`` bug.

``final_formatting`` decides whether to keep the anthropogenic-carbon-adjusted
estimates (``Cant_adjusted``, built by ``pH_DIC_nn_adjustment``) or fall back
to the raw, unadjusted ones (``Est_pre``). The original condition,
``("pH" or "DIC") in DesiredVariables``, evaluates the parenthesized ``or``
first: since the string ``"pH"`` is truthy, Python short-circuits and the
whole expression collapses to the literal ``"pH" in DesiredVariables`` --
silently dropping the ``"DIC"`` check. A caller requesting
``DesiredVariables=["DIC"]`` alone (no ``"pH"``) would therefore always hit
the ``else`` branch: the properly Canth-adjusted DIC estimate computed by
``pH_DIC_nn_adjustment`` was discarded in favor of the unadjusted one, even
though ``pH_DIC_nn_adjustment`` itself printed "Estimating anthropogenic
carbon..." and did the work -- the classic "estimated but not used" bug.
"""

from PyESPER.final_formatting import final_formatting


def test_dic_only_request_keeps_cant_adjusted_estimates():
    """A DesiredVariables list containing "DIC" but not "pH" must still take
    the Cant_adjusted branch -- this is the exact case that was broken."""
    cant_adjusted = {"DIC8": [2010.0, 2011.0]}
    est_pre = {"DIC8": [2000.0, 2001.0]}

    result = final_formatting(["DIC"], cant_adjusted, est_pre)

    assert result == cant_adjusted, (
        "DesiredVariables=['DIC'] must use the anthropogenic-carbon-adjusted "
        "estimates, not silently fall back to the unadjusted ones."
    )


def test_ph_only_request_keeps_cant_adjusted_estimates():
    cant_adjusted = {"pH16": [8.05, 8.06]}
    est_pre = {"pH16": [8.00, 8.01]}

    result = final_formatting(["pH"], cant_adjusted, est_pre)

    assert result == cant_adjusted


def test_neither_dic_nor_ph_falls_back_to_unadjusted():
    """A request for e.g. TA/oxygen/nutrients only has no anthropogenic-carbon
    correction to apply, so it must use the raw estimates."""
    cant_adjusted = {"TA8": [2300.0, 2301.0]}
    est_pre = {"TA8": [2300.0, 2301.0]}

    result = final_formatting(["TA"], cant_adjusted, est_pre)

    assert result == est_pre
