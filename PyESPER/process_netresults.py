def process_netresults(Equations, code={}, df={}, EstAtl={}, EstOther={}):

    """
    Regional smoothing and processing net outputs

    Inputs:
        Equations: List of equations to use
        code: Dictionary of input for each equation-variable scenario
        df: Dictionary of coordinates and boolean indicators for regions
        EstAtl: Dictionary of estimates for the Atlantic and Arctic
            for each combination
        EstOther: Dictionary of estimates for not Atlantic and Arctic,
            for each combination

    Output:
        Estimate: Dictionary of estimates for each combination

    Notes
    -----
    This is a vectorized rewrite of the original per-point Python loops. Every
    ``np.tile(...)`` in the original smoothing loops tiled a *scalar* to shape
    ``(1, len(Equations))`` and then immediately re-extracted index ``[0][0]`` --
    i.e. the tile width never affected the result, so the tiling was pure
    overhead. Replacing the per-point loops with whole-array numpy expressions
    is therefore numerically identical to the original, just without the
    O(n_points) Python-level `np.tile`/`np.mean` call overhead that dominated
    runtime at large n_points.
    """

    import numpy as np

    def _as_bool(values):
        """Boolean view of a region indicator, accepting the legacy string form."""
        arr = np.asarray(values)
        if arr.dtype.kind in "US":
            return arr == "True"
        return arr.astype(bool)

    # Function to provide the mean of values when needed. `estimates[key]` is
    # an (n_points, 4) array (stacked ensemble members from run_nets); a single
    # vectorized mean over axis=1 replaces the previous per-point `np.mean()`
    # call in a Python loop.
    def process_estimates(estimates):
        return {
            key: np.asarray(value, dtype=float).mean(axis=1)
            for key, value in estimates.items()
        }

    Esta = process_estimates(EstAtl)
    Esto = process_estimates(EstOther)

    # Processing regionally in the Atlantic and Bering
    EstA, EstB, EB2, ESat, ESat2, ESaf, Estimate  = {}, {}, {}, {}, {}, {}, {}

    for i in code:
        code[i]["AAInds"] = df["AAInds"]
        code[i]["BeringInds"] = df["BeringInds"]
        code[i]["SAtlInds"] = df["SAtlInds"]
        code[i]["SoAfrInds"] = df["SoAfrInds"]

    for codename, codedata in code.items():
        aainds, beringinds, satlinds, latitude, safrinds = (
            codedata[key] for key in ["AAInds", "BeringInds", "SAtlInds", "Latitude", "SoAfrInds"]
        )
        latitude = np.asarray(latitude, dtype=float)
        aainds = np.asarray(aainds, dtype=bool)
        beringinds = np.asarray(beringinds, dtype=bool)
        # define_polygons now returns real boolean arrays. Older versions returned lists
        # of the strings 'True'/'False'; accept both so a caller passing a hand-built
        # `df` (or an older define_polygons) keeps working.
        satlinds_bool = _as_bool(satlinds)
        safrinds_bool = _as_bool(safrinds)

        esta = Esta[codename]
        esto = Esto[codename]

        Estatl = np.where(aainds, esta, esto)

        # Smoothing for the Atlantic
        Estb = esta * ((latitude - 62.5) / 7.5) + esto * ((70.0 - latitude) / 7.5)

        eb2 = np.where(beringinds, Estb, Estatl)

        # Smoothing for the Bering
        Estsat = esta * ((latitude + 44.0) / 10.0) + esto * ((-34.0 - latitude) / 10.0)

        EstA[codename], EstB[codename], EB2[codename], ESat[codename] = Estatl, Estb, eb2, Estsat

        # Regional processing for S. Atlantic
        ESat2[codename] = np.where(satlinds_bool, ESat[codename], EB2[codename])

        # Regional processing for S. Africa
        lon = np.asarray(df["Lon"], dtype=float)
        esafr = ESat2[codename] * ((27.0 - lon) / 8.0) + esto * ((lon - 19.0) / 8.0)
        ESaf[codename] = esafr

        Estimate[codename] = np.where(safrinds_bool, ESaf[codename], ESat2[codename])

    # Values stay as float64 ndarrays. They used to be converted back to Python lists
    # here to preserve an older return-type contract, but every consumer in this package
    # immediately calls np.array/np.asarray on them again (pH_DIC_nn_adjustment,
    # final_formatting, mixed, xr_methods), so the round-trip only cost memory and time
    # -- at 200k points it was a measurable fraction of the whole call, against a neural
    # net evaluation that now takes well under a tenth of a second.
    return Estimate
