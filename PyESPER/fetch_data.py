def fetch_data(DesiredVariables, Path):
    """
    Gathers the necessary LIR files that were pre-trained in MATLAB ESPERs

    Inputs:
        DesiredVariables: List of desired output estimate variables
        Path: User-defined computer path of locations of files

    Outputs:
        LIR_data: List of dictionaries of LIR data

    The four ``Mat_fullgrid/*.mat`` files per variable (~82 MB of coefficients each) are
    loaded once per process and memoised by
    :func:`PyESPER.kernels.grid_cache.variable_grids`; this function only assembles the
    per-variable results into the historical list-of-dicts return shape. Previously every
    call re-read them from disk, which under ``xr_methods`` meant once per dask chunk.

    Note the ``UncGrid`` element: it is a single grid, not a per-variable dict, so for a
    multi-variable request it is the *last* variable's -- preserved here exactly as the
    original behaved. It is only consumed by ``emlr_estimate``, which calls this one
    variable at a time, so the ambiguity has never been reachable in practice.
    """
    from PyESPER.kernels.grid_cache import variable_grids

    AAIndsCs, GridCoords, Cs = {}, {}, {}
    UncGrid = None

    for v in DesiredVariables:
        entry = variable_grids(v, Path)
        GridCoords[v] = entry["grid_coords"]
        AAIndsCs[v] = entry["aainds"]
        Cs[v] = entry["cs"]
        UncGrid = entry["uncgrid"]

    return [GridCoords, Cs, AAIndsCs, UncGrid]
