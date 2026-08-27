def define_polygons(C={}):

    """
    Defining and structuring indexing within ocean region polygons.
    First defines the polygons, then assesses the location of user-provided
    coordinates within polygons.

    Inputs:
        C: Dictionary of adjusted coordinates

    Output:
        df: Dictionary of adjusted coordinates with boolean indicators for specific
            ocean regions
    """

    import numpy as np
    import matplotlib.path as mpltPath

    # Define polygons for Atlantic and Arctic (AA) or other (Else) ocean basins
    LNAPoly = np.array([[300, 0], [260, 20], [240, 67], [260, 40], [361, 40], [361, 0], [298, 0]])
    LSAPoly = np.array([[298, 0], [292, -40.01], [361, -40.01], [361, 0], [298, 0]])
    LNAPolyExtra = np.array([[-1, 50], [40, 50], [40, 0], [-1, 0], [-1, 50]])
    LSAPolyExtra = np.array([[-1, 0], [20, 0], [20, -40], [-1, -40], [-1, 0]])
    LNOPoly = np.array([[361, 40], [361, 91], [-1, 91], [-1, 50], [40, 50], [40, 40], [104, 40], [104, 67], [240, 67],
                    [280, 40], [361, 40]])
    xtra = np.array([[0.5, -39.9], [.99, -39.9], [0.99, -40.001], [0.5, -40.001]])
    polygons = [LNAPoly, LSAPoly, LNAPolyExtra, LSAPolyExtra, LNOPoly, xtra]

    # Create Paths
    paths = [mpltPath.Path(poly) for poly in polygons]

    # Extract coordinates
    longitude, latitude, depth = np.array(C["longitude"]), np.array(C["latitude"]), np.array(C["depth"])

    # Check if coordinates are within each polygon
    conditions = [path.contains_points(np.column_stack((longitude, latitude))) for path in paths]

    # Combine conditions
    AAIndsM = np.logical_or.reduce(conditions)

    # Adding Bering Sea, S. Atl., and S. African Polygons separately
    Bering = np.array([[173, 70], [210, 70], [210, 62.5], [173, 62.5], [173, 70]])
    beringpath = mpltPath.Path(Bering)
    beringconditions = beringpath.contains_points(np.column_stack((longitude, latitude)))

    # Vectorised replacement for the original per-point Python loop, which built two
    # n_points-long lists of the *strings* 'True'/'False' and left the caller to compare
    # them back to booleans. At production point counts that loop, and the numpy string
    # array it forced ``process_netresults`` to materialise, cost more than the neural
    # nets themselves. The inequalities below are transcribed strictly (all four bounds
    # exclusive, matching the original ``-34 > z > -44`` / ``i > 290 or i < 20`` /
    # ``19 < i < 27``), and the result is now a plain boolean array.
    in_south_band = (latitude > -44) & (latitude < -34)
    SAtlInds = in_south_band & ((longitude > 290) | (longitude < 20))
    SoAfrInds = in_south_band & (longitude > 19) & (longitude < 27)

    # Create Dictionary with boolean indicators
    df = {'AAInds': AAIndsM, 'BeringInds': beringconditions, 'SAtlInds': SAtlInds, \
        'SoAfrInds': SoAfrInds, 'Lat': latitude, 'Lon': longitude, 'Depth': depth}
    
    return df

