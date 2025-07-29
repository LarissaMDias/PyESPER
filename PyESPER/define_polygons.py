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
    SAtlInds, SoAfrInds = [], []
    for i, z in zip(longitude, latitude):
        # Check if the conditions are met for Southern Atlantic
        if (-34 > z > -44):  # Check latitude first to reduce unnecessary checks
            if i > 290 or i < 20:
                SAtlInds.append('True')   
            else:
                SAtlInds.append('False')

            # Check if the condition is met for Southern Africa
            if 19 < i < 27:
                SoAfrInds.append('True')
            else:
                SoAfrInds.append('False')
        else:
            SAtlInds.append('False')
            SoAfrInds.append('False')

    # Create Dictionary with boolean indicators
    df = {'AAInds': AAIndsM, 'BeringInds': beringconditions, 'SAtlInds': SAtlInds, \
        'SoAfrInds': SoAfrInds, 'Lat': latitude, 'Lon': longitude, 'Depth': depth}
    
    return df

