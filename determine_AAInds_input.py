def determine_AAInds_input(code={}):
    
    """
    A function which returns a numpy array for boolean (T/F) values of 
whether desired OutputCoordinates (user-generated) 
        are within the Atlantic and Arctic Oceans or not. 
    
    Inputs:
        code (list): List of pd DataFrames of input parameters (adjusted 
to potential temperature and AOU, and to \
            molality if PerKgSwTF is False) fit into the correct position 
for each desired equation case and \
            DesiredVariable combination, along with the remaining 
variables from InputAll; dictionary of pandas \
            DataFrames
    
    Parameters:
       Polys: Polygons encompassing the Atlantic and Arctic regions (I 
added Caribbean to the pre-existing polygons from 
           ESPER MATLAB version)
            
    Output: 
        df (pd DataFrame): consists of OutputCoordinates and AAIndsM (n by 
1 numpy array), an array of boolean (True \
            or False) values, whereby "True" reveals that desired output 
coordinates are within the Atlantic or Arctic \
            and "False" means not within the Atlantic or Arctic Oceans
    """
    import numpy as np
    import pandas as pd
    import matplotlib.path as mpltPath

    # Define Polygons
    LNAPoly = np.array([[300, 0], [260, 20], [240, 67], [260, 40], [361, 
40], [361, 0], [298, 0]])
    LSAPoly = np.array([[298, 0], [292, -40.01], [361, -40.01], [361, 0], 
[298, 0]])
    LNAPolyExtra = np.array([[-1, 50], [40, 50], [40, 0], [-1, 0], [-1, 
50]])
    LSAPolyExtra = np.array([[-1, 0], [20, 0], [20, -40], [-1, -40], [-1, 
0]])
    LNOPoly = np.array([[361, 40], [361, 91], [-1, 91], [-1, 50], [40, 
50], [40, 40], [104, 40], [104, 67], [240, 67],
                        [280, 40], [361, 40]])
    xtra = np.array([[0.5, -39.9], [.99, -39.9], [0.99, -40.001], [0.5, 
-40.001]])

    polygons = [LNAPoly, LSAPoly, LNAPolyExtra, LSAPolyExtra, LNOPoly, 
xtra]

    # Create Paths
    paths = [mpltPath.Path(poly) for poly in polygons]

    # Extract coordinates
    latitude, longitude, depth = np.array(code['Latitude']), 
np.array(code['Longitude']), np.array(code['Depth'])

    # Check if coordinates are within each polygon
    conditions = [path.contains_points(np.column_stack((longitude, 
latitude))) for path in paths]

    # Combine conditions
    AAIndsM = np.logical_or.reduce(conditions)

    # Adding Bering Sea, S. Atl., and S. African Polygons separately 
    Bering = np.array([[173, 70], [210, 70], [210, 62.5], [173, 62.5], 
[173, 70]])
    beringpath = mpltPath.Path(Bering)
    beringconditions = 
beringpath.contains_points(np.column_stack((longitude, latitude)))
    SAtlInds, SoAfrInds = [], []
    for i, z in zip(longitude, latitude):
        if i > 290 and -34 > z > -44:
            SAtlInds.append('True')
        elif i < 20 and -34 > z > -44:
            SAtlInds.append('True')
        else: 
            SAtlInds.append('False')
        if 19 < i < 27 and -34 > z > -44:
            SoAfrInds.append('True')
        else:
            SoAfrInds.append('False')
      
    # Create DataFrame
    df = pd.DataFrame({'AAInds': AAIndsM, 'BeringInds': beringconditions, 
'SAtlInds': SAtlInds, \
                       'SoAfrInds': SoAfrInds, 'Lat': latitude, 'Lon': 
longitude, 'Depth': depth})

    return df
