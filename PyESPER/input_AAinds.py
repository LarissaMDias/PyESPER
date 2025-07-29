def input_AAinds(C={}, code={}):

    """
    Separates user-defined inpus into Atlantic and Arctic regions or other
        regions, defined as in ESPERv1 for MATLAB.

    Inputs:
        C: Dictionary of pre-adjusted grid coordinates
        code: Dictionary of iterated equation-case scenario inputs for
            user-requested variable-equation cases

    Outputs:
        AAdata: Dictionary of code data separated for areas encompassed by the
            Atlantic and Arctic Oceans only
        Elsedata: Dictionary of code data separated for areas not encompassed by
            the Atlantic and Arctic Oceans
    """

    import numpy as np
    import matplotlib.path as mpltPath

    # Define Polygons that encompass the Atlantic and Arctic Oceans, geographically
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
    AAIndsM = AAIndsM.astype(int)

    # Create dictionary
    df = {
        'AAInds': AAIndsM,
        'Lat': latitude,
        'Lon': longitude,
        'Depth': depth
    }
                        
    for df_code in code.values():
        df_code['AAInds'] = df['AAInds']
    
    # Initialize dictionaries for AA (Atlantic Arctic) and Else (not Atlantic Arctic) data
    AAdata = {}   
    Elsedata = {}

    # Iterate over each key in code
    for i in code:
        # Extract data arrays from the DataFrame
        data_arrays = np.array([code[i][key] for key in ['Depth', 'Latitude', 'Longitude', 'S', 'T', 'A', 'B', 'C',
                                                                'Order', 'Salinity_u', 'Temperature_u', 'Phosphate_u',
                                                                'Nitrate_u', 'Silicate_u', 'Oxygen_u', 'AAInds']])
    
        # Unpack arrays into separate variables
        depth, latitude, longitude, S, T, A, B, C, order, sal_u, temp_u, phos_u, nitr_u, sil_u, oxyg_u, aainds = [
            np.array(arr, dtype=float) for arr in data_arrays
        ]
    
        # Scale the depth values
        depth = depth / 25
        
        # Reshape data arrays to match the number of rows
        NumRows_out = len(longitude)
        reshaped_data = [arr.reshape(NumRows_out, 1) for arr in [depth, latitude, longitude, S, T, A, B, C, order, sal_u,
                                                                 temp_u, phos_u, nitr_u, sil_u, oxyg_u, aainds]]
        dep, lat, lon, sal, temp, avar, bvar, cvar, orde, salu, tempu, phosu, nitru, silu, oxygu, aai = reshaped_data
    
        # Combine the data into one array for further splitting
        InputBool = np.hstack(reshaped_data)
    
        # Define columns for the final dictionary
        columns = ['d2d', 'Latitude', 'Longitude', 'S', 'T', 'A', 'B', 'C', 'Order', 'Salinity_u', 'Temperature_u',
                   'Phosphate_u', 'Nitrate_u', 'Silicate_u', 'Oxygen_u', 'AAInds']
        NumCols_out = len(columns)
        
        # Function to filter arrays based on condition
        def split(arr, cond):
            return arr[cond]
        
        # Split the data into AA and Else data
        InputAA_01 = split(InputBool, InputBool[:, -1] == 1) 
        InputElse_01 = split(InputBool, InputBool[:, -1] == 0)

        # Reshape and create dictionaries for AA and Else
        AAInput = {
            col: InputAA_01.reshape(len(InputAA_01), NumCols_out)[:, i]
            for i, col in enumerate(columns)
        }
        ElseInput = {
            col: InputElse_01.reshape(len(InputElse_01), NumCols_out)[:, i]
            for i, col in enumerate(columns)
        }
        
        # Store the results in the dictionaries
        AAdata[i] = AAInput
        Elsedata[i] = ElseInput
        
    return AAdata, Elsedata

