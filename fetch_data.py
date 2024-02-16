def fetch_data(DesiredVariables): 
    """
    A function to retrieve data from the online repository and access 
within python as pandas dataframes. Fetches all \
        16 equation cases for each desired variable. If data has already 
been saved to the computer, this will bypass \
        the download and simply access the data file. Needs adjustment 
based on user file path. 
    
    Input: 
        DesiredVariables (1 by v list): Required list of str names 
representing desired output (variables to be \
            returned) in umol/kg, with naming/abbreviations standardized 
to TA, DIC, pH, phosphate, nitrate, silicate, and \
            oxygen    
            
    Outputs:
        LIR_data (list): Contains the following:
            GridCoords: 106400 x 4 pandas DataFrame with columns as 
longitude, latitude, depth, and density
            Cs##: 106400 x 6 pandas DataFrames whereby ## is the 2-digit 
equation number (1-16), and columns are \
                coefficients. 
            AAIndsCs: 50225 x 1 boolean pandas DataFrame indicating 1 for 
True when the Cs are located within the \
                Atlantic or Arctic Ocean and 0 for false when the Cs are 
not located within the Atlantic or Arctic Ocean
    """
    import pandas as pd
    from scipy.io import loadmat
    
    AAIndsCs = {}
    GridCoords = {}
    Cs = {}
    
    for v in DesiredVariables:
        fname = f"/Users/larissadias/Documents/NN_files_{v}_Unc_Poly.mat" # You will need to have downloaded this file; 
            # change this directory address as needed
        name = f'NN_files_{v}_Unc_Poly'
        NNs = loadmat(fname)
        Polys, UncGrid = NNs['Polys'][0][0], NNs['UncGrid'][0][0]
        
    NN_data = [Polys, UncGrid]
    
    return NN_data
