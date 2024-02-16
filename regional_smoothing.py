def regional_smoothing(Equations, EstAtl={}, EstOther={}, code={}):

    """
    IN PROGRESS. The following code creates a single estimate, adjusted 
regionally. Needs work to replicate some of the smoothing from \
        MATLAB. 

    Inputs:
        Equations (list): Input equations (user-generated as kwarg in 
ESPER function call, or all 16 equations \
            (default))
        Esta, Esto (dictionaries): Dictionaries of lists of estimates for 
each of the user-requested equation case combinations
        code (list): List of pd DataFrames of input parameters (adjusted 
to potential temperature and AOU, and to \
            molality if PerKgSwTF is False) fit into the correct position 
for each desired equation case and \
            DesiredVariable combination, along with the remaining 
variables from InputAll; dictionary of pandas \
            DataFrames
    Outputs:
        Est (dictionary): Dict of estimates for each user-requested 
desired variable-equation case scenario.
    """
    #import numpy as np
    
    def regional_smoothing(Equations, EstAtl={}, EstOther={}, code={}):
    Est = {}
    for name, values in code.items():
        EstL = []
        latitude = values['Latitude']

        for i, aa_ind in enumerate(values['AAInds']):
            if aa_ind:
                EstL.append(EstAtl[name][i])
            else:
                EstL.append(EstOther[name][i])
                        
        for l, bering_ind in enumerate(values['BeringInds']):
            if bering_ind:
                repeated_values = (latitude[l]-62.5)/7.5
                B1 = EstAtl[name][l] * np.tile(repeated_values, 
len(Equations))
                repeated_values2 = (70-latitude[l])/7.5
                B2 = EstOther[name][l] * np.tile(repeated_values2, 
len(Equations))
                EstL[l] = B1[0][0] + B2[0][0]

        Est[name] = EstL
            
    return Est
