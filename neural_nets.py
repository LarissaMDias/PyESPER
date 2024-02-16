def neuralnets(DesiredVariables, Equations, code = {}):

    """ 
    The following script uses Python translations of previously-defined 
neural network outputs to provide neural network estimates that \
        are nearly identical to those from MATLAB ESPER neural networks. 
Requires user to have downloaded Python scripts for the \
        variable-equation case requested 

    Inputs:
        DesiredVariables (1 by v list): Required list of str names 
representing desired output (variables to be \
            returned) in umol/kg, with naming/abbreviations standardized 
to TA, DIC, pH, phosphate, nitrate, silicate, and \
            oxygen
        Equations (list): Input equations (user-generated as kwarg in 
ESPER function call, or all 16 equations \
            (default))
        code (list): List of pd DataFrames of input parameters (adjusted 
to potential temperature and AOU, and to \
            molality if PerKgSwTF is False) fit into the correct position 
for each desired equation case and \
            DesiredVariable combination, along with the remaining 
variables from InputAll; dictionary of pandas \
            DataFrames
    Outputs:
        Esta, Esto (dictionaries): Dictionaries of lists of estimates for 
each of the user-requested equation case combinations
    """            

    import importlib
    from statistics import mean
    import numpy as np
    
    combos = list(code.keys())
    combovalues = list(code.values())
    EstAtl, EstOther = {}, {}
    P, Sd, Td, Ad, Bd, Cd = {}, {}, {}, {}, {}, {} 

    for c in range(0, len(combos)):
        cosd = list(np.cos(np.deg2rad(combovalues[c]['Longitude'] - 20)))
        sind = list(np.sin(np.deg2rad(combovalues[c]['Longitude'] - 20)))
        lat, depth, Sl, Tl, Al, Bl, Cl = list(combovalues[c]['Latitude']), 
list(combovalues[c]['Depth']), list(combovalues[c]['S']), \
            list(combovalues[c]['T']), list(combovalues[c]['A']), 
list(combovalues[c]['B']), list(combovalues[c]['C'])
        Sl, Tl, Al, Bl, Cl = (list(combovalues[c][key]) for key in ['S', 
'T', 'A', 'B', 'C'])
        Sd[c+1], Td[c+1], Ad[c+1], Bd[c+1], Cd[c+1] = Sl, Tl, Al, Bl, Cl

    for v in DesiredVariables:
        for e in Equations:
            name = v + str(e)
            params = [cosd, sind, lat, depth, Sd[e], Td[e], Ad[e], Bd[e], 
Cd[e]][:16-e]
            P[name] = [[[params]]]

            netstimateAtl, netstimateOther = [], []
            for n in range(1, 5):
                fOName = f'ESPER_{v}_{e}_Other_{n}'
                fAName = f'ESPER_{v}_{e}_Atl_{n}'
                moda = importlib.import_module(fAName)
                modo = importlib.import_module(fOName)
                   
                netstimateAtl.append(moda.PyESPER_NN(P[name]))
                netstimateOther.append(modo.PyESPER_NN(P[name]))
            
            EstAtlL = [[netstimateAtl[na][0][eatl] for na in range(4)] for 
eatl in range(len(netstimateAtl[0][0]))]
            EstOtherL = [[netstimateOther[no][0][eoth] for no in range(4)] 
for eoth in range(len(netstimateOther[0][0]))]

            EstAtl[name] = EstAtlL
            EstOther[name] = EstOtherL
    
    def compute_means(data_dict):
        result = {}
        for key, value in data_dict.items():
            result[key] = [mean(inputs) for inputs in zip(*value)]
        return result
    Esta = compute_means(EstAtl)
    Esto = compute_means(EstOther)

    return Esta, Esto
