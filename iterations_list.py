def iterations_list(DesiredVariables, Equations, PerKgSwTF, InputAll={}):
    """
    Creates a list of pandas DataFrames filled with user input values and 
the correct equations for specified \
        variables. Limits to only the measurements that are needed, 
according to user input. Also performs any \
        necessary conversions on predictor measurements.
    
    Inputs:
        DesiredVariables (list): Same as input for PyESPER call, 
indicating desired output variables
        Equations (list): Input equations (user-generated as kwarg in 
ESPER function call, or all 16 equations \
            (default))
        PerKgSwTF (bool): Indicates whether user requested conversion from 
molar (umol/L) to molal (umol/kg) units \
            (False) or not (units provided are molality; default/True)
        InputAll (dict): Includes OutputCoordinates (with processing of 
longitude already complete), \
            PredictorMeasurements (unprocessed), MeasurementUncertainties 
(processed), Dates (processed or default), \
            and Order of input (for later sorting)
                   
    Outputs:
        code (list): List of pd DataFrames of input parameters (adjusted 
to potential temperature and AOU, and to \
            molality if PerKgSwTF is False) fit into the correct position 
for each desired equation case and \
            DesiredVariable combination, along with the remaining 
variables from InputAll; dictionary of pandas \
            DataFrames
        need (dict): Order indicating which input parameter 
(PredictorMeasurement) is associated with each equation- \
            DesiredVariable combination the user requested
    """
    import numpy as np
    import pandas as pd
    import seawater as sw
    
    np.set_printoptions(precision=15)
    depth = np.array(InputAll['Depth'])
    latitude = np.array(InputAll['Latitude'])
    salinity = np.array(InputAll['Salinity'])
    n = len(salinity)
    if 'Temperature' in InputAll:
        temp= np.array(InputAll['Temperature'])
    else: 
        temp = np.tile(10, n)
    temp2 = sw.ptmp(salinity, temp, sw.pres(depth, latitude), pr=0)
    temperature =[]
    for t in temp2:
        if t == 5:
            t = 5.000000001
        temperature.append('{:.12g}'.format(t))    
    if "Oxygen" in InputAll:
        oxyg = np.array(InputAll['Oxygen'])
        oxygen = sw.satO2(salinity, temp2)*44.6596 - (oxyg)
    else: 
        oxygen = np.tile('nan', n) 
    for i in range(0, len(oxygen)):
        if oxygen[i]<0.0001:
            if oxygen[i]>-0.0001:
                oxygen[i] = 0
    if "Phosphate" in InputAll:
        phosphate = np.array(InputAll['Phosphate'])
    else:
        phosphate = np.tile('nan', n)
    if "Nitrate" in InputAll:
        nitrate = np.array(InputAll['Nitrate'])
    else:
        nitrate = np.tile('nan', n)
    if "Silicate" in InputAll:
        silicate = np.array(InputAll['Silicate'])
    else: 
        silicate = np.tile('nan', n)
    
    if PerKgSwTF == False:
        densities = sw.dens(salinity, temperature, sw.pres(depth, 
latitude))/1000
        if 'Phosphate' in InputAll:
            phosphate = phosphate/densities
        if 'Nitrate' in InputAll:
            nitrate = nitrate/densities
        if 'Silicate' in InputAll:
            silicate = silicate/densities
            
    EqsString = [str(e) for e in Equations]
    
    NeededForProperty = pd.DataFrame({
            "TA": [1, 2, 4, 6, 5], 
            "DIC": [1, 2, 4, 6, 5], 
            "pH": [1, 2, 4, 6, 5],  
            "phosphate": [1, 2, 4, 6, 5], 
            "nitrate": [1, 2, 3, 6, 5], 
            "silicate": [1, 2, 3, 6, 4], 
            "oxygen": [1, 2, 3, 4, 5]
            })
            
    VarVec = pd.DataFrame({
            '1': [1, 1, 1, 1, 1],
            '2': [1, 1, 1, 0, 1],
            '3': [1, 1, 0, 1, 1],
            '4': [1, 1, 0, 0, 1],
            '5': [1, 1, 1, 1, 0],
            '6': [1, 1, 1, 0, 0],
            '7': [1, 1, 0, 1, 0],
            '8': [1, 1, 0, 0, 0],
            '9': [1, 0, 1, 1, 1],
            '10': [1, 0, 1, 0, 1],
            '11': [1, 0, 0, 1, 1],
            '12': [1, 0, 0, 0, 1],
            '13': [1, 0, 1, 1, 0],
            '14': [1, 0, 1, 0, 0],
            '15': [1, 0, 0, 1, 0],
            '16': [1, 0, 0, 0, 0],
        })
        
    prod, prodz, Name = [], [], []
    need, code = {}, {}
    
    for d in DesiredVariables:
        dv = NeededForProperty[d]
        for e in EqsString:
            eq=VarVec[e]
            name = d + e
            Name.append(name)
            prod.append(eq * dv)
            prodN = np.array(eq * dv)
            prodStr = []
                        
            nan = np.where(prodN==0, 'nan', prodN)
            sal = np.where(nan=='1', 'salinity', nan)
            temp = np.where(sal=='2', 'temperature', sal)
            ph = np.where(temp=='3', 'phosphate', temp)
            nit = np.where(ph=='4', 'nitrate', ph)
            sil = np.where(nit=='5', 'silicate', nit)
            oxy = np.where(sil=='6', 'oxygen', sil)
            need[name] = oxy
    
    for p in range(0, len(prod)):
        prodc = np.tile(prod[p], (n, 1))
        prodc = prodc.astype('str')
        for v in range(0, n):
            prodc[v][prodc[v] == '0'] = 'nan'
            prodc[v][prodc[v] == '1'] = salinity[v]
            prodc[v][prodc[v] == '2'] = temperature[v]
            prodc[v][prodc[v] == '3'] = phosphate[v]
            prodc[v][prodc[v] == '4'] = nitrate[v]
            prodc[v][prodc[v] == '5'] = silicate[v]
            prodc[v][prodc[v] == '6'] = oxygen[v]
            prodz.append(prodc)
    
    listz = list(range(0, len(prod)*n, n))
    prodF = []
    
    for l in listz:
        prodF.append(prodz[l])
    
    for r in range(0, len(listz)):
        code2[Name[r]] = prodF[r]  
    
    S, T, A, B, C, code = [], [], [], [], [], {}
    
    for value in code2.values():
        S.append(value[:, 0])
        T.append(value[:, 1])
        A.append(value[:, 2])
        B.append(value[:, 3])
        C.append(value[:, 4])
        
    N = list(code2.keys())
        
    for n in range(0, len(N)):
        data = [S[n], T[n], A[n], B[n], C[n]]
        p = pd.DataFrame(data).T
        p.columns = ['S', 'T', 'A', 'B', 'C']
        i = N[n]
        code[i] = p
        common_columns = ['Order', 'Dates', 'Longitude', 'Latitude', 
'Depth', 'Salinity_u', 'Temperature_u', \
                         'Phosphate_u', 'Nitrate_u', 'Silicate_u', 
'Oxygen_u']
        code[i][common_columns] = InputAll[common_columns]
          
    return need, code
