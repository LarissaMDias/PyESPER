def organize_data(
    aaLCs, 
    elLCs, 
    aaInterpolants_pre, 
    elInterpolants_pre, 
    Gdf={}, 
    AAdata={},
    Elsedata={}
    ):

    """
    Organize interpolation output into more usable formatting and compute estimates

    Inputs:
        aaLCs: List of coefficients for input data from Atlantic/Arctic regions
        elLCs: List of coefficients for input data not from Atlantic/Arctic
        aaInterpolants_pre: Scipy interpolant for Atlantic/Arctic
        elInterpolants_pre: Scipy interpolant for not Atlantic/Arctic
        Gdf: Dictionary of grid for interpolation, separated into regions
        AAdata: Dictionary of user input data for Atlantic/Arctic
        ElseData: Dictionary of user input data not for Atlantic/Arctic

    Outputs:
        Estimate: Dictionary of estimates for each equation-desired variable combination
        CoefficientsUsed: Dictionary of dictionaries of coefficients for each equation-
            desired variable combination
    """

    import numpy as np

    # Determine combinations
    Gkeys = list(Gdf.keys())

    AAOvalues, ElseOvalues = list(AAdata.values()), list(Elsedata.values())
    # Initialize lists for storing interpolated values
    aaIntCT2, aaIntCA2, aaIntCB2, aaIntCC2, aaTo2, aaAo2, aaBo2, aaCo2 = [[] for _ in range(8)]
    elIntCT2, elIntCA2, elIntCB2, elIntCC2, elTo2, elAo2, elBo2, elCo2 = [[] for _ in range(8)]
    aaInterpolants, elInterpolants = {}, {}

    # Separate out stored values
    for i in range(0, len(aaLCs)):
        aaIntalpha, elIntalpha = aaLCs[i][:, 0], elLCs[i][:, 0]
        aaIntCS, elIntCS = aaLCs[i][:, 1], elLCs[i][:, 1]
        aaIntCT, elIntCT = aaLCs[i][:, 2], elLCs[i][:, 2]
        aaIntCA, elIntCA = aaLCs[i][:, 3], elLCs[i][:, 3]
        aaIntCB, elIntCB = aaLCs[i][:, 4], elLCs[i][:, 4]
        aaIntCC, elIntCC = aaLCs[i][:, 5], elLCs[i][:, 5]

        # A function to process data in the correct manner and
        # proper handling of nan's
        def process_list(int_values, val_values):
            int_values = np.asarray(int_values, dtype=float)
            val_values = np.asarray(val_values, dtype=object)
    
            # Replace NaNs in int_values with 0
            int2 = np.where(np.isnan(int_values), 0, int_values)
    
            # Replace string "nan" and float NaN in val_values with 0
            val_cleaned = []
            for val in val_values:
                if isinstance(val, str) and val.lower() == "nan":
                    val_cleaned.append(0)
                elif isinstance(val, float) and np.isnan(val):
                    val_cleaned.append(0)
                else:
                    val_cleaned.append(val)
            val2 = np.array(val_cleaned, dtype=object)
        
            return int2, val2

        # Correcting equation instances from nan to zero 
        # for calculations
        key = Gkeys[i]
        is_key_1 = key[-1] == "1" and key[-2] != "1"
        is_key_2 = key[-1] == "2" and key[-2] != "1"
        is_key_3 = key[-1] == "3" and key[-2] != "1"
        is_key_4 = key[-1] == "4" and key[-2] != "1"
        is_key_5 = key[-1] == "5" and key[-2] != "1"
        is_key_6 = key[-1] == "6" and key[-2] != "1"
        is_key_7 = key[-1] == "7"
        is_key_8 = key[-1] == "8"
        is_key_9 = key[-1] == "9"
        is_key_10 = key[-1] == "0" and Gkeys[i][-2] == "1"
        is_key_11 = key[-1] == "1" and key[-2] == "1"
        is_key_12 = key[-1] == "2" and key[-2] == "1"
        is_key_13 = key[-1] == "3" and key[-2] == "1"
        is_key_14 = key[-1] == "4" and key[-2] == "1"
        is_key_15 = key[-1] == "5" and key[-2] == "1"
        is_key_16 = key[-1] == "6" and key[-2] == "1"
                
        # Atlantic and Arctic data equation processing
        aaDatao = AAOvalues[i]
        aaSo, aaTo, aaAo, aaBo, aaCo = aaDatao['S'], aaDatao['T'], aaDatao['A'], aaDatao['B'], aaDatao['C']
            
        # Determine which values to use
        if is_key_1:
            aaIntCT2, aaIntCA2, aaIntCB2, aaIntCC2 = aaIntCT, aaIntCA, aaIntCB, aaIntCC
            aaTo2, aaAo2, aaBo2, aaCo2 = aaTo, aaAo, aaBo, aaCo
        
        elif is_key_2:
            aaIntCT2, aaIntCA2, aaIntCC2 = aaIntCT, aaIntCA, aaIntCC
            aaTo2, aaAo2, aaCo2 = aaTo, aaAo, aaCo  
        
            aaIntCB2, aaBo2 = process_list(aaIntCB, aaBo)
        
        elif is_key_3:
            aaIntCT2, aaIntCB2, aaIntCC2 = aaIntCT, aaIntCB, aaIntCC
            aaTo2, aaBo2, aaCo2 = aaTo, aaBo, aaCo
        
            aaIntCA2, aaAo2 = process_list(aaIntCA, aaAo)
        
        elif is_key_4:
            aaIntCT2, aaIntCC2 = aaIntCT, aaIntCC
            aaTo2, aaCo2 = aaTo, aaCo
            aaIntCA2, aaAo2 = process_list(aaIntCA, aaAo)
            aaIntCB2, aaBo2 = process_list(aaIntCB, aaBo)
        
        elif is_key_5:
            aaIntCT2, aaIntCA2, aaIntCB2 = aaIntCT, aaIntCA, aaIntCB
            aaTo2, aaAo2, aaBo2 = aaTo, aaAo, aaBo
            aaIntCC2, aaCo2 = process_list(aaIntCC, aaCo)

        elif is_key_6:
            aaIntCT2, aaIntCA2 = aaIntCT, aaIntCA
            aaTo2, aaAo2 = aaTo, aaAo
            aaIntCB2, aaBo2 = process_list(aaIntCB, aaBo)
            aaIntCC2, aaCo2 = process_list(aaIntCC, aaCo)
        
        elif is_key_7:
            aaIntCT2, aaIntCB2 = aaIntCT, aaIntCB
            aaTo2, aaBo2 = aaTo, aaBo
            
            aaIntCA2, aaAo2 = process_list(aaIntCA, aaAo)
            aaIntCC2, aaCo2 = process_list(aaIntCC, aaCo)
            
        elif is_key_8:
            aaIntCT2 = aaIntCT
            aaTo2 = aaTo
            
            aaIntCA2, aaAo2 = process_list(aaIntCA, aaAo)
            aaIntCB2, aaBo2 = process_list(aaIntCB, aaBo)
            aaIntCC2, aaCo2 = process_list(aaIntCC, aaCo)
        
        elif is_key_9:
            aaIntCA2, aaIntCB2, aaIntCC2 = aaIntCA, aaIntCB, aaIntCC
            aaAo2, aaBo2, aaCo2 = aaAo, aaBo, aaCo
            
            aaIntCT2, aaTo2 = process_list(aaIntCT, aaTo)

        elif is_key_10:
            aaIntCA2, aaIntCC2 = aaIntCA, aaIntCC
            aaAo2, aaCo2 = aaAo, aaCo
            aaIntCT2, aaTo2 = process_list(aaIntCT, aaTo)
            aaIntCB2, aaBo2 = process_list(aaIntCB, aaBo)
            
        elif is_key_11:
            aaIntCB2, aaIntCC2 = aaIntCB, aaIntCC
            aaBo2, aaCo2 = aaBo, aaCo
            aaIntCT2, aaTo2 = process_list(aaIntCT, aaTo)
            aaIntCA2, aaAo2 = process_list(aaIntCA, aaAo)
        
        elif is_key_12:
            aaIntCC2 = aaIntCC
            aaCo2 = aaCo
            
            aaIntCT2, aaTo2 = process_list(aaIntCT, aaTo)
            aaIntCA2, aaAo2 = process_list(aaIntCA, aaAo)
            aaIntCB2, aaBo2 = process_list(aaIntCB, aaBo)

        elif is_key_13:
            aaIntCA2, aaIntCB2 = aaIntCA, aaIntCB 
            aaAo2, aaBo2 = aaAo, aaBo
            
            aaIntCT2, aaTo2 = process_list(aaIntCT, aaTo)
            aaIntCC2, aaCo2 = process_list(aaIntCC, aaCo)
            
        elif is_key_14:
            aaIntCA2 = aaIntCA
            aaAo2 = aaAo
            aaIntCT2, aaTo2 = process_list(aaIntCT, aaTo)
            aaIntCB2, aaBo2 = process_list(aaIntCB, aaBo)
            aaIntCC2, aaCo2 = process_list(aaIntCC, aaCo)
            
        elif is_key_15:
            aaIntCB2 = aaIntCB
            aaBo2 = aaBo
        
            aaIntCT2, aaTo2 = process_list(aaIntCT, aaTo)
            aaIntCA2, aaAo2 = process_list(aaIntCA, aaAo)
            aaIntCC2, aaCo2 = process_list(aaIntCC, aaCo)
            
        elif is_key_16:
            aaIntCT2, aaTo2 = process_list(aaIntCT, aaTo)
            aaIntCA2, aaAo2 = process_list(aaIntCA, aaAo)
            aaIntCB2, aaBo2 = process_list(aaIntCB, aaBo)
            aaIntCC2, aaCo2 = process_list(aaIntCC, aaCo)

        # Convert data lists to NumPy arrays and fix specific value
        aaAo2 = ['-0.000002' if x == '-2.4319000000000003e-' else x for x in aaAo2]
        data = [aaIntalpha, aaIntCS, aaIntCT2, aaIntCA2, aaIntCB2, aaIntCC2, aaSo, aaTo2, aaAo2, aaBo2, aaCo2]
        aaIal, aaICS, aaICT, aaICA, aaICB, aaICC, aaS, aaT, aaA, aaB, aaC = map(lambda x: np.array(x, dtype=float), data)
        
        # Compute `aaEst`, the estimate for Atlantic and Arctic
        aaEst = np.array([a + b*c + d*e + f*g + h*i + j*k
                          for a, b, c, d, e, f, g, h, i, j, k
                          in zip(aaIal, aaICS, aaS, aaICT, aaT, aaICA, aaA, aaICB, aaB, aaICC, aaC)])    
            
        # Store results
        aaInterpolants[key] = (aaIal, aaICS, aaICT, aaICA, aaICB, aaICC, aaEst)
            
        # Reprocessing "NaN" to 0 as needed for calculations for non-Atlantic and Arctic
        elDatao = ElseOvalues[i]
        elSo, elTo, elAo, elBo, elCo = elDatao['S'], elDatao['T'], elDatao['A'], elDatao['B'], elDatao['C']
            
        # Determine which values to use
        if is_key_1:
            elIntCT2, elIntCA2, elIntCB2, elIntCC2 = elIntCT, elIntCA, elIntCB, elIntCC
            elTo2, elAo2, elBo2, elCo2 = elTo, elAo, elBo, elCo
            
        elif is_key_2:
            elIntCT2, elIntCA2, elIntCC2 = elIntCT, elIntCA, elIntCC
            elTo2, elAo2, elCo2 = elTo, elAo, elCo
        
            elIntCB2, elBo2 = process_list(elIntCB, elBo)
        
        elif is_key_3:
            elIntCT2, elIntCB2, elIntCC2 = elIntCT, elIntCB, elIntCC
            elTo2, elBo2, elCo2 = elTo, elBo, elCo
        
            elIntCA2, elAo2 = process_list(elIntCA, elAo)
                          
        elif is_key_4:
            elIntCT2, elIntCC2 = elIntCT, elIntCC
            elTo2, elCo2 = elTo, elCo
            
            elIntCA2, elAo2 = process_list(elIntCA, elAo)
            elIntCB2, elBo2 = process_list(elIntCB, elBo)
        
        elif is_key_5:
            elIntCT2, elIntCA2, elIntCB2 = elIntCT, elIntCA, elIntCB
            elTo2, elAo2, elBo2 = elTo, elAo, elBo
            
            elIntCC2, elCo2 = process_list(elIntCC, elCo)
            
        elif is_key_6:
            elIntCT2, elIntCA2 = elIntCT, elIntCA
            elTo2, elAo2 = elTo, elAo
        
            elIntCB2, elBo2 = process_list(elIntCB, elBo)
            elIntCC2, elCo2 = process_list(elIntCC, elCo)
        
        elif is_key_7:
            elIntCT2, elIntCB2 = elIntCT, elIntCB 
            elTo2, elBo2 = elTo, elBo
            
            elIntCA2, elAo2 = process_list(elIntCA, elAo)
            elIntCC2, elCo2 = process_list(elIntCC, elCo)
            
        elif is_key_8:
            elIntCT2 = elIntCT
            elTo2 = elTo
            
            elIntCA2, elAo2 = process_list(elIntCA, elAo)
            elIntCB2, elBo2 = process_list(elIntCB, elBo)
            elIntCC2, elCo2 = process_list(elIntCC, elCo)
            
        elif is_key_9:
            elIntCA2, elIntCB2, elIntCC2 = elIntCA, elIntCB, elIntCC
            elAo2, elBo2, elCo2 = elAo, elBo, elCo
        
            elIntCT2, elTo2 = process_list(elIntCT, elTo)
            
        elif is_key_10:
            elIntCA2, elIntCC2 = elIntCA, elIntCC
            elAo2, elCo2 = elAo, elCo
        
            elIntCT2, elTo2 = process_list(elIntCT, elTo)
            elIntCB2, elBo2 = process_list(elIntCB, elBo)
            
        elif is_key_11:
            elIntCB2, elIntCC2 = elIntCB, elIntCC
            elBo2, elCo2 = elBo, elCo
            
            elIntCT2, elTo2 = process_list(elIntCT, elTo)
            elIntCA2, elAo2 = process_list(elIntCA, elAo)
            
        elif is_key_12:
            elIntCC2 = elIntCC
            elCo2 = elCo
            
            elIntCT2, elTo2 = process_list(elIntCT, elTo)
            elIntCA2, elAo2 = process_list(elIntCA, elAo)
            elIntCB2, elBo2 = process_list(elIntCB, elBo)
            
        elif is_key_13:
            elIntCA2, elIntCB2 = elIntCA, elIntCB
            elAo2, elBo2 = elAo, elBo
        
            elIntCT2, elTo2 = process_list(elIntCT, elTo)
            elIntCC2, elCo2 = process_list(elIntCC, elCo)
        
        elif is_key_14:
            elIntCA2 = elIntCA
            elAo2 = elAo
        
            elIntCT2, elTo2 = process_list(elIntCT, elTo)
            elIntCB2, elBo2 = process_list(elIntCB, elBo)
            elIntCC2, elCo2 = process_list(elIntCC, elCo)
            
        elif is_key_15:
            elIntCB2 = elIntCB
            elBo2 = elBo
            
            elIntCT2, elTo2 = process_list(elIntCT, elTo)
            elIntCA2, elAo2 = process_list(elIntCA, elAo)
            elIntCC2, elCo2 = process_list(elIntCC, elCo)
            
        elif is_key_16:
            
            elIntCT2, elTo2 = process_list(elIntCT, elTo)
            elIntCA2, elAo2 = process_list(elIntCA, elAo)
            elIntCB2, elBo2 = process_list(elIntCB, elBo)
            elIntCC2, elCo2 = process_list(elIntCC, elCo)

        # Convert all input lists to NumPy arrays in one go
        data2 = [elIntalpha, elIntCS, elIntCT2, elIntCA2, elIntCB2, elIntCC2, elSo, elTo2, elAo2, elBo2, elCo2]
        elIal, elICS, elICT, elICA, elICB, elICC, elS, elT, elA, elB, elC = map(lambda x: np.array(x, dtype=float), data2)
            
        # compute 'elEst', the estimate for not Alantic or Atctic
        elEst = np.array([a + b*c + d*e + f*g + h*i + j*k
                          for a, b, c, d, e, f, g, h, i, j, k
                          in zip(elIal, elICS, elS, elICT, elT, elICA, elA, elICB, elB, elICC, elC)])
        # Store the results
        elInterpolants[key] = (elIal, elICS, elICT, elICA, elICB, elICC, elEst)

    Estimate, CoefficientsUsed = {}, {}
    for kcombo in AAdata.keys():
        AAdata[kcombo]["C0"] = aaInterpolants[kcombo][0]
        AAdata[kcombo]["CS"] = aaInterpolants[kcombo][1] 
        AAdata[kcombo]["CT"] = aaInterpolants[kcombo][2] 
        AAdata[kcombo]["CA"] = aaInterpolants[kcombo][3] 
        AAdata[kcombo]["CB"] = aaInterpolants[kcombo][4]
        AAdata[kcombo]["CC"] = aaInterpolants[kcombo][5]
        AAdata[kcombo]["Estimate"] = aaInterpolants[kcombo][6]
    for kcombo in Elsedata.keys():
        Elsedata[kcombo]["C0"] = elInterpolants[kcombo][0]
        Elsedata[kcombo]["CS"] = elInterpolants[kcombo][1]
        Elsedata[kcombo]["CT"] = elInterpolants[kcombo][2]
        Elsedata[kcombo]["CA"] = elInterpolants[kcombo][3]
        Elsedata[kcombo]["CB"] = elInterpolants[kcombo][4] 
        Elsedata[kcombo]["CC"] = elInterpolants[kcombo][5]
        Elsedata[kcombo]["Estimate"] = elInterpolants[kcombo][6]
            
        # Merge AA and Else data by key
        merged = {}
        for key in AAdata[kcombo].keys():
            merged[key] = np.concatenate([AAdata[kcombo][key], Elsedata[kcombo][key]])
        
        # Get sort order based on "Order"
        sort_index = np.argsort(merged["Order"])
    
        # Sort each field in the merged dictionary
        TotData = {key: val[sort_index] for key, val in merged.items()}
        
        # Store estimate values as dictionary with 1 key 
        Estimate[kcombo] =  TotData["Estimate"]

        # Store coefficients as dictionary with named keys
        CoefficientsUsed[kcombo] = {
            "Intercept": TotData["C0"],
            "Coef S": TotData["CS"],
            "Coef T": TotData["CT"],
            "Coef A": TotData["CA"],
            "Coef B": TotData["CB"],
            "Coef C": TotData["CC"]
        }
        
    return Estimate, CoefficientsUsed
