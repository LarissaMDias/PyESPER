def iterations(
    DesiredVariables,
    Equations,
    PerKgSwTF,
    C={},
    PredictorMeasurements={},
    InputAll={},
    Uncertainties={},
    DUncertainties={}
):

    """
    A function to iterate and define equation iputs depending upon 
    Equation, DesiredVariable, and other user specifications.

    Inputs:
        DesiredVariables: List of desired variables for estimates
        Equations: List of equations to be used for estimates
        PerKgSwTF: Boolean indicating whether user input was in molal (default) or molar units
        C: Dictionary of pre-processed user geographic coordinates
        PredictorMeasurements: Dictionary of all input, with temperature processed as needed
        InputAll: Dictionary of deafault and user inputs with order stamp
        Uncertainties: Dictionary of user-defined uncertainties (or default if missing)
        DUncertainties: Dictionary of default uncertainties

    Output:
        code: Dictionary of filled-in combinations of predictor measurements relative to the reques$
            variable combination
        unc_combo_dict: Dictionary of filled-in uncertainty combinations for the requisite equation$
        dunc_combo_dict: Dictionary of filled-in default uncertainty combinations, as above

    NOTE: This function uses the now-deprecated seawater package for some calculations. This aligns with the
        current version of ESPERs (v1.0); however, this will be updated to the Gibb's seawater package in the
        next release.
    """

    import numpy as np
    import seawater as sw

    n = max(len(v) for v in C.values()) # Once again calculate number of estimates requested

    # Creating numpy arrays for depth, latitude, and salinity
    depth, latitude, salinity = np.array(C["depth"]), np.array(C["latitude"]), np.array(PredictorMeasurements["salinity"])

    # Calculating potential temperature (will be updated to conservative temperature in the next release)
    temp = np.array(PredictorMeasurements["temperature"]) if "temperature" in PredictorMeasurements else np.full(n, 10)
    temp_sw = sw.ptmp(salinity, temp, sw.pres(depth, latitude), pr=0)
    # Some whole numbers are treated incorrectly as strings by Python, this alleviates this problem
    temp_map = {3: 3.000000001, 4: 4.000000001, 5: 5.000000001, 6: 6.000000001}

    # Convert temp_sw to a NumPy array
    temp_sw_np = np.array(temp_sw)

    # Addressing potentially erroneous values
    mapped_temp = np.vectorize(lambda t: temp_map.get(t, 10 if t < -100 else t))(temp_sw_np)

    # Format each temperature to a string using 15 significant digits, necessary to propertly
    # process iterations in the next step
    temperature_processed = np.array(["{:.15g}".format(val) for val in mapped_temp])

    # Calculating AOU from oxygen and defining nan's as applicable
    if "oxygen" in PredictorMeasurements:
        oxyg = np.array(PredictorMeasurements["oxygen"])
        oxyg_sw = sw.satO2(salinity, temp_sw)*44.6596 - (oxyg)
    else:
        oxyg_sw = np.tile("nan", n)
    # For very small AOUs Python processes the numbers incorrectly, this rounds those to zero to avoid this
    for i in range(len(oxyg_sw)):
        if oxyg_sw[i] != "nan" and -0.0001 < oxyg_sw[i] < 0.0001:
            oxyg_sw[i] = 0
    # Format AOU to string as with temperature, for processing within iterations
    oxygen_processed = ["{:.5g}".format(o) if o != "nan" else o for o in oxyg_sw]
   
    # Process predictor measurements as np arrays or nan's, as applicable
    processed_measurements = {}
    for param in ["phosphate", "nitrate", "silicate"]:
        processed_measurements[param] = (
    np.array(PredictorMeasurements[param]) if param in PredictorMeasurements else np.tile("nan", n)
        )
    
    phosphate_processed = processed_measurements["phosphate"]
    nitrate_processed = processed_measurements["nitrate"]
    silicate_processed = processed_measurements["silicate"]
            
    # Adjust molar to molal units as needed
    if not PerKgSwTF:
        densities = sw.dens(salinity, temperature_processed, sw.pres(depth, latitude)) / 1000
        for nutrient in ["phosphate", "nitrate", "silicate"]:
            if nutrient in PredictorMeasurements:
                globals()[f"{nutrient}_processed"] /= densities
            
    # Convert equations to string for processing
    EqsString = [str(e) for e in Equations]

    # Definition of which input variable is needed for which property
    NeededForProperty = {
            "TA": [1, 2, 4, 6, 5],
            "DIC": [1, 2, 4, 6, 5],
            "pH": [1, 2, 4, 6, 5],
            "phosphate": [1, 2, 4, 6, 5],
            "nitrate": [1, 2, 3, 6, 5],
            "silicate": [1, 2, 3, 6, 4], 
            "oxygen": [1, 2, 3, 4, 5]
        }
    
    # Definition of equations
    VarVec = {
            "1": [1, 1, 1, 1, 1],
            "2": [1, 1, 1, 0, 1],
            "3": [1, 1, 0, 1, 1],
            "4": [1, 1, 0, 0, 1],
            "5": [1, 1, 1, 1, 0],
            "6": [1, 1, 1, 0, 0],
            "7": [1, 1, 0, 1, 0],
            "8": [1, 1, 0, 0, 0],
            "9": [1, 0, 1, 1, 1],
            "10": [1, 0, 1, 0, 1],
            "11": [1, 0, 0, 1, 1],
            "12": [1, 0, 0, 0, 1],
            "13": [1, 0, 1, 1, 0],
            "14": [1, 0, 1, 0, 0],
            "15": [1, 0, 0, 1, 0], 
            "16": [1, 0, 0, 0, 0], 
        }

    # Pre-defining lists and dictionaries to be used for output
    product, product_processed, name = [], [], []
    need, precode = {}, {}
         
    # Create a list of names and process products
    replacement_map = {
        "0": "nan",
        "1": "salinity",
        "2": "temperature",
        "3": "phosphate",
        "4": "nitrate",
        "5": "silicate",
        "6": "oxygen"
    }
            
    # Determining exactly which variable-equation scenarios required
    for d in DesiredVariables:
        dv = np.array(NeededForProperty[d])
        for e in EqsString:
            eq = np.array(VarVec[e])
            prename = d + e
            name.append(prename)  
            
            product.append(eq * dv)
            prodnp = np.array(eq * dv)
    
            # Replace values using the mapping   
            processed = np.vectorize(lambda x: replacement_map.get(str(x), x))(prodnp)
            need[prename] = processed
    
    for p in range(0, len(product)): # Same but for list of input values
        prodnptile = np.tile(product[p], (n, 1))
        prodnptile = prodnptile.astype("str")
        
        for v in range(0, len(salinity)):
            prodnptile[v][prodnptile[v] == "0"] = "nan"
            prodnptile[v][prodnptile[v] == "1"] = salinity[v]
            prodnptile[v][prodnptile[v] == "2"] = temperature_processed[v]
            prodnptile[v][prodnptile[v] == "3"] = phosphate_processed[v]
            prodnptile[v][prodnptile[v] == "4"] = nitrate_processed[v]
            prodnptile[v][prodnptile[v] == "5"] = silicate_processed[v]
            prodnptile[v][prodnptile[v] == "6"] = oxygen_processed[v]
            product_processed.append(prodnptile)
        
    listofprods = np.arange(0, len(product) * n, n)
    prodlist = []
            
    names_values = list(need.values())
    names_keys = list(need.keys()) 
            
    # Repeat for uncertainties
    unc_combo_dict = {}
    dunc_combo_dict = {}
            
    for numb_combos, names_keyscombo in enumerate(names_values):
        # A function to help process the uncertainties and nan's
        def define_unc_arrays(lengthofn, listorder, parnames, unames):
            for numoptions in range(0, len(parnames)):
                if names_keyscombo[listorder] == parnames[numoptions]:
                    udfvalues = np.array(Uncertainties[unames[numoptions]])
                    dudfvalues = np.array(DUncertainties[unames[numoptions]])
                elif names_keyscombo[listorder] == "nan":
                    udfvalues = np.empty((lengthofn))
                    udfvalues[:] = np.nan
                    dudfvalues = np.empty((lengthofn))
                    dudfvalues[:] = np.nan
            return udfvalues, dudfvalues

        for names_items in range(0, len(names_keyscombo)):
            udfvalues1 = np.array(Uncertainties['sal_u'])
            dudfvalues1 = np.array(DUncertainties['sal_u'])
            udfvalues2, dudfvalues2 = define_unc_arrays(n, 1, ["temperature"], ["temp_u"])
            udfvalues3, dudfvalues3 = define_unc_arrays(n, 2, ["nitrate", "phosphate"], ["nitrate_u", "phosphate_u"])
            udfvalues4, dudfvalues4 = define_unc_arrays(n, 3, ["oxygen", "nitrate"], ["oxygen_u", "nitrate_u"])
            udfvalues5, dudfvalues5 = define_unc_arrays(n, 4, ["silicate", "nitrate"], ["silicate_u", "nitrate_u"])
    
        # Convert to NumPy arrays for efficient comparison
        udfvalues = np.array([udfvalues1, udfvalues2, udfvalues3, udfvalues4, udfvalues5])
        dudfvalues = np.array([dudfvalues1, dudfvalues2, dudfvalues3, dudfvalues4, dudfvalues5])
    
        # Update `udfvalues` based on `dudfvalues` using element-wise maximum
        udfvalues = udfvalues.astype(np.float64)
        dudfvalues = dudfvalues.astype(np.float64)
        udfvalues = np.maximum(udfvalues, dudfvalues)
                    
        new_unames = ['US', 'UT', 'UA', 'UB', 'UC']
        # Transpose and convert to dictionaries of NumPy arrays
        uncertaintyvalues_dict = {name: np.array(col) for name, col in zip(new_unames, udfvalues)}
        duncertaintyvalues_dict = {name: np.array(col) for name, col in zip(new_unames, dudfvalues)}
                    
        # Update the target dictionaries  
        unc_combo_dict[names_keys[numb_combos]] = uncertaintyvalues_dict
        dunc_combo_dict[names_keys[numb_combos]] = duncertaintyvalues_dict
        
    # Append the required products to `prodlist` and populate `precode`
    prodlist = [product_processed[item] for item in listofprods]
    precode = {name[i]: prodlist[i] for i in range(len(listofprods))}
            
    S, T, A, B, Z, code = [], [], [], [], [], {}
            
    # Final filling in of values and code
    for value in precode.values():
        S.append(value[:, 0])
        T.append(value[:, 1])
        A.append(value[:, 2])
        B.append(value[:, 3])
        Z.append(value[:, 4])
        
    codenames = list(precode.keys())
                    
    code = {}
    for n, code_name in enumerate(codenames):
        # Create a dictionary for each set of data
        p = {
            "S": np.array(S[n]),
            "T": np.array(T[n]),
            "A": np.array(A[n]),
            "B": np.array(B[n]),
            "C": np.array(Z[n])
        }
    
        for col in ["Order", "Dates", "Longitude", "Latitude", "Depth", "Salinity_u", "Temperature_u", "Phosphate_u", "Nitrate_u", "Silicate_u", "Oxygen_u"]:
            p[col] = np.array(InputAll[col])

        # Assign the dictionary to the code dictionary
        code[code_name] = p

    return code, unc_combo_dict, dunc_combo_dict

