def emlr_estimate(Equations, DesiredVariables, Path, OutputCoordinates={}, PredictorMeasurements={}, UDict={}, DUDict={}, Coefficients={}, **kwargs):
    
    """
    Uncertainty estimation step 1 for LIRs

    Inputs:
        Equations: List of equations
        DesiredVariables: List of variables to estimate
        Path: User-defined computer path
        OutputCoordinates: Dictionary of locations where estimates are requested
        PredictorMeasurements: Dictionary of measurements provided by user
        UDict: Dictionary of user-defined measurement uncertainties
        DUDict: Dictionary of default measurement uncertainties
        Coefficients: Dictionary of dictionaries of coefficients for each
            variable-equation scenario
        **kwargs: Please see README for full description

    Output:
        EMLR: Dictionary of uncertainty values for each desired variable-equation
            case scenario and estimate
    """

    import numpy as np
    from scipy.interpolate import griddata
    from PyESPER.fetch_data import fetch_data

    # Predefine dictionary and lists to fill
    EMLR, varnames, EqM = {}, [], []

    # Iterating over variables to fetch data this time
    for dv in DesiredVariables:
        # Fetch LIR data and process into grid arrays
        LIR_data = fetch_data([dv], Path)

        # Construct the grid array
        LIR_data = fetch_data([dv], Path)

        # Some formatting of the uncertainties from the import
        arr = np.array(LIR_data)
        arr = arr[3]
        arritem = arr.item()

        UGridArray = np.array([
            np.nan_to_num([arritem[i][c][b][a] for a in range(16) for b in range(11) for c in range(8)])
            for i in range(len(arritem))
        ]).T

        # Grid columns: UDepth, USal, Eqn, RMSE
        UDepth, USal, Eqn, RMSE = UGridArray.T
        UGridPoints = (UDepth, USal, Eqn)
        UGridValues = RMSE
    
        # Iterating over equations within variables to interpolate the uncertainties to
        # desired locations
        for eq in Equations:
            varname = dv + str(eq)
            varnames.append(varname)
            EM = []

            eq_repeated = np.full_like(OutputCoordinates['depth'], eq)
            UGridPointsOut = (
                np.array(OutputCoordinates['depth']),
                np.array(PredictorMeasurements['salinity']),  
                eq_repeated
            )
            emlr = griddata(UGridPoints, UGridValues, UGridPointsOut, method='linear')

            combo = f"{dv}{eq}"
            Coefs = {
                k: np.nan_to_num(np.array(UDict[combo][k]))
                for k in ["US", "UT", "UA", "UB", "UC"]
            }
        
            uncdfs, duncdfs = UDict[combo], DUDict[combo]
        
            # Extract keys
            keys = list(uncdfs.keys())
        
            # Function to fill arrays with floats
            def safe_fill(arr, fill_val):
                arr = np.array(arr, dtype=float)
                arr[np.isnan(arr)] = fill_val
                return arr

            # Fill=-9999 if needed
            USu2 = [safe_fill(uncdfs[k], -9999.0) for k in keys]
            UTu2 = [safe_fill(uncdfs[k], -9999.0) for k in keys]
            UAu2 = [safe_fill(uncdfs[k], -9999.0) for k in keys]
            UBu2 = [safe_fill(uncdfs[k], -9999.0) for k in keys]
            UCu2 = [safe_fill(uncdfs[k], -9999.0) for k in keys]
            
            DUSu2 = [safe_fill(duncdfs[k], -9999.0) for k in keys]
            DUTu2 = [safe_fill(duncdfs[k], -9999.0) for k in keys]
            DUAu2 = [safe_fill(duncdfs[k], -9999.0) for k in keys]
            DUBu2 = [safe_fill(duncdfs[k], -9999.0) for k in keys]
            DUCu2 = [safe_fill(duncdfs[k], -9999.0) for k in keys]

           # Compute uncertainty estimates
            EM = []

            for cucombo in range(len(Coefs["US"])):
                # Grab each coefficient
                s = Coefs["US"][cucombo]
                t = Coefs["UT"][cucombo]
                a = Coefs["UA"][cucombo]
                b = Coefs["UB"][cucombo]
                c = Coefs["UC"][cucombo]

                # Main uncertainty components
                s1 = (s * USu2[0][cucombo]) ** 2
                t1 = (t * UTu2[1][cucombo]) ** 2
                a1 = (a * UAu2[2][cucombo]) ** 2
                b1 = (b * UBu2[3][cucombo]) ** 2
                c1 = (c * UCu2[4][cucombo]) ** 2
                sum2 = s1 + t1 + a1 + b1 + c1

                # Delta uncertainties
                ds1 = (s * DUSu2[0][cucombo]) ** 2
                dt1 = (t * DUTu2[1][cucombo]) ** 2
                da1 = (a * DUAu2[2][cucombo]) ** 2
                db1 = (b * DUBu2[3][cucombo]) ** 2
                dc1 = (c * DUCu2[4][cucombo]) ** 2
                dsum2 = ds1 + dt1 + da1 + db1 + dc1

               # Final uncertainty
                uncestimate = np.sqrt(sum2 - dsum2 + emlr[cucombo] ** 2)
                EM.append(uncestimate)

            EqM.append(EM)
                    
            # Post-process and apply nan masks
            EqM2 = []
            for EM_arr in EqM:
                UncertEst = np.array(EM_arr, dtype=float)
                    
                # Convert -9999 markers to np.nan based on rules
                UncertEst[USu2[0] == -9999] = np.nan
                    
                if eq in [1, 2, 3, 4, 5, 6, 7, 8]:  
                    UncertEst[UTu2[1] == -9999] = np.nan
                if eq in [1, 2, 5, 6, 9, 10, 13, 14]:
                    UncertEst[UAu2[2] == -9999] = np.nan
                if eq in [1, 3, 5, 7, 9, 11, 13, 15]: 
                    UncertEst[UBu2[3] == -9999] = np.nan
                if eq in [1, 2, 3, 4, 9, 10, 11, 12]: 
                    UncertEst[UCu2[4] == -9999] = np.nan

                EqM2.append(UncertEst)
                
            # Final assembly into dictionary
            for i, key in enumerate(varnames):
                EMLR[key] = EqM2[i]
                    
    return EMLR

