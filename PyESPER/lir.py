def PyESPER_LIR(DesiredVariables, Path, OutputCoordinates={}, PredictorMeasurements={}, **kwargs):
    
    """
    Locally Interpolated Regressions (LIRs) for Empirical Seawater Property Estimation
    Python interpretation of LIRv.3; ESPERv.1.1

    Empirical Seawater Property Estimation Routines: Estimates seawater properties and estimate uncertainty from combinations of other parameter
    measurements.  PYESPER_LIR refers specifically to code that uses collections of interpolated linear regressions as opposed to neural
    networks, and Python rather than MATLAB coding languages.  

    Reserved for version update notes: (no updates, first version)
  
    Documentation and citations:
    LIARv1: Carter et al., 2016, doi: 10.1002/lom3.10087
    LIARv2, LIPHR, LINR citation: Carter et al., 2018, https://doi.org/10.1002/lom3.10232
    LIPR, LISIR, LIOR, first described/used: Carter et al., 2021, https://doi.org/10.1029/2020GB006623
    LIRv3 and ESPER_NN (ESPERv1.1): Carter, 2021, https://10.5281/ZENODO.5512697

    PyESPER_LIR is a Python replicate of ESPER_LIR:
    Carter et al. 2021: https://doi.org/10.1002/lom3.10461
    ESPER_NN is inspired by CANYON-B, which also uses neural networks: 
    Bittig et al. 2018: https://doi.org/10.3389/fmars.2018.00328

    This function needs numpy, scipy, pandas, math, matplotlib, importlib, and statistics packages. The seawater package is required if 
    measurements are provided in molar units or if potential temperature or AOU are needed but not provided by the user.  Scale differences from 
    TEOS-10 are a negligible component of alkalinity estimate error. PyCO2SYS is required if pH on the total scale is a desired output variable. 
 
   ****************************************************************************
    Input/Output dimensions:
    ............................................................................
    p:   Integer number of desired property estimate types (e.g., TA, pH, NO3-)
    n:   Integer number of desired estimate locations
    e:   Integer number of equations used at each location
    y:   Integer number of parameter measurement types provided by the user.
    n*e: Total number of estimates returned as an n by e array
    ****************************************************************************
    
    Required Inputs:
   
    DesiredVariables: (required 1 by p list, where p specifies the desired variable(s) in string format):
        List elements specify which variables will be returned. Excepting unitless pH, all outputs are in micromol per kg seawater. Naming of list
        elements must be exactly as demonstrated below (exs: ["TA"], ["DIC", "phosphate", "oxygen"]).

        Desired Variable:                    | List Element Name (String Format):
        *********************************************************************
        Total Titration Seawater Alkalinity  | TA
        Total Dissolved Inorganic Carbon     | DIC
        in situ pH on the total scale        | pH
        Phosphate                            | phosphate
        Nitrate                              | nitrate
        Silicate                             | silicate
        Dissolved Oxygen (O2)                | oxygen

    Path (required string):
        Path directing Python to the location of saved/downloaded LIR files on the user's computer (ex: '/Users/lara/Documents/Python'). 

    OutputCoordinates (required n by 3 dictionary, where n are the number of desired estimate locations and the three dictionary keys are 
    longitude, latitude, and depth): 
        Coordinates at which estimates are desired.  The keys should be longitude (degrees E), latitude (degrees N), and positive integer depth  
        (m), with dictionary keys named 'longitude', 'latitude', and 'depth' (ex: OutputCoordinates={"longitude": [0, 180, -50, 10], "latitude": 
        [85, -20, 18, 0.5], "depth": [10, 1000, 0, 0]} or OutputCoordinates={'longitude': long, 'latitude': lat, 'depth': depth} when referring 
        to a set of predefined lists or numpy arrays of latitude, longitude, and depth information.
 
    PredictorMeasurements (required n by y dictionary, where n are the number of desired estimate locations and y are the dictionary keys  
    representing each possible input): 
       Parameter measurements that will be used to estimate desired variables. Concentrations should be expressed as micromol per kg seawater  
       unless PerKgSwTF is set to false in which case they should be expressed as micromol per L, temperature should be expressed as degrees C, and 
       salinity should be specified with the unitless convention.  NaN inputs are acceptable, but will lead to NaN estimates for any equations that 
       depend on that parameter.The key order (y columns) is arbitrary, but naming of keys must adhere to  the following convention (ex: 
       PredictorMeasurements={"salinity":[35, 34.1, 32, 33], "temperature": [0.1, 10, 0.5, 2], "oxygen": [202.3, 214.7, 220.5, 224.2]}or 
       PredictorMeasurements={'salinity': sal, 'temperature': temp, 'phosphate': phos, 'nitrate': nitr} when referring to predefined lists or 
       numpy arrays of measurements):

       Input Parameter:                       | Dictionary Key Name:
       **********************************************************************
       Salinity                               | salinity
       Temperature                            | temperature
       Phosphate                              | phosphate
       Nitrate                                | nitrate
       Silicate                               | silicate
       O2                                     | oxygen   
       **********************************************************************
       
    Optional inputs:  All remaining inputs must be specified as sequential input argument pairs (e.g. "EstDates"=EstDates when referring to a 
    predefined list of dates, 'Equations'=[1:16], pHCalcTF=True, etc.)

    EstDates (optional but recommended n by 1 list or 1 by 1 value, default 2002.0): 
        A list of decimal dates for the estimates (e.g. July 1 2020 would be "2020.5").  If only a single date is supplied that value is used
        for all estimates.  It is highly recommended that date(s) be provided for estimates of DIC and pH.  This version of the code will accept
        1 by n inputs as well.
    
    Equations (optional 1 by e list, default [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]): 
        List indicating which equations will be used to estimate desired variables. If [] is input or the input is not specified then all 16 
        equations will be used.
     
         (S=salinity, T=temperature, oxygen=dissolved oxygen molecule... see 'PredictorMeasurements' for units).
        ...............................................................
        Output Equation Key (See below for explanation of A, B, and C):
        1.  S, T, A, B, C
        2.  S, T, A, C
        3.  S, T, B, C
        4.  S, T, C
        5.  S, T, A, B
        6.  S, T, A
        7.  S, T, B
        8.  S, T
        9.  S, A, B, C
        10. S, A, C
        11. S, B, C
        12. S, C
        13. S, A, B
        14. S, A
        15. S, B
        16. S
    
        DesiredVar   | A             B             C
        _____________|_____________________________________
        TA           | nitrate       oxygen        silicate
        DIC          | nitrate       oxygen        silicate
        pH           | nitrate       oxygen        silicate
        phosphate    | nitrate       oxygen        silicate
        nitrate      | phosphate     oxygen        silicate
        silicate     | phosphate     oxygen        nitrate
        O2           | phosphate     nitrate       silicate
    
    MeasUncerts (Optional n by y dictionary or 1 by y dictionary, default: [0.003 S, 0.003 degrees C T or potential temperature, 2% phosphate, 
    2% nitrate, 2% silicate, 1% AOU or O2]): 
        Dictionary of measurement uncertainties (see 'PredictorMeasurements' for units). Providing these estimates will improve PyESPER_LIR
        estimate uncertainties. Measurement uncertainties are a small part of PyESPER_LIR estimate uncertainties for WOCE-quality measurements. 
        However, estimate uncertainty scales with measurement uncertainty, so it is recommended that measurement uncertainties be specified for 
        sensor measurements.  If this optional input argument is not provided, the default WOCE-quality uncertainty is assumed.  If a 1 by y array 
        is provided then the uncertainty estimates are assumed to apply uniformly to all input parameter measurements. Uncertainties should be 
        presented with the following naming convention:

       Input Uncertainties:                   | Key Name:
       ********************************************************
       Salinity                               | sal_u
       Temperature                            | temp_u
       Phosphate                              | phosphate_u
       Nitrate                                | nitrate_u
       Silicate                               | silicate_u
       O2                                     | oxygen_u
       
    pHCalcTF (Optional boolean, default false): 
        If set to true, PyESPER will recalculate the pH to be a better estimate of what the seawater pH value would be if calculated from TA and 
        DIC instead of measured with purified m-cresol dye. This is arguably also a better estimate of the pH that would be obtained from pre-2011
        measurements with impure dyes.  See the LIPHR paper for details
    
    PerKgSwTF (Optional boolean, default true): 
        Many sensors provide measurements in micromol per L (molarity) instead of micromol per kg seawater. Indicate false if provided
        measurements are expressed in molar units (concentrations must be micromol per L if so).  Outputs will remain in molal units regardless.
    
    VerboseTF (Optional boolean, default true): 
        Setting this to false will reduce the number of updates, warnings, and errors printed by PyESPER_NN. And additional step can be taken  
        before executing the PyESPER_LIR function (see below) that will further reduce updates, warnings, and errors, if desired. 
        
    *************************************************************************
    Outputs:
 
    Estimates: 
        A n by e pandas DataFrame of estimates specific to the coordinates and parameter measurements provided as inputs.  Units are micromoles  
        per kg (equivalent to the deprecated microeq per kg seawater). Column names are the unique desired variable-equation combinations  
        requested by the user. 

    Coefficients:
        A n by e pandas DataFrame of equation intercepts and coefficients specific to the coordinates and parameter measurements provided as 
        inputs. Column names are the unique desired variable-equation combinations requested by the user. 
	
     Uncertainties: 
        A n by e dictionary of uncertainty estimates specific to the coordinates, parameter measurements, and parameter uncertainties provided.
        Units are micromoles per kg (equivalent to the deprecated microeq per kg seawater). Column names are the unique desired variable-equation 
        combinations requested by the user. 
        
    *************************************************************************
    Missing data: should be indicated with a NaN.  A nan coordinate will yield nan estimates for all equations at that coordinate.  A NaN
    parameter value will yield NaN estimates for all equations that require that parameter.
 
    *************************************************************************
    Please send questions or related requests about PyESPER to lmdias@uw.edu.
    ************************************************************************* 
    """

     # Importing packages
    import math
    import os
    import time

    import matplotlib.path as mpltPath
    import numpy as np
    import pandas as pd
    import PyCO2SYS as pyco2
    import scipy.interpolate
    import seawater as sw
    from scipy.interpolate import griddata
    from scipy.io import loadmat
    from scipy.spatial import Delaunay

     # Starting the timer
    tic = time.perf_counter() 

     # Checking for presence of required input parameters and raising a custom error message if needed
    class CustomError(Exception):
        pass
    
    required_coords = ("longitude", "latitude", "depth")
    for coord_name in required_coords:
        if coord_name not in OutputCoordinates:
            raise CustomError(f"Warning: Missing {coord_name} in OutputCoordinates.")
            
    if "salinity" not in PredictorMeasurements: 
        raise CustomError("Warning: Missing salinity measurements. Salinity is a required input.")
            
    if "oxygen" in PredictorMeasurements and "temperature" not in PredictorMeasurements:
        raise CustomError("Warning: Missing temperature measurements. Temperature is required when oxygen is provided.")

    # Check temperature sanity and print a warning for out-of-range values
    if "temperature" in PredictorMeasurements and any(t < -5 or t > 50 for t in PredictorMeasurements["temperature"]):
        print("Warning: Temperatures below -5°C or above 50°C found. PyESPER is not designed for seawater with these properties. Ensure temperatures are in Celsius.")
                
    if any(s < 5 or s > 50 for s in PredictorMeasurements["salinity"]):
        print("Warning: Salinities less than 5 or greater than 50 have been found. ESPER is not intended for seawater with these properties.")
        
    if any(d < 0 for d in OutputCoordinates["depth"]):
        print("Warning: Depth can not be negative.")
        
    if any(l > 90 for l in OutputCoordinates["latitude"]):
        print("Warning: A latitude >90 deg (N or S) has been detected. Verify latitude is entered correctly as an input.")
    
    # Checking for commonly used missing data indicator flags. Consider adding your commonly used flags here.  
    if any(l == -9999 or l == -9 or l == -1e20 for l in OutputCoordinates["latitude"]):
           print("Warning: A common non-NaN missing data indicator (e.g., -999, -9, -1e20) was detected in the input measurements provided. Missing data should be replaced with NaNs. Otherwise, ESPER will interpret your inputs at face value and give terrible estimates.")  
  
    # Check and define Equations based on user-defined kwargs, or use default values
    Equations = kwargs.get("Equations", list(range(1, 17)))
        
    # Reading dimensions of user input
    n = max(len(v) for v in OutputCoordinates.values()) # number of rows out
                
    # Checking kwargs for presence of VerboseTF and EstDates, and Equations, and defining defaults, as needed
    VerboseTF = kwargs.get("VerboseTF", True)
        
    # Set EstDates based on kwargs, defaulting to 2002.0 if not provided
    if "EstDates" in kwargs:
        d = np.array(kwargs["EstDates"])
        EstDates = (
            [item for sublist in [kwargs["EstDates"]] * (n + 1) for item in sublist]
            if len(d) != n else list(d)
        )
    else:
        EstDates = [2002.0] * n
        
    # Bookkeeping coordinates
    C = {}
    longitude = np.array(OutputCoordinates["longitude"])
    longitude[longitude > 360] = np.remainder(longitude[longitude > 360], 360)
    longitude[longitude < 0] = longitude[longitude<0] + 360
    C["longitude"] = longitude
    C["latitude"] = OutputCoordinates["latitude"]
    C["depth"] = OutputCoordinates["depth"]   
    
    # Defining or reading in PerKgSwTF
    PerKgSwTF = kwargs.get("PerKgSwTF", True)

    def process_uncertainties(param, default_factor, PredictorMeasurements, n):
        if param in MeasUncerts:
            result = np.array(MeasUncerts.get(param))
            if len(result) < n:
                result = np.tile(result, n)
            if param.replace('_u', '') in PredictorMeasurements:
                dresult = np.array([i * default_factor for i in PredictorMeasurements[param.replace('_u', '')]])
            else:
                dresult = result
        else:
            if param.replace('_u', '') in PredictorMeasurements:
                result = np.array([i * default_factor for i in PredictorMeasurements[param.replace('_u', '')]])
                dresult = result
            else:
                result = np.tile('nan', n)
                dresult = np.tile(0, n)
        return result, dresult

    MeasUncerts_processed, DefaultUAll = {}, {}
    MeasUncerts = kwargs.get("MeasUncerts", {})

    # Validate MeasUncerts dimensions
    if MeasUncerts:
        if max(len(v) for v in MeasUncerts.values()) != n:
            if min(len(v) for v in MeasUncerts.values()) != 1:
                raise CustomError(
                    "MeasUncerts must be undefined, a vector with the same number of elements as "
                    "PredictorMeasurements has columns, or a matrix of identical dimension to PredictorMeasurements."
                )
        if len(MeasUncerts) != len(PredictorMeasurements):
            print("Warning: Different numbers of input uncertainties and input measurements.")

    # Default salinity uncertainties
    sal_u = np.array(MeasUncerts.get("sal_u", [0.003]))
    sal_u = np.tile(sal_u, n) if len(sal_u) < n else sal_u
    sal_defu = np.tile(0.003, n)

    # Temperature uncertainties
    temp_u = np.tile(np.array(MeasUncerts.get("temp_u", [0.003])), n) if "temp_u" in MeasUncerts or "temperature" in PredictorMeasurements else np.tile("nan", n)
    temp_defu = np.tile(0.003 if "temp_u" in MeasUncerts or "temperature" in PredictorMeasurements else 0, n)

    # Process other parameters
    parameters = {
        "phosphate_u": 0.02,
        "nitrate_u": 0.02,
        "silicate_u": 0.02,
        "oxygen_u": 0.01
    }

    for param, factor in parameters.items():
        MeasUncerts_processed[param], DefaultUAll[f"{param.replace('_u', '_defu')}"] = process_uncertainties(
            param, factor, PredictorMeasurements, n
        )

    # Update MeasUncerts and DefaultUAll dictionaries
    meas_uncerts_keys = ["sal_u", "temp_u", *parameters.keys()]
    default_uall_keys = ["sal_defu", "temp_defu", *[k.replace('_u', '_defu') for k in parameters.keys()]]

    MeasUncerts.update(dict(zip(meas_uncerts_keys, [sal_u, temp_u, *MeasUncerts_processed.values()])))
    DefaultUAll.update(dict(zip(default_uall_keys, [sal_defu, temp_defu, *DefaultUAll.values()])))

    # Create DataFrames
    keys = meas_uncerts_keys
    Uncerts = np.column_stack([MeasUncerts[k] for k in keys])
    Uncertainties_pre = pd.DataFrame(Uncerts, columns=keys)

    DUncerts = np.column_stack([DefaultUAll[k] for k in default_uall_keys])
    DUncertainties_pre = pd.DataFrame(DUncerts, columns=keys)

    # This function is the primary function of the PyESPER_LIR, which preprocesses all data and performs the interpolation
    def preprocess_interpolate(
        DesiredVariables, 
        Equations, 
        EstDates, 
        VerboseTF, 
        C={}, 
        PredictorMeasurements={}, 
        Uncertainties={}, 
        DUncertainties={}
    ):
    
        n = max(len(v) for v in C.values()) # number of rows out

        # Redefining and organizing all data thus far
        order = list(range(n))
        input_data = {
            "Order": order,
            "Longitude": C["longitude"],
            "Latitude": C["latitude"],
            "Depth": C["depth"],
            "Salinity": PredictorMeasurements["salinity"],
            "Dates": EstDates,
            "Salinity_u": Uncertainties["sal_u"],
            "Temperature_u": Uncertainties["temp_u"],
            "Phosphate_u": Uncertainties["phosphate_u"],
            "Nitrate_u": Uncertainties["nitrate_u"],
            "Silicate_u": Uncertainties["silicate_u"],
            "Oxygen_u": Uncertainties["oxygen_u"]
        }
    
        # Map PredictorMeasurements keys to input_data keys
        for key, label in {
            "temperature": "Temperature",
            "phosphate": "Phosphate",
            "nitrate": "Nitrate",
            "silicate": "Silicate",
            "oxygen": "Oxygen"
        }.items():
            if key in PredictorMeasurements:
                input_data[label] = PredictorMeasurements[key]

        InputAll = pd.DataFrame(input_data)
        # created a dataframe with order stamp and dropped all nans from a replicate dataframe

        # Printing a custom warning if temperature is absent but needed 
        if "EstDates" in kwargs and "pH" in DesiredVariables:
            if "temperature" not in PredictorMeasurements:
                print(
                    "Warning: Carbonate system calculations will be used to adjust the pH, but no temperature is "
                    "specified so 10 C will be assumed. If this is a poor estimate for your region, consider supplying "
                    "your own value in the PredictorMeasurements input."
                )
                Temperature = [10] * n
            else:
                Temperature = InputAll["Temperature"]
    
            PredictorMeasurements["temperature"] = Temperature
            InputAll["temperature"] = Temperature

        # Beginning treatment of inputs and iterations 
        depth, latitude, salinity = np.array(C["depth"]), np.array(C["latitude"]), np.array(PredictorMeasurements["salinity"])
        temp = np.array(PredictorMeasurements["temperature"]) if "temperature" in PredictorMeasurements else np.full(n, 10)
        temp_sw = sw.ptmp(salinity, temp, sw.pres(depth, latitude), pr=0)
        temperature_processed = [
            "{:.15g}".format(
                {3: 3.000000001, 4: 4.000000001, 5: 5.000000001, 6: 6.000000001}.get(t, 10 if t < -100 else t)
            ) 
            for t in temp_sw
        ]
        if "oxygen" in PredictorMeasurements:
            oxyg = np.array(PredictorMeasurements["oxygen"])
            oxyg_sw = sw.satO2(salinity, temp_sw)*44.6596 - (oxyg)
        else: 
            oxyg_sw = np.tile("nan", n)
        for i in range(len(oxyg_sw)):
            if oxyg_sw[i] != "nan" and -0.0001 < oxyg_sw[i] < 0.0001:
                oxyg_sw[i] = 0
        oxygen_processed = ["{:.5g}".format(o) if o != "nan" else o for o in oxyg_sw]
        # Process predictor measurements
        processed_measurements = {}
        for param in ["phosphate", "nitrate", "silicate"]:
            processed_measurements[param] = (
        np.array(PredictorMeasurements[param]) if param in PredictorMeasurements else np.tile("nan", n)
            )

        phosphate_processed = processed_measurements["phosphate"]
        nitrate_processed = processed_measurements["nitrate"]
        silicate_processed = processed_measurements["silicate"]
    
        if not PerKgSwTF:
            densities = sw.dens(salinity, temperature_processed, sw.pres(depth, latitude)) / 1000
            for nutrient in ["phosphate", "nitrate", "silicate"]:
                if nutrient in PredictorMeasurements:
                    globals()[f"{nutrient}_processed"] /= densities
            
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
            })
    
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

        for d in DesiredVariables:
            dv = NeededForProperty[d]
            for e in EqsString:
                eq = VarVec[e]
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
                
        listofprods = list(range(0, len(product)*n, n))
        prodlist = []

        names_values = list(need.values())
        names_keys = list(need.keys())
        unc_combo_dict = {}
        dunc_combo_dict = {}

        def get_uncertainty_array(name, uncertainties, default_size):
            if name in uncertainties:
                return np.array(uncertainties[name])
            else:
                return np.full(default_size, np.nan)
                
        for numb_combos, names_keyscombo in enumerate(names_values):

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
            
            for names_items in range(0, len(names_keyscombo)): # Fix this later
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
            udfvalues = np.maximum(udfvalues, dudfvalues)

            # Create DataFrames and set column names
            new_unames = ['US', 'UT', 'UA', 'UB', 'UC']
            uncertaintyvalues_df = pd.DataFrame(udfvalues.T, columns=new_unames)
            duncertaintyvalues_df = pd.DataFrame(dudfvalues.T, columns=new_unames)

            # Update dictionaries
            unc_combo_dict[names_keys[numb_combos]] = uncertaintyvalues_df
            dunc_combo_dict[names_keys[numb_combos]] = duncertaintyvalues_df

        # Append the required products to `prodlist` and populate `precode`
        prodlist = [product_processed[item] for item in listofprods]
        precode = {name[i]: prodlist[i] for i in range(len(listofprods))}

        S, T, A, B, Z, code = [], [], [], [], [], {}
    
        for value in precode.values():
            S.append(value[:, 0])
            T.append(value[:, 1])
            A.append(value[:, 2])
            B.append(value[:, 3])
            Z.append(value[:, 4])
        
        codenames = list(precode.keys())
        
        for n, code_name in enumerate(codenames):
            # Create a DataFrame for each set of data
            data = [S[n], T[n], A[n], B[n], Z[n]]
            p = pd.DataFrame(data).T
            p.columns = ["S", "T", "A", "B", "C"]
            
            # Assign the DataFrame to the dictionary with the code name as the key
            code[code_name] = p

            # List of common columns to be added
            common_columns = ["Order", "Dates", "Longitude", "Latitude", "Depth", "Salinity_u", "Temperature_u", 
                              "Phosphate_u", "Nitrate_u", "Silicate_u", "Oxygen_u"]

            # Assign the common columns from InputAll to the DataFrame
            code[code_name][common_columns] = InputAll[common_columns]
        
        # Loading the data
        AAIndsCs, GridCoords, Cs = {}, {}, {}
        def fetch_data (DesiredVariables):
            for v in DesiredVariables:               
                fname1 = os.path.join(Path, f"Mat_fullgrid/LIR_files_{v}_fullCs1.mat")
                fname2 = os.path.join(Path, f"Mat_fullgrid/LIR_files_{v}_fullCs2.mat")
                fname3 = os.path.join(Path, f"Mat_fullgrid/LIR_files_{v}_fullCs3.mat")
                fname4 = os.path.join(Path, f"Mat_fullgrid/LIR_files_{v}_fullGrids.mat")

                Cs1 = loadmat(fname1)
                Cs2 = loadmat(fname2)
                Cs3 = loadmat(fname3)
                Grid = loadmat(fname4)
                
                UncGrid = Grid["UncGrid"][0][0] 
                GridCoodata, AAinds = np.array(Grid["GridCoords"]), np.array(Grid["AAIndsM"])
                Csdata1, Csdata2, Csdata3 = np.array(Cs1["Cs1"]), np.array(Cs2["Cs2"]), np.array(Cs3["Cs3"]) 
                AAIndsCs[v] = pd.DataFrame(data=AAinds)
                GridCoords[v] = pd.DataFrame(data=GridCoodata[:, :])
                Csdata = np.concatenate((Csdata1, Csdata2, Csdata3), axis=1)
                Cs[v] = [pd.DataFrame(data=Csdata[:, :, i]) for i in range(16)]
      
            LIR_data = [GridCoords, Cs, AAIndsCs, UncGrid]
            return LIR_data

        LIR_data = fetch_data(DesiredVariables)
        
        # Assessing the locations/regions of user-provided outputcoordinates
        # Define Polygons
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

        # Create DataFrame
        df = pd.DataFrame({'AAInds': AAIndsM, 'Lat': latitude, 'Lon': longitude, 'Depth': depth})

        for df_code in code.values():
            df_code['AAInds'] = df['AAInds']
        
        # Initialize dictionaries for AA and Else data
        AAdata = {}
        Elsedata = {}

        # Iterate over each key in code
        for i in code:
            # Extract data arrays from the DataFrame
            data_arrays = np.array([code[i][key].values for key in ['Depth', 'Latitude', 'Longitude', 'S', 'T', 'A', 'B', 'C', 
                                                                    'Order', 'Salinity_u', 'Temperature_u', 'Phosphate_u', 
                                                                    'Nitrate_u', 'Silicate_u', 'Oxygen_u', 'AAInds']])

            # Unpack arrays into separate variables
            depth, latitude, longitude, S, T, A, B, C, order, sal_u, temp_u, phos_u, nitr_u, sil_u, oxyg_u, aainds = data_arrays
    
            # Normalize the depth values
            depth = depth / 25
            # Reshape data arrays to match the number of rows
            NumRows_out = len(longitude)
            reshaped_data = [arr.reshape(NumRows_out, 1) for arr in [depth, latitude, longitude, S, T, A, B, C, order, sal_u, 
                                                                     temp_u, phos_u, nitr_u, sil_u, oxyg_u, aainds]]
            dep, lat, lon, sal, temp, avar, bvar, cvar, orde, salu, tempu, phosu, nitru, silu, oxygu, aai = reshaped_data
    
            # Combine the data into one array for further splitting
            InputBool = np.hstack(reshaped_data)
    
            # Define columns for the final DataFrame
            columns = ['d2d', 'Latitude', 'Longitude', 'S', 'T', 'A', 'B', 'C', 'Order', 'Salinity_u', 'Temperature_u', 
                       'Phosphate_u', 'Nitrate_u', 'Silicate_u', 'Oxygen_u', 'AAInds']
            NumCols_out = len(columns)

            # Function to filter arrays based on condition
            def split(arr, cond):
                return arr[cond]

            # Split the data into AA and Else data
            InputAA_01 = split(InputBool, InputBool[:, -1] == 1)
            InputElse_01 = split(InputBool, InputBool[:, -1] == 0)

            # Reshape and create DataFrames for AA and Else
            AAInput = pd.DataFrame(InputAA_01.reshape(len(InputAA_01), NumCols_out), columns=columns)
            ElseInput = pd.DataFrame(InputElse_01.reshape(len(InputElse_01), NumCols_out), columns=columns)

            # Store the results in the dictionaries
            AAdata[i] = AAInput
            Elsedata[i] = ElseInput

        # Use boolean for AA or Else to separate coefficients into Atlantic or not
        GridCoords, Cs, AAInds = LIR_data[:3]    
        DVs, CsVs = list(Cs.keys()), list(Cs.values())   
        ListVars = list(range(len(AAInds)))
        GridValues, AAIndValues = list(GridCoords.values())[0], list(AAInds.values())[0]
    
        lon_grid, lat_grid, d2d_grid, aainds = np.array((GridValues[0])), np.array((GridValues[1])), \
           np.array(GridValues[2])/25, np.array(AAIndValues[0])
        names = ['lon', 'lat', 'd2d', "C_alpha", "C_S", "C_T", "C_A", "C_B", "C_C", 'AAInds']
    
        Gdf, CsDesired = {}, {}   
        for l, name in zip(ListVars, DVs):
            Cs2 = CsVs[:][l][:]
            for e in Equations:
                CsName = f'Cs{name}{e}'
                CsDesired[CsName] = Cs2[e-1][:]
                Cs3 = Cs2[e-1][:]
                C_alpha, C_S, C_T, C_A, C_B, C_C = np.array(Cs3[0]), np.array(Cs3[1]), np.array(Cs3[2]), np.array(Cs3[3]), \
                    np.array(Cs3[4]), np.array(Cs3[5])
                grid_indices = np.column_stack((lon_grid, lat_grid, d2d_grid, C_alpha, C_S, C_T, C_A, C_B, C_C, aainds))
                Gdf[f"{name}{e}"] = pd.DataFrame(data=grid_indices, columns=names)
                
        # Interpolate
        Gkeys, Gvalues = list(Gdf.keys()), list(Gdf.values())
        AAOvalues, ElseOvalues = list(AAdata.values()), list(Elsedata.values())

        def process_grid(grid_values, data_values):
            results = []
            for i in range(len(grid_values)):
                grid = grid_values[i]
                points = np.array([list(grid['lon']), list(grid['lat']), list(grid['d2d'])]).T
                tri = Delaunay(points)
        
                values = np.array([
                    list(grid['C_alpha']),
                    list(grid['C_S']),
                    list(grid['C_T']),
                    list(grid['C_A']),
                    list(grid['C_B']),
                    list(grid['C_C'])
                ]).T
                interpolant = scipy.interpolate.LinearNDInterpolator(tri, values) 
                        
                data = data_values[i]
                points_to_interpolate = (list(data['Longitude']), list(data['Latitude']), list(data['d2d']))
                results.append(interpolant(points_to_interpolate))

            return results, interpolant

        # Process AA and EL grids
        aaLCs, aaInterpolants_pre = process_grid(Gvalues, AAOvalues)
        elLCs, elInterpolants_pre = process_grid(Gvalues, ElseOvalues)   

        # Initialize lists for storing interpolated values
        aaIntCT2, aaIntCA2, aaIntCB2, aaIntCC2, aaTo2, aaAo2, aaBo2, aaCo2 = [[] for _ in range(8)]
        elIntCT2, elIntCA2, elIntCB2, elIntCC2, elTo2, elAo2, elBo2, elCo2 = [[] for _ in range(8)]
        aaInterpolants, elInterpolants = {}, {}  

        for i in range(0, len(aaLCs)): 
            aaIntalpha, elIntalpha = aaLCs[i][:, 0], elLCs[i][:, 0]
            aaIntCS, elIntCS = aaLCs[i][:, 1], elLCs[i][:, 1]
            aaIntCT, elIntCT = aaLCs[i][:, 2], elLCs[i][:, 2]
            aaIntCA, elIntCA = aaLCs[i][:, 3], elLCs[i][:, 3]
            aaIntCB, elIntCB = aaLCs[i][:, 4], elLCs[i][:, 4]
            aaIntCC, elIntCC = aaLCs[i][:, 5], elLCs[i][:, 5]

            # Handle missing data (NaN handling)
            def process_list(int_values, val_values):
                int2, val2 = [], []
                for item, val in zip(int_values, val_values):
                    int2.append(0 if pd.isna(item) else item)
                    val2.append(0 if val == "nan" else val)
                return int2, val2
            # Reprocessing "NaN" to 0 as needed for calculations

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

            # Compute `aaEst`
            aaEst = np.array([a + b*c + d*e + f*g + h*i + j*k 
                              for a, b, c, d, e, f, g, h, i, j, k 
                              in zip(aaIal, aaICS, aaS, aaICT, aaT, aaICA, aaA, aaICB, aaB, aaICC, aaC)])
        
            # Store results
            aaInterpolants[key] = (aaIal, aaICS, aaICT, aaICA, aaICB, aaICC, aaEst)

            # Reprocessing "NaN" to 0 as needed for calculations
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

            # Compute 'elEst'
            elEst = np.array([a + b*c + d*e + f*g + h*i + j*k
                              for a, b, c, d, e, f, g, h, i, j, k
                              in zip(elIal, elICS, elS, elICT, elT, elICA, elA, elICB, elB, elICC, elC)])

            # Store the results
            elInterpolants[key] = (elIal, elICS, elICT, elICA, elICB, elICC, elEst)

        Estimate, Coefficients, CoefficientsUsed = {}, {}, {}
        for kcombo in AAdata.keys():
            AAdata[kcombo]["C0"] = aaInterpolants[kcombo][0]
            AAdata[kcombo]["CS"] = aaInterpolants[kcombo][1]
            AAdata[kcombo]["CT"] = aaInterpolants[kcombo][2]
            AAdata[kcombo]["CA"] = aaInterpolants[kcombo][3]
            AAdata[kcombo]["CB"] = aaInterpolants[kcombo][4]
            AAdata[kcombo]["CC"] = aaInterpolants[kcombo][5]
            AAdata[kcombo]["Estimate"] = aaInterpolants[kcombo][6]
            Elsedata[kcombo]["C0"] = elInterpolants[kcombo][0]
            Elsedata[kcombo]["CS"] = elInterpolants[kcombo][1]
            Elsedata[kcombo]["CT"] = elInterpolants[kcombo][2]
            Elsedata[kcombo]["CA"] = elInterpolants[kcombo][3]
            Elsedata[kcombo]["CB"] = elInterpolants[kcombo][4]
            Elsedata[kcombo]["CC"] = elInterpolants[kcombo][5]
            Elsedata[kcombo]["Estimate"] = elInterpolants[kcombo][6]

            # Combine, sort, and extract TotData
            TotData = pd.concat([AAdata[kcombo], Elsedata[kcombo]]).sort_values(by=["Order"])
            Coefficients[kcombo] = TotData
            Estimate[kcombo] = TotData["Estimate"]
    
            # Extract coefficients into a DataFrame
            coefnames = ["Intercept", "Coef S", "Coef T", "Coef A", "Coef B", "Coef C"]
            coefdata = [TotData[col].values for col in ["C0", "CS", "CT", "CA", "CB", "CC"]]
            CoefficientsUsed[kcombo] = pd.DataFrame(np.array(coefdata).T, columns=coefnames)

        # Estimating EMLR
        def EMLR_Estimate(Equations, DesiredVariables, OutputCoordinates={}, PredictorMeasurements={}, UDict={}, DUDict={}, Coefficients={}, **kwargs):
            EMLR, varnames, EqM = {}, [], []
            
            for dv in DesiredVariables:
                # Fetch LIR data and process into grid arrays
                LIR_data = fetch_data([dv])
                grid_names = ['UDepth', 'USal', 'Eqn', 'RMSE']
                UGridArray = pd.DataFrame(
                    [np.nan_to_num([LIR_data[3][i][c][b][a] for a in range(16) for b in range(11) for c in range(8)]) for i in range(4)]
                ).T
                UGridArray.columns = grid_names
                UGridPoints, UGridValues = (UGridArray['UDepth'], UGridArray['USal'], UGridArray['Eqn']), UGridArray['RMSE']
        
                for eq in range(len(Equations)):
                    varnames.append(dv + str(Equations[eq]))
                    EM = []
                    eq_str = str(Equations[eq])
                    eq_repeated = [Equations[eq]] * len(PredictorMeasurements['salinity'])
                    UGridPointsOut = (OutputCoordinates['depth'], PredictorMeasurements['salinity'], eq_repeated)
                    emlr = griddata(UGridPoints, UGridValues, UGridPointsOut, method='linear')
                    combo = f"{dv}{eq_str}"
                    Coefs = {k: np.nan_to_num(np.array(Coefficients[combo][k])) for k in ["C0", "CS", "CT", "CA", "CB", "CC"]}
                    uncdfs, duncdfs = UDict[combo], DUDict[combo]
                    keys = uncdfs.columns.to_numpy()
                    USu, UTu, UAu, UBu, UCu = [np.nan_to_num(uncdfs[key].fillna(0).astype(float)) for key in keys]
                    DUSu, DUTu, DUAu, DUBu, DUCu = [np.nan_to_num(duncdfs[key].fillna(0).astype(float)) for key in keys]
                    USu2, UTu2, UAu2, UBu2, UCu2 = [np.nan_to_num(uncdfs[key].fillna(-9999).astype(float)) for key in keys]
                    DUSu2, DUTu2, DUAu2, DUBu2, DUCu2 = [np.nan_to_num(duncdfs[key].fillna(-9999).astype(float)) for key in keys]
                               
                    for cucombo in range(len(Coefs["CS"])):
    
                        s1 = (Coefs["CS"][cucombo]*USu[cucombo])**2
                        t1 = (Coefs["CT"][cucombo]*UTu[cucombo])**2
                        a1 = (Coefs["CA"][cucombo]*UAu[cucombo])**2
                        b1 = (Coefs["CB"][cucombo]*UBu[cucombo])**2
                        c1 = (Coefs["CC"][cucombo]*UCu[cucombo])**2
                        sum2 = s1+t1+a1+b1+c1
                        ds1 = (Coefs["CS"][cucombo]*DUSu[cucombo])**2
                        dt1 = (Coefs["CT"][cucombo]*DUTu[cucombo])**2
                        da1 = (Coefs["CA"][cucombo]*DUAu[cucombo])**2
                        db1 = (Coefs["CB"][cucombo]*DUBu[cucombo])**2
                        dc1 = (Coefs["CC"][cucombo]*DUCu[cucombo])**2
                        dsum2 = ds1+dt1+da1+db1+dc1
                        
                        uncestimate = (sum2 - dsum2 + emlr[cucombo]**2)**0.5
                        EM.append(uncestimate)            
                    EqM.append(EM)
            
            EqM2 = []
            for i in EqM:
                UncertEst = np.array(i)
                UncertEst = UncertEst.astype('float')
                UncertEst[USu2==-9999]=['nan']
                if Equations[eq] == 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8:
                    UncertEst[UTu2==-9999]=['nan']
                if Equations[eq] == 1 | 2 | 5 | 6 | 9 | 10 | 13 | 14:
                    UncertEst[UAu2==-9999]=['nan']
                if Equations[eq] == 1 | 3 | 5 | 7 | 9 | 11 | 13 | 15:
                    UncertEst[UBu2==-9999]=['nan']
                if Equations[eq] == 1 | 2 | 3 | 4 | 9 | 10 | 11 | 12:
                    UncertEst[UCu2==-9999]=['nan']
                EqM2.append(UncertEst)
            for key in range(0, len(varnames)):
                EMLR[varnames[key]] = EqM2[key]
                EMLR = pd.DataFrame(EMLR)
                
            return EMLR

        Uncerts = EMLR_Estimate(Equations, DesiredVariables, OutputCoordinates, PredictorMeasurements, unc_combo_dict, dunc_combo_dict, Coefficients=Coefficients)
        return Estimate, Uncerts, CoefficientsUsed # PredictorMeasurements: Dictionary, Estimate: Dictionary, 

    # Uncertainties: pd DataFrame, DUncertainties: pd DataFrame, EMLR:list
    Est_pre, Uncertainties, CoefficientsUsed = preprocess_interpolate(
        DesiredVariables, 
        Equations, 
        EstDates, 
        VerboseTF, 
        C, 
        PredictorMeasurements, 
        Uncertainties_pre, 
        DUncertainties_pre
    )
    YouHaveBeenWarnedCanth = False
        
    def SimpleCantEstimateLR(EstDates, longitude, latitude, depth):
        # Load interpolation points and values
        CantIntPoints = pd.read_csv('SimpleCantEstimateLR_full.csv')
        pointsi = (
            CantIntPoints['Int_long'] * 0.25,
            CantIntPoints['Int_lat'],
            CantIntPoints['Int_depth'] * 0.025,
        )
        values = CantIntPoints['values']

        # Scale input coordinates
        pointso = (
            np.array(longitude) * 0.25,
            np.array(latitude),
            np.array(depth) * 0.025,
        )

        # Interpolate and compute Cant2002
        Cant2002 = griddata(pointsi, values, pointso, method='linear')

        # Adjust for estimation dates
        CantMeas = [
            c * math.exp(0.018989 * (date - 2002)) for c, date in zip(Cant2002, EstDates)
        ]

        return CantMeas, Cant2002
   
    Cant_adjusted = {}
    combos2 = list(Est_pre.keys())
    values2 = list(Est_pre.values())

    if "EstDates" in kwargs and ("DIC" in DesiredVariables or "pH" in DesiredVariables):      
        if not YouHaveBeenWarnedCanth:
            if VerboseTF:
                print("Estimating anthropogenic carbon for PyESPER_LIR.")
            longitude = np.mod(OutputCoordinates["longitude"], 360)
            latitude = np.array(OutputCoordinates["latitude"])
            depth = np.array(OutputCoordinates["depth"])
            Cant, Cant2002 = SimpleCantEstimateLR(EstDates, longitude, latitude, depth)
            YouHaveBeenWarnedCanth = True
        for combo, a in zip(combos2, values2):
            dic = []
            if combo.startswith("DIC"):
                for vala, Canta, Cant2002a in zip(a, Cant, Cant2002):
                    if math.isnan(vala): 
                        dic.append("nan")
                    else:
                        dic.append(vala + Canta - Cant2002a)
            else:
                dic = list(a)
            Cant_adjusted[combo] = dic
                 
        if "pH" in DesiredVariables:
            print('pH is detected')
            warning = []
            for combo, values in zip(combos2, values2):
                if combo.startswith("pH"):
                    salinity = PredictorMeasurements["salinity"]
                    PM_pH = {'salinity': salinity}
                    eq = [16]
                    alkest, _, _ = preprocess_interpolate(
                        ["TA"], eq, EstDates, ['False'], C, PM_pH, Uncertainties_pre, DUncertainties_pre
                    )
                    EstAlk = np.array(alkest["TA16"])
                    EstSi = EstP = [0] * len(EstAlk)
                    Pressure = sw.pres(OutputCoordinates["depth"], OutputCoordinates["latitude"])
                    Est = np.array(values)

                    # CO2SYS calculations
                    Out = pyco2.sys(
                        par1=EstAlk, par2=Est, par1_type=1, par2_type=3, salinity=salinity,
                        temperature=PredictorMeasurements["temperature"], temperature_out=PredictorMeasurements["temperature"],
                        pressure=Pressure, pressure_out=Pressure, total_silicate=EstSi, total_phosphate=EstP, opt_total_borate=2
                    )
                    DICadj = Out["dic"] + Cant - Cant2002
                    OutAdj = pyco2.sys(
                        par1=EstAlk, par2=DICadj, par1_type=1, par2_type=2, salinity=salinity,
                        temperature=PredictorMeasurements["temperature"], temperature_out=PredictorMeasurements["temperature"],
                        pressure=Pressure, pressure_out=Pressure, total_silicate=EstSi, total_phosphate=EstP, opt_total_borate=2
                    )
                    pHadj = OutAdj["pH"]

                    # Check for convergence warnings
                    if any(np.isnan(pHadj)):
                        warning_message = (
                            "Warning: CO2SYS took >20 iterations to converge. The corresponding estimate(s) will be NaN. "
                            "This typically happens when ESPER_LIR is poorly suited for estimating water with the given properties "
                            "(e.g., very high or low salinity or estimates in marginal seas)."
                        )
                        warning.append(warning_message)
                else:
                    pHadj = np.array(values)

                Cant_adjusted[combo] = pHadj.tolist()

            # Print warnings if any
            if warning:
                print(warning[0])
                
    elif "EstDates" not in kwargs and ("DIC" or "pH" in DesiredVariables) and VerboseTF and not YouHaveBeenWarnedCanth:
        print("Warning: DIC or pH is a requested output but the user did not provide dates for the desired estimates.  The estimates "
              "will be specific to 2002.0 unless the optional EstDates input is provided (recommended).")
        YouHaveBeenWarnedCanth = True

    if kwargs.get("pHCalcTF") and "pH" in DesiredVariables:
        if VerboseTF:
            print("Recalculating the pH to be appropriate for pH values calculated from TA and DIC.")
        for combo in range(0, len(combos2)):
            if combos2[combo].startswith("pH"):
                pH_adjcalc_Est = []
                pH_adjcalc = values2[combo]
                for v in pH_adjcalc:
                    pH_adjcalc_Est.append((pH_adjcalc[v]+0.3168)/1.0404)
            Cant_adjusted[combos2[combo]] = pH_adjcalc_Est

    combos3 = Cant_adjusted.keys()
    values3 = Cant_adjusted.values()

    Estimates = {}
    k2 = list(combos2)
    v2 = list(values2)
    k3 = list(combos3)
    v3 = list(values3)
    for keys2 in range(0, len(k2)):
        ar2 = np.array(v2[keys2])
        for keys3 in range(0, len(k3)):
            ar2[k2[keys2] == k3[keys3]] = v3[keys3]
            
        Estimates[k2[keys2]] = ar2
        Estimates = pd.DataFrame(Estimates)
       
    toc = time.perf_counter()
    print(f"PyESPER_LIR took {toc - tic:0.4f} seconds, or {(toc-tic)/60:0.4f} minutes to run")    
    
    return Estimates, Uncertainties, CoefficientsUsed
