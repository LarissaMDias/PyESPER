def PyESPER_NN(DesiredVariables, Path, OutputCoordinates={}, PredictorMeasurements={}, **kwargs):

    """
    Python interpretation of ESPER_NNv1.1

    Empirical Seawater Property Estimation Routines: Estimates seawater properties and estimate uncertainty from combinations of other parameter
    measurements.  PYESPER_NN refers specifically to code that uses neural networks as opposed to collections of interpolated linear regressions
    (LIRs), and Python rather than MATLAB coding languages.  

    Reserved for version update notes: (no updates, first version)
  
    Documentation and citations:
    LIARv1: Carter et al., 2016, doi: 10.1002/lom3.10087
    LIARv2, LIPHR, LINR citation: Carter et al., 2018, https://doi.org/10.1002/lom3.10232
    LIPR, LISIR, LIOR, first described/used: Carter et al., 2021, https://doi.org/10.1029/2020GB006623
    LIRv3 and ESPER_NN (ESPERv1.1): Carter, 2021, https://10.5281/ZENODO.5512697

    PyESPER_NN is a Python replicate of ESPER_NN:
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
        Path directing Python to the location of saved/downloaded neural net files on the user's computer (ex: '/Users/lara/Documents/Python'). 

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
       PredictorMeasurements={"salinity":[35, 34.1, 32, 33], "temperature": [0.1, 10, 0.5, 2], "oxygen": [202.3, 214.7, 220.5, 224.2]} or 
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
        Dictionary of measurement uncertainties (see 'PredictorMeasurements' for units). Providing these estimates will improve PyESPER_NN
        estimate uncertainties. Measurement uncertainties are a small part of PyESPER_NN estimate uncertainties for WOCE-quality measurements. 
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
        Setting this to false will reduce the number of updates, warnings, and errors printed by PyESPER_NN. And additional step can be taken before 
        before executing the PyESPER_NN function (see below) that will further reduce updates, warnings, and errors, if desired. 
        
    *************************************************************************
    Outputs:
 
    Estimates: 
        A n by e pandas DataFrame of estimates specific to the coordinates and parameter measurements provided as inputs.  Units are micromoles  
        per kg (equivalent to the deprecated microeq per kg seawater). Column names are the unique desired variable-equation combinations requrested by 
        the user. 
	
     Uncertainties: 
        A n by e dictionary of uncertainty estimates specific to the coordinates, parameter measurements, and parameter uncertainties provided.
        Units are micromoles per kg (equivalent to the deprecated microeq per kg seawater). Column names are the unique desired variable-equation 
        combinations requrested by the user. 
        
    *************************************************************************
    Missing data: should be indicated with a nan.  A nan coordinate will yield nan estimates for all equations at that coordinate.  A nan
    parameter value will yield NaN estimates for all equations that require that parameter.
 
    *************************************************************************
    Please send questions or related requests about PyESPER to lmdias@uw.edu.
    ************************************************************************* 
    """

    import importlib
    import math
    import os
    import time
    from statistics import mean

    import matplotlib.path as mpltPath
    import numpy as np
    import pandas as pd
    import PyCO2SYS as pyco2
    import seawater as sw
    from scipy.interpolate import griddata
    from scipy.io import loadmat
    
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

    # Checking the sanity of input values and printing warnings for erroneous input
    if "temperature" in PredictorMeasurements:
        if any(t < -5 or t > 50 for t in PredictorMeasurements["temperature"]):
               print("Warning: Temperatures less than -5 C or greater than 100 C have been found. PyESPER is not intended for seawater with these properties. Note that PyESPER expects temperatures in Centigrade.")
                
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

    # Checking kwargs for presence of VerboseTF and defining defaults as needed
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

    # Define or read in PerKgSwTF
    PerKgSwTF = kwargs.get("PerKgSwTF", True)

    # Reading in MeasUncerts, if user-defined uncertainties are present in kwargs. Interpreting these, or defining measurement and
    # default uncertainties, if absent from kwargs.
    MeasUncerts_processed, DefaultUAll = {}, {}
    if "MeasUncerts" in kwargs:  
        MeasUncerts=kwargs.get("MeasUncerts")
        if max(len(v) for v in MeasUncerts.values()) != n:
            if min(len(v) for v in MeasUncerts.values()) != 1: 
                raise CustomError("MeasUncerts must be undefined, a vector with the same number of elements as \
                PredictorMeasurements has columns, or a matrix of identical dimension to PredictorMeasurements.")
        if len(MeasUncerts) != len(PredictorMeasurements):
            print("Warning: There are different numbers of columns of input uncertainties and input measurements.")
               
         # Default salinity uncertainties
        sal_u = np.array(MeasUncerts.get("sal_u", [0.003]))
        sal_u = np.tile(sal_u, n) if len(sal_u) < n else sal_u
        sal_defu = np.tile(0.003, n)
        
        temp_u = np.array(MeasUncerts.get("temp_u", [0.003])) if "temp_u" in MeasUncerts or "temperature" in PredictorMeasurements else np.tile("nan", n)
        temp_u = np.tile(temp_u, n) if len(temp_u) < n else temp_u
        temp_defu = np.tile(0.003, n) if "temp_u" in MeasUncerts or "temperature" in PredictorMeasurements else np.tile(0, n)

        def process_uncert(param, default_factor):
            if f"{param}_u" in MeasUncerts:
                result = np.array(MeasUncerts[f"{param}_u"])
                result = np.tile(result, n) if len(result) < n else result
            else:
                result = (
                    np.array([i * default_factor for i in PredictorMeasurements[param]])
                    if param in PredictorMeasurements else np.tile("nan", n)
                )
            dresult = result if param not in PredictorMeasurements else np.array([i * default_factor for i in PredictorMeasurements[param]])
            MeasUncerts_processed[f"{param}_u"] = result
            DefaultUAll[f"{param}_defu"] = dresult

        # Process all parameters with their respective default factors
        for param, default_factor in [("phosphate", 0.02), ("nitrate", 0.02), ("silicate", 0.02), ("oxygen", 0.01)]:
            process_uncert(param, default_factor)

        # Extract processed uncertainties
        phosphate_u, phosphate_defu = np.array(MeasUncerts_processed["phosphate_u"]), np.array(DefaultUAll["phosphate_defu"])
        nitrate_u, nitrate_defu = np.array(MeasUncerts_processed["nitrate_u"]), np.array(DefaultUAll["nitrate_defu"])
        silicate_u, silicate_defu = np.array(MeasUncerts_processed["silicate_u"]), np.array(DefaultUAll["silicate_defu"])
        oxygen_u, oxygen_defu = np.array(MeasUncerts_processed["oxygen_u"]), np.array(DefaultUAll["oxygen_defu"])
                
    else:
        MeasUncerts = {}
        sal_u = sal_defu = np.tile(0.003, n)

        # Helper function to set uncertainties and default uncertainties
        def default_uncert(param, factor):
            if param in PredictorMeasurements:
                result = np.array([i * factor for i in PredictorMeasurements[param]])
            else:
                result = np.tile("nan", n)
            return result

        temp_u, temp_defu = (np.tile(0.003, n), np.tile(0.003, n)) if "temperature" in PredictorMeasurements else (np.tile("nan", n), np.tile(0, n))

        phosphate_u = phosphate_defu = default_uncert("phosphate", 0.02)
        nitrate_u = nitrate_defu = default_uncert("nitrate", 0.02)
        silicate_u = silicate_defu = default_uncert("silicate", 0.02)
        oxygen_u = oxygen_defu = default_uncert("oxygen", 0.01)
            
    # Define the keys and corresponding variables for MeasUncerts
    meas_uncerts_keys = ["sal_u", "temp_u", "phosphate_u", "nitrate_u", "silicate_u", "oxygen_u"]
    meas_uncerts_values = [sal_u, temp_u, phosphate_u, nitrate_u, silicate_u, oxygen_u]

    # Update MeasUncerts using a dictionary comprehension
    MeasUncerts.update(dict(zip(meas_uncerts_keys, meas_uncerts_values)))

    # Define the keys and corresponding variables for DefaultUAll
    default_uall_keys = ["sal_defu", "temp_defu", "phosphate_defu", "nitrate_defu", "silicate_defu", "oxygen_defu"]
    default_uall_values = [sal_defu, temp_defu, phosphate_defu, nitrate_defu, silicate_defu, oxygen_defu]

    # Update DefaultUAll using a dictionary comprehension
    DefaultUAll.update(dict(zip(default_uall_keys, default_uall_values)))
       
    keys = ["sal_u", "temp_u", "phosphate_u", "nitrate_u", "silicate_u", "oxygen_u"]
    Uncerts = np.column_stack((sal_u, temp_u, phosphate_u, nitrate_u, silicate_u, oxygen_u))
    Uncertainties_pre = pd.DataFrame(Uncerts, columns=keys) 
    DUncerts = np.column_stack((sal_defu, temp_defu, phosphate_defu, nitrate_defu, silicate_defu, oxygen_defu))
    DUncertainties_pre = pd.DataFrame(DUncerts, columns=keys)

    # This function is the primary function of the PyESPER_NN, which preprocesses all data and applies the saved neural networks to the input data
    def preprocess_applynets(
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

        # Organizing data thus far
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
    
        if "temperature" in PredictorMeasurements:
            input_data["Temperature"] = PredictorMeasurements["temperature"]
        if "phosphate" in PredictorMeasurements:
            input_data["Phosphate"] = PredictorMeasurements["phosphate"]
        if "nitrate" in PredictorMeasurements:
            input_data["Nitrate"] = PredictorMeasurements["nitrate"]
        if "silicate" in PredictorMeasurements:
            input_data["Silicate"] = PredictorMeasurements["silicate"]
        if "oxygen" in PredictorMeasurements:
            input_data["Oxygen"] = PredictorMeasurements["oxygen"]

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
        if "phosphate" in PredictorMeasurements:
            phosphate_processed = np.array(PredictorMeasurements["phosphate"])
        else:
            phosphate_processed = np.tile("nan", n)
        if "nitrate" in PredictorMeasurements:
            nitrate_processed = np.array(PredictorMeasurements["nitrate"])
        else:
            nitrate_processed = np.tile("nan", n)
        if "silicate" in PredictorMeasurements:
            silicate_processed = np.array(PredictorMeasurements["silicate"])
        else: 
            silicate_processed = np.tile("nan", n)
    
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
            prodnptile = np.tile(product[p], (n, 1)).astype("str")

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
        
        common_columns = [
            "Order", "Dates", "Longitude", "Latitude", "Depth", 
            "Salinity_u", "Temperature_u", "Phosphate_u", 
            "Nitrate_u", "Silicate_u", "Oxygen_u"
        ]

        # Iterate over codenames and create DataFrames
        for n, cname in enumerate(codenames):
            data = np.column_stack([S[n], T[n], A[n], B[n], Z[n]])
            code[cname] = pd.DataFrame(data, columns=["S", "T", "A", "B", "C"])
            code[cname][common_columns] = InputAll[common_columns]
   
        # Loading the data       
        def fetch_data (DesiredVariables):
            for v in DesiredVariables:
                fname = os.path.join(Path, f"Uncertainty_Polys/NN_files_{v}_Unc_Poly.mat")
                NNs = loadmat(fname)
                Polys, UncGrid = NNs["Polys"][0][0], NNs["UncGrid"][0][0]
        
            NN_data = [Polys, UncGrid]
            return NN_data

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
        latitude, depth = np.array(C["latitude"]), np.array(C["depth"])
    
        # Check if coordinates are within each polygon
        conditions = [path.contains_points(np.column_stack((longitude, latitude))) for path in paths]

        # Combine conditions
        AAIndsM = np.logical_or.reduce(conditions)

        # Adding Bering Sea, S. Atl., and S. African Polygons separately 
        Bering = np.array([[173, 70], [210, 70], [210, 62.5], [173, 62.5], [173, 70]])
        beringpath = mpltPath.Path(Bering)
        beringconditions = beringpath.contains_points(np.column_stack((longitude, latitude)))
        SAtlInds, SoAfrInds = [], []
        for i, z in zip(longitude, latitude):
            # Check if the conditions are met for Southern Atlantic
            if (-34 > z > -44):  # Check latitude first to reduce unnecessary checks
                if i > 290 or i < 20:
                    SAtlInds.append('True')
                else:
                    SAtlInds.append('False')
        
                # Check if the condition is met for Southern Africa
                if 19 < i < 27:
                    SoAfrInds.append('True')
                else:
                    SoAfrInds.append('False')
            else:
                SAtlInds.append('False')
                SoAfrInds.append('False')
      
        # Create DataFrame
        df = pd.DataFrame({'AAInds': AAIndsM, 'BeringInds': beringconditions, 'SAtlInds': SAtlInds, \
                           'SoAfrInds': SoAfrInds, 'Lat': latitude, 'Lon': longitude, 'Depth': depth})

        # Running the neural networks
        combos = list(code.keys())
        combovalues = list(code.values())
        EstAtl, EstOther = {}, {}
        P, Sd, Td, Ad, Bd, Cd = {}, {}, {}, {}, {}, {} 

        for name, value in zip(combos, combovalues):
            cosd = np.cos(np.deg2rad(value["Longitude"] - 20)).tolist()
            sind = np.sin(np.deg2rad(value["Longitude"] - 20)).tolist()
            lat, depth = value["Latitude"].tolist(), value["Depth"].tolist()
            # Convert columns to lists of floats
            Sd[name] = value["S"].astype(float).tolist()
            Td[name] = value["T"].astype(float).tolist()
            Ad[name] = value["A"].astype(float).tolist()
            Bd[name] = value["B"].astype(float).tolist()
            Cd[name] = value["C"].astype(float).tolist()

        # Define a mapping from equations to the list of variables
        equation_map = {
            1: ["Sd", "Td", "Ad", "Bd", "Cd"],
            2: ["Sd", "Td", "Ad", "Cd"],
            3: ["Sd", "Td", "Bd", "Cd"],
            4: ["Sd", "Td", "Cd"],
            5: ["Sd", "Td", "Ad", "Bd"],
            6: ["Sd", "Td", "Ad"],
            7: ["Sd", "Td", "Bd"],
            8: ["Sd", "Td"],
            9: ["Sd", "Ad", "Bd", "Cd"],
            10: ["Sd", "Ad", "Cd"],
            11: ["Sd", "Bd", "Cd"],
            12: ["Sd", "Cd"],
            13: ["Sd", "Ad", "Bd"],
            14: ["Sd", "Ad"],
            15: ["Sd", "Bd"],
            16: ["Sd"]
        }

        # Create the correct vector for each equation case
        for v in DesiredVariables:
            for e in Equations:
                name = v + str(e)
                # Get the corresponding variables for the equation
                variables = [locals()[var][name] for var in equation_map[e]]
                P[name] = [[[cosd, sind, lat, depth] + variables]]
                netstimateAtl, netstimateOther = [], []
                for n in range(1, 5):
                    fOName = f"NeuralNetworks.ESPER_{v}_{e}_Other_{n}"
                    fAName = f"NeuralNetworks.ESPER_{v}_{e}_Atl_{n}"
                    moda = importlib.import_module(fAName)
                    modo = importlib.import_module(fOName)
                    from importlib import reload
                    reload(moda)
                    reload(modo)

                    netstimateAtl.append(moda.PyESPER_NN(P[name]))
                    netstimateOther.append(modo.PyESPER_NN(P[name]))
            
                # Process estimates for Atlantic and Other regions
                EstAtlL = [[netstimateAtl[na][0][eatl] for na in range(4)] for eatl in range(len(netstimateAtl[0][0]))]
                EstOtherL = [[netstimateOther[no][0][eoth] for no in range(4)] for eoth in range(len(netstimateOther[0][0]))]

                # Store the results
                EstAtl[name] = EstAtlL
                EstOther[name] = EstOtherL
    
        def process_estimates(estimates):
            keys = list(estimates.keys())
            values = list(estimates.values())
            result = {}
            for i, key in enumerate(keys):
                result[key] = [mean(values[i][v]) for v in range(len(values[0]))]
            return result

        Esta = process_estimates(EstAtl)
        Esto = process_estimates(EstOther)

        # Processing regionally in the Atlantic and Bering
        EstA, EstB, EB2, ESat, ESat2, ESaf, Estimate  = {}, {}, {}, {}, {}, {}, {}

        for i in code:
            code[i]["AAInds"] = df["AAInds"]
            code[i]["BeringInds"] = df["BeringInds"]
            code[i]["SAtlInds"] = df["SAtlInds"]
            code[i]["SoAfrInds"] = df["SoAfrInds"]
        
        for codename, codedata in code.items():
            Estatl, Estb, eb2, Estsat, esafr = [], [], [], [], []
            aainds, beringinds, satlinds, latitude, safrinds = (
                codedata[key] for key in ["AAInds", "BeringInds", "SAtlInds", "Latitude", "SoAfrInds"]
            )

            Estatl = [Esta[codename][i] if aa_ind else Esto[codename][i] for i, aa_ind in enumerate(aainds)]

            for l in range(0, len(Estatl)):
                repeated_values = (latitude[l]-62.5)/7.5
                B = np.tile(repeated_values, (1, len(Equations)))
                C = Esta[codename][l]
                B1 = C * B
                repeated_values2 = (70-latitude[l])/7.5
                D = np.tile(repeated_values2, (1, len(Equations)))
                E = Esto[codename][l]
                B2 = E * D
                Estb.append(B1[0][0] + B2[0][0])

            eb2 = [Estb[j] if b_ind else Estatl[j] for j, b_ind in enumerate(beringinds)]

            for n in range(0, len(satlinds)): 
                repeated_values = (latitude[n]+44)/10
                F1 = Esta[codename][n]
                F = np.tile(repeated_values, (1, len(Equations)))
                G1 = F1 * F
                repeated_values2 = (-34-latitude[n])/10
                H1 = Esto[codename][n]
                H = np.tile(repeated_values2, (1, len(Equations)))
                G2 = H1 * H
                Estsat.append(G1[0][0] + G2[0][0])
    
            EstA[codename], EstB[codename], EB2[codename], ESat[codename] = Estatl, Estb, eb2, Estsat

        # Regional processing for S. Atlantic
            ESat2[codename] = [
                ESat[codename][i] if satlinds[i] == "True" else EB2[codename][i]
                for i in range(len(satlinds))
            ]
            
        # Regional processing for S. Africa
            for s in range(0, len(safrinds)):
                repeated_values = (27-longitude[s])/8
                F1 = ESat2[codename][s]
                F = np.tile(repeated_values, (1, len(Equations)))
                G1 = F1*F
                repeated_values2 = (longitude[s]-19)/8
                H1 = Esto[codename][s]
                H = np.tile(repeated_values2, (1, len(Equations)))
                G2 = H1 * H
                esafr.append(G1[0][0] + G2[0][0])

            ESaf[codename] = esafr

            Estimate[codename] = [
                ESaf[codename][i] if safrinds[i] == "True" else ESat2[codename][i]
                for i in range(len(safrinds))
            ]

        # Bookkeeping blanks back to NaN as needed
        Estimate = {k: ('NaN' if v == '' else v) for k, v in Estimate.items()}
        no_equations = len(Equations)

        # Estimating EMLR
        def EMLR_Estimate(DesiredVariables, OutputCoordinates={}, PredictorMeasurements={}, **kwargs):
           
            EMLR = []
            for dv in DesiredVariables:
                NN_data = fetch_data([dv])

                data_arrays = [
                    np.nan_to_num(np.array([
                        NN_data[1][i][c][b][a]
                        for a in range(16)
                        for b in range(11)
                        for c in range(8)
                    ]))
                    for i in range(4)
                ]
        
                # Create DataFrame with meaningful column names
                UGridArray = pd.DataFrame({
                    'UDepth': data_arrays[0],
                    'USal': data_arrays[1],
                    'Eqn': data_arrays[2],
                    'RMSE': data_arrays[3],
                })
        
                UGridPoints = (UGridArray['UDepth'], UGridArray['USal'], UGridArray['Eqn'])
                UGridValues = UGridArray['RMSE']
        
                # Perform estimation for each equation
                EM = [
                    griddata(
                        UGridPoints, UGridValues,
                        (OutputCoordinates['depth'], PredictorMeasurements['salinity'], [Equations[eq]] * len(PredictorMeasurements['salinity'])),
                        method='linear'
                    )
                    for eq in range(no_equations)
                ]
        
                EMLR.append(EM)
    
            return EMLR

        EMLR = EMLR_Estimate(DesiredVariables, OutputCoordinates, PredictorMeasurements)
        
        return PredictorMeasurements, Estimate, Uncertainties, DUncertainties, EMLR # PredictorMeasurements: Dictionary, Estimate: Dictionary, 
    
    # Uncertainties: pd DataFrame, DUncertainties: pd DataFrame, EMLR:list
    PD_final, DPD_final, Unc_final, DUnc_final = [], [], [], []
    PMs_pre, Est_pre, U_pre, DU_pre, EMLR = preprocess_applynets(
        DesiredVariables, 
        Equations, 
        EstDates,
        VerboseTF,
        C, 
        PredictorMeasurements,
        Uncertainties_pre,
        DUncertainties_pre
    )

    for d, var in enumerate(DesiredVariables):
        Pertdv, DPertdv, Unc, DUnc = [], [], [], []
        var = [var]  # Wrap single variable in a list
        keys = ["sal_u", "temp_u", "phosphate_u", "nitrate_u", "silicate_u", "oxygen_u"]

        PredictorMeasurements2, Est, Uncertainties, DUncertainties, emlr = preprocess_applynets(
            var, 
            Equations, 
            EstDates, 
            ['False'], 
            OutputCoordinates, 
            PredictorMeasurements, 
            Uncertainties_pre, 
            DUncertainties_pre
        )
        
        names = list(PredictorMeasurements2.keys())
        PMs = list(PredictorMeasurements2.values())

        # Replace "nan" with 0 in PMs using list comprehensions
        PMs_nonan = [[0 if val == "nan" else val for val in pm] for pm in PMs]

        # Transpose PMs_nonan
        PMs = np.transpose(PMs_nonan)

        PMs3, DMs3 = {}, {}

        for pred in range(len(PredictorMeasurements2)):
            num_coords = len(OutputCoordinates['longitude'])
            num_preds = len(PredictorMeasurements2)

            # Initialize perturbation arrays
            Pert = np.zeros((num_coords, num_preds))
            DefaultPert = np.zeros((num_coords, num_preds))

            # Populate perturbation arrays
            Pert[:, pred] = Uncertainties[keys[pred]]
            DefaultPert[:, pred] = DUncertainties[keys[pred]]

            # Apply perturbations
            PMs2 = PMs + Pert
            DMs2 = PMs + DefaultPert

           # Update PMs3 and DMs3 dictionaries
            for col, name in enumerate(names):
                PMs3[name] = PMs2[:, col].tolist()
                DMs3[name] = DMs2[:, col].tolist()

            # Run preprocess_applynets for perturbed and default data
            VTF = False
            _, PertEst, _, _, _ = preprocess_applynets(
                var, Equations, EstDates, VTF, OutputCoordinates, PMs3, Uncertainties_pre, DUncertainties_pre
            )
            _, DefaultPertEst, _, _, _ = preprocess_applynets(
                var, Equations, EstDates, VTF, OutputCoordinates, DMs3, Uncertainties_pre, DUncertainties_pre
            )

            # Extract estimates and perturbation results
            combo, estimates = list(Est.keys()), list(Est.values())
            pertests, defaultpertests = list(PertEst.values()), list(DefaultPertEst.values())

            # Initialize result lists
            PertDiff, DefaultPertDiff, Unc_sub2, DUnc_sub2 = [], [], [], []
        
            for c in range(len(Equations)):
                # Compute differences and squared differences using list comprehensions
                PD = [estimates[c][e] - pertests[c][e] for e in range(len(estimates[c]))]
                DPD = [estimates[c][e] - defaultpertests[c][e] for e in range(len(estimates[c]))]
                Unc_sub1 = [(estimates[c][e] - pertests[c][e])**2 for e in range(len(estimates[c]))]
                DUnc_sub1 = [(estimates[c][e] - defaultpertests[c][e])**2 for e in range(len(estimates[c]))]

                # Append results to their respective lists
                PertDiff.append(PD)
                DefaultPertDiff.append(DPD)
                Unc_sub2.append(Unc_sub1)
                DUnc_sub2.append(DUnc_sub1)
            Pertdv.append(PertDiff)
            DPertdv.append(DefaultPertDiff)
            Unc.append(Unc_sub2)
            DUnc.append(DUnc_sub2)
        PD_final.append(Pertdv)
        DPD_final.append(DPertdv)
        Unc_final.append(Unc)
        DUnc_final.append(DUnc) # CHECK THIS WHOle shebang next

    est = list(Est_pre.values())
    Uncertainties = []
    propu = []
    for dv in range(0, len(DesiredVariables)):
        dvu = []
        for eq in range(0, len(Equations)):
            sumu = []
            for n in range(0, len(est[0])):
                u, du = [], []
                for pre in range(0, len(PredictorMeasurements)):
                    u.append(Unc_final[dv][pre][eq][n])
                    du.append(DUnc_final[dv][pre][eq][n])
                eu = EMLR[dv][eq][n]
                sumu.append((sum(u) - sum(du) + eu**2)**(1/2))
            dvu.append(sumu)
        propu.append(dvu)
    Uncertainties.append(propu)
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
                print("Estimating anthropogenic carbon for PyESPER_NN.")
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
            warning = []
            for combo, values in zip(combos2, values2):
                if combo.startswith("pH"):
                    salinity = PredictorMeasurements["salinity"]
                    PM_pH = {'salinity': salinity}
                    eq = [16]
                    alkpm, alkest, _, _, _ = preprocess_applynets(
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
                            "This typically happens when ESPER_NN is poorly suited for estimating water with the given properties "
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
        print("Warning: DIC or pH is a requested output but the user did not provide dates for the desired estimates.  The estimates will be specific to 2002.0 unless the optional EstDates input is provided (recommended).")
        YouHaveBeenWarnedCanth = True

    if kwargs.get("pHCalcTF") and "pH" in DesiredVariables:
        if VerboseTF:
            print("Recalculating the pH to be appropriate for pH values calculated from TA and DIC.")
        for combo, pH_values in zip(combos2, values2):
            if combo.startswith("pH"):
                pH_adjcalc_Est = [(pH + 0.3168) / 1.0404 for pH in pH_values]
                Cant_adjusted[combo] = pH_adjcalc_Est

    # Prepare data for processing
    combos3 = Cant_adjusted.keys()
    values3 = Cant_adjusted.values()
    Us = Uncertainties[0]
    Us2 = [u2 for u in Us for u2 in u]

    # Convert combos and values to lists for iteration
    k2, v2 = list(combos2), list(values2)
    k3, v3 = list(combos3), list(values3)

    # Initialize estimates and uncertainties dictionaries
    Estimates, Uncerts = {}, {}

    for key2, value2 in zip(k2, v2):
        # Adjust values in v2 based on matches in k3
        adjusted_array = np.array(value2)
        for key3, value3 in zip(k3, v3):
            adjusted_array[key2 == key3] = value3

        # Store adjusted values and uncertainties
        Estimates[key2] = adjusted_array
        Uncerts[key2] = Us2[k2.index(key2)]

    # Convert results to DataFrame
    Estimates = pd.DataFrame(Estimates)
    Uncertainties = pd.DataFrame(Uncerts)

    toc = time.perf_counter()
    print(f"PyESPER_NN took {toc - tic:0.4f} seconds, or {(toc-tic)/60:0.4f} minutes to run")    

    return Estimates, Uncertainties
