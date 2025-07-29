def adjust_pH_DIC(DesiredVariables, VerboseTF, Dates, Est_pre={}, PredictorMeasurements={}, OutputCoordinates={}, **kwargs):

    """
    If present, adjusting pH and DIC for anthropogenic carbon (Cant) within LIRs. Cant adjustment methods
        are based on those from ESPERv1, which is a TTD-based assumption/simplification but does not
        use the Tracer-based Rapid Anthropogenic Carbon Estimation data product, TRACE. Rather,
        interpolation from a gridded product is used to produce estimates for the year 2002 and data is
        adjusted to/from this reference year. This is the first of three steps for Cant adjustment

    Inputs:
        DesiredVariables: List of desired variables to estimate
        VerboseTF: Boolean indicating whether the user wants suppression of warnings
        Dates: List of dates for estimates
        Est_pre: Dictionary of preliminary estimates for each variable-equation case scenario
        PredictorMeasurements: Dictionary of input measurements for each variable-equation case scenario
        OutputCoordinates: Dictionary of coordinates for locations of estimates
        **kwargs: Please see README for full informations

    Outputs:
        Cant_adjusted: Dictionary of values adjusted for anthropogenic carbon for each combination
        Cant: Numpy array of estimates for anthropogenic carbon for each estimate
        Cant2002: Numpy array of estimates for anthropogenic carbon in the year 2002 for each estimate
    """

    import numpy as np
    from PyESPER.simplecantestimatelr import simplecantestimatelr

    # Predefining output dictionary and formatting estimates
    Cant_adjusted={}
    combos2 = list(Est_pre.keys())
    values2 = []
    for c, v in Est_pre.items():
        vals = np.array([v])
        vals = vals.flatten()
        values2.append(vals)
    values2 = np.array(values2)

    # Predefining anthropogenic carbon numpy arrays
    n = len(Dates)
    Cant, Cant2002 = np.zeros(n), np.zeros(n)

    # Only proceed if adjustment is needed
    if "EstDates" in kwargs and ("DIC" in DesiredVariables or "pH" in DesiredVariables):
        if VerboseTF:
            print("Estimating anthropogenic carbon for PyESPER_LIR.")

        # Normalize longitude to [0, 360]
        longitude = np.mod(np.array(OutputCoordinates["longitude"]), 360)
        latitude = np.array(OutputCoordinates["latitude"])
        depth = np.array(OutputCoordinates["depth"])
    
        # Estimate anthropogenic carbon (Cant) and anthropogenic carbon for the year 2002 (Cant2002)
        Cant, Cant2002 = simplecantestimatelr(Dates, longitude, latitude, depth)
        Cant, Cant2002 = np.array(Cant), np.array(Cant2002)
    
        for combo in range(0, len(combos2)):
            comb = combos2[combo]
            val = values2[combo]
            est1 = []
        
            # Only adjust if combo is DIC
            if "dic" in comb.lower():
                adjusted = np.where(val == "nan", np.nan, val + Cant - Cant2002)
                est1.append(adjusted)
    
            if "dic" not in comb.lower():
                nanfix = np.where(val == "nan", np.nan, val)
                est1.append(nanfix)
        
            Cant_adjusted[combos2[combo]] = est1

    return Cant_adjusted, Cant, Cant2002 

