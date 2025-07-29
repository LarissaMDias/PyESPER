def process_uncertainties(param, default_factor, MeasUncerts, PredictorMeasurements, n):
   
    """ 
  Helps proces uncertainties as needed by formatting any possible inputs, calculating
        outputs, and processing "nan" values

    Inputs:
        param: String name of uncertainty to be calculated
        default_factor: Scalar default uncertainty value
        MeasUncerts: Dictionary of measurement uncertainty values
        PredictorMeasurements: Dictionary of input measurements
        n: Scalar number of estimates

    Ouputs:
        result: Numpy array of user uncertainty-based uncertainties for each param
        dresult: Numpy array of default uncertainties for each param
    """
    
    import numpy as np

   # Determining whether the defined seawater property uncertainty has been used as input
    if param in MeasUncerts:
        # Obtain the predefined uncertainty if provided
        result = np.array(MeasUncerts.get(param))
        # Determining if unique uncertainties for each measurement were provided of if
        # only one uncertainty per property was given for all properties, and formatting
        # to the same length if so
        if len(result) < n:
            result = np.tile(result, n)
        # Formatting the naming convention of default uncertainties
        if param.replace('_u', '') in PredictorMeasurements:
            dresult = np.array([i * default_factor for i in PredictorMeasurements[param.replace('_u', '')]])
        else:
            dresult = result
    # Giving defaults in the case where user-provided uncertainties were not provided
    else:
        if param.replace('_u', '') in PredictorMeasurements:
            result = np.array([i * default_factor for i in PredictorMeasurements[param.replace('_u', '')]])
            dresult = result
        else:
            result = np.tile('nan', n)
            dresult = np.tile(0, n)
    return result, dresult

def measurement_uncertainty_defaults(n, PredictorMeasurements={}, MeasUncerts={}):
    
    """
    Inputs:
        n: Scalar number of estimates requested
        PredictorMeasurements: Dictionary of predictor measurements used for analyses
        MeasUncerts: User-provided measurement uncertainties or empty dictionary, if not provided
            
    Outputs:
        Uncertainties_pre: Dictionary of user-provided measurement uncertainties
        DUncertainties_pre: Dictionary of default measurement uncertainties
    """
    
    import numpy as np

    Uncertainties_pre, DUncertainties_pre = {}, {}
            
    # User-input salinity measurement uncertainties
    sal_u = np.array(MeasUncerts.get("sal_u", [0.003]))
    sal_u = np.tile(sal_u, n) if len(sal_u) < n else sal_u
    # Default salinity measurement uncertainties
    sal_defu = np.tile(0.003, n)
    
    # User-defined and default temperature measurement uncertainties
    temp_u = np.tile(np.array(MeasUncerts.get("temp_u", [0.003])), n) if "temp_u" in MeasUncerts or "temperature" in PredictorMeasurements else np.tile("nan", n)
    temp_defu = np.tile(0.003 if "temp_u" in MeasUncerts or "temperature" in PredictorMeasurements else 0, n)
    
    # Process other parameters
    parameters = {
        "phosphate_u": 0.02,
        "nitrate_u": 0.02,
        "silicate_u": 0.02,
        "oxygen_u": 0.01
    }
        
    # User process_uncertainties function to calculate defaults and user-defined uncertainties for each
    # parameter in a dictionary
    for param, factor in parameters.items():
        Uncertainties_pre[param], DUncertainties_pre[param] = process_uncertainties(
            param,
            factor,
            MeasUncerts,
            PredictorMeasurements,
            n
        )
    
    # Update MeasUncerts and DefaultUAll dictionary keys to include salinity, temperature, and all other properties
    meas_uncerts_keys = ["sal_u", "temp_u", *parameters.keys()]
    
    # Populating the dictionaries
    Uncertainties_pre.update(dict(zip(meas_uncerts_keys, [sal_u, temp_u, *Uncertainties_pre.values()])))
    DUncertainties_pre.update(dict(zip(meas_uncerts_keys, [sal_defu, temp_defu, *DUncertainties_pre.values()])))
    
    return Uncertainties_pre, DUncertainties_pre

