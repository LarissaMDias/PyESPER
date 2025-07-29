def lir(DesiredVariables, Path, OutputCoordinates={}, PredictorMeasurements={}, **kwargs):
    
    """
    Locally Interpolated Regressions (LIRs) for Empirical Seawater Property Estimation
    Runs all associated functions that calculate seawater properties from geographic coordinates,
        salinity, and optional other input variables using interpolation methods. Please refer
        to the README for more information.

    Inputs:
        DesiredVariables: List of desired output variables
        Path: Optional change of path location relative to user computer paths
        OutputCoordinates: Dictionary of "latitude", "longitude", and "depth" location outputs
        PredictorMeasurements: Dictionary of "salinity" and other optional predictor measurements
        **kwargs include MeasUncerts, EstDates, Equations, PerKgSwTF, VerboseTF - see README for
            full explanations

    Outputs:
        Estimates: Dictionary of estimates for each desired variable - equation combination
        Coefficients: Dictionary of dictionaries of coefficients for each combination
        Uncertainties: Dictionary of uncertainties for each combination
    """

    import time
    from PyESPER.errors import errors
    from PyESPER.defaults import defaults
    from PyESPER.lir_uncertainties import measurement_uncertainty_defaults
    from PyESPER.inputdata_organize import inputdata_organize
    from PyESPER.temperature_define import temperature_define
    from PyESPER.iterations import iterations
    from PyESPER.fetch_data import fetch_data
    from PyESPER.input_AAinds import input_AAinds
    from PyESPER.coefs_AAinds import coefs_AAinds
    from PyESPER.interpolate import interpolate
    from PyESPER.organize_data import organize_data
    from PyESPER.emlr_estimate import emlr_estimate
    from PyESPER.adjust_pH_DIC import adjust_pH_DIC
    from PyESPER.pH_adjustment import pH_adjustment
    from PyESPER.pH_adjcalc import pH_adjcalc
    from PyESPER.final_formatting import final_formatting

    # Starting the timer
    tic = time.perf_counter() 
    
    # Providing custom error messages for erroneous input
    errors(OutputCoordinates, PredictorMeasurements)

    # Setting defaults for various input parameters, including defining kwargs and
    # ensuring that coordinates use the correct format
    Equations, n, VerboseTF, EstDates, C, PerKgSwTF, MeasUncerts = defaults(
        DesiredVariables, 
        PredictorMeasurements,
        OutputCoordinates, 
        **kwargs
    )
    
    # Processing the input values (Uncertainties_pre) and calculating default 
    # measurement uncertainties 
    Uncertainties_pre, DUncertainties_pre  = measurement_uncertainty_defaults(
        n, 
        PredictorMeasurements, 
        MeasUncerts
    )
    
    # Creating an updated dictionary of all input data
    InputAll  = inputdata_organize(
        EstDates, 
        C, 
        PredictorMeasurements, 
        Uncertainties_pre
    )

    # Defining temperature as needed
    PredictorMeasurements, InputAll = temperature_define(
        DesiredVariables,
        PredictorMeasurements,
        InputAll,
        **kwargs
    )

    # Performing iterations for equation-desired variable combinations;
    # pre-defines the correct input data for LIRs
    code, unc_combo_dict, dunc_combo_dict = iterations(
        DesiredVariables, 
        Equations, 
        PerKgSwTF,
        C,
        PredictorMeasurements, 
        InputAll,
        Uncertainties_pre,
        DUncertainties_pre
    )

    # Loading the pre-trained algorithm data
    LIR_data = fetch_data(
        DesiredVariables, 
        Path
    )

    # Separating user-defined coordinates into Atlantic and Arctic (AAdata)
    # or other regions (Elsedata)
    AAdata, Elsedata = input_AAinds(
        C, 
        code
    )

    # Separating ESPER pre-defined coefficients into Atlantic and Arctic or other regions
    Gdf, CsDesired = coefs_AAinds(
        Equations, 
        LIR_data
    )

    # Interpolate
    aaLCs, aaInterpolants_pre, elLCs, elInterpolants_pre = interpolate(
        Gdf, 
        AAdata, 
        Elsedata
    )
 
    # Organize data and compute estimates
    Estimate, CoefficientsUsed = organize_data(
        aaLCs, 
        elLCs, 
        aaInterpolants_pre, 
        elInterpolants_pre, 
        Gdf, 
        AAdata,
        Elsedata
    )

    # Calculate initial uncertainties for lirs
    Uncertainties = emlr_estimate(
        Equations, 
        DesiredVariables, 
        Path,
        OutputCoordinates, 
        PredictorMeasurements, 
        unc_combo_dict, 
        dunc_combo_dict,
        Coefficients=CoefficientsUsed)
   
    # First of three steps to adjust pH and DIC for
    # anthropogenic carbon, as needed
    Cant_adjusted, Cant, Cant2002 = adjust_pH_DIC(
        DesiredVariables,
        VerboseTF,
        EstDates,
        Estimate,
        PredictorMeasurements,
        OutputCoordinates,
        **kwargs
    )

    # Second of three steps for Cant adjustment, for pH only
    Cant_adjusted = pH_adjustment(
        Path,
        DesiredVariables, 
        EstDates, 
        Cant,
        Cant2002, 
        PerKgSwTF,
        Cant_adjusted, 
        Estimate,
        PredictorMeasurements,
        OutputCoordinates,
        C,
        Uncertainties_pre,
        DUncertainties_pre,
        **kwargs
    )

    # Last of three adjustments for anthropogenic carbon
    Cant_adjusted, combos2, values2 = pH_adjcalc(
        DesiredVariables,
        VerboseTF,
        Estimate,
        Cant_adjusted,
        **kwargs
    )

    # Finalizing formatting of estimate output
    Estimates = final_formatting(
        DesiredVariables, 
        Cant_adjusted, 
        Estimate
    )
   
     # Stopping the timer
    toc = time.perf_counter()
    print(f"PyESPER_LIR took {toc - tic:0.4f} seconds, or {(toc-tic)/60:0.4f} minutes to run")    

    return Estimates, CoefficientsUsed, Uncertainties
