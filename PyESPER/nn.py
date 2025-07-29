def nn(DesiredVariables, Path, OutputCoordinates={}, PredictorMeasurements={}, **kwargs):

    """
    Neural networks for seawater property estimation as part of PyESPERsv1.0.0

    Inputs: 
        DesiredVariables: list of desired variables to estimate
        Path: User-defined computer path
        OutputCoordinates: List of coordinates to produce estimates for
        PredictorMeasurements: List of predictor measurements to use in NNs
        **kwargs: Optional inputs specific to users (please see README for full description)

    Outputs:
        Estimates: Dictionary of estimates for each equation-desired variable combination
        Uncertainties: Dictionary of uncertainties for each equation-desired variable combination
    """

    import time
    from PyESPER.errors import errors
    from PyESPER.defaults import defaults
    from PyESPER.lir_uncertainties import measurement_uncertainty_defaults
    from PyESPER.inputdata_organize import inputdata_organize
    from PyESPER.temperature_define import temperature_define
    from PyESPER.iterations import iterations
    from PyESPER.fetch_polys_NN import fetch_polys_NN
    from PyESPER.define_polygons import define_polygons
    from PyESPER.run_nets import run_nets
    from PyESPER.process_netresults import process_netresults
    from PyESPER.organize_nn_output import organize_nn_output
    from PyESPER.pH_DIC_nn_adjustment import pH_DIC_nn_adjustment
    from PyESPER.final_formatting import final_formatting

    # Starting the timer 
    tic = time.perf_counter()
    
    # Function that provides custom error messages for erroneous inputs
    errors(OutputCoordinates, PredictorMeasurements)

    # Function which calculates default measurement uncertainties
    Equations, n, VerboseTF, EstDates, C, PerKgSwTF, MeasUncerts = defaults(
        DesiredVariables,
        PredictorMeasurements,
        OutputCoordinates,
        **kwargs
    )

    # Function that processes the input values and default uncertainties and makes sense of it
    Uncertainties_pre, DUncertainties_pre = measurement_uncertainty_defaults(
        n,
        PredictorMeasurements,
        MeasUncerts
    )

    # Creating a dictionary of input data
    InputAll = inputdata_organize(
        EstDates,
        C,
        PredictorMeasurements,
        Uncertainties_pre
    )

    # Defining temperature if not provided in the correct format
    PredictorMeasurements, InputAll = temperature_define(
        DesiredVariables, 
        PredictorMeasurements,
        InputAll,
        **kwargs
    )

    # Iterating through possible variable-equation combinations 
    # to produce a usable dictionary of predictor values
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

    # Creating boolean indicators for different ocean regions
    df = define_polygons(C)

    # Running the actual neural nets
    EstAtl, EstOther = run_nets(
        DesiredVariables, 
        Equations, 
        code
    )

    # Processing and organizing results from nets, including regional
    # smoothing based on boolean indicators
    Estimates = process_netresults(
        Equations, 
        code, 
        df, 
        EstAtl, 
        EstOther
    )

    # Organize output and iteratively calculate uncertainties
    Uncertainties = organize_nn_output(
        Path,
        DesiredVariables,
        OutputCoordinates,
        PredictorMeasurements,
        **kwargs
    )

    # Adjust pH and DIC for anthropogenic carbon
    YouHaveBeenWarnedCanth=False
    Cant_adjusted = pH_DIC_nn_adjustment(
        Path,
        DesiredVariables, 
        Estimates,
        YouHaveBeenWarnedCanth,
        OutputCoordinates,
        PredictorMeasurements,
        **kwargs
    )
    
    # Final formatting and presentation of code
    Estimates = final_formatting(
        DesiredVariables,
        Cant_adjusted,
        Estimates
    )
       
    toc = time.perf_counter()
    print(f"PyESPER_NN took {toc - tic:0.4f} seconds, or {(toc-tic)/60:0.4f} minutes to run")

    return Estimates, Uncertainties
