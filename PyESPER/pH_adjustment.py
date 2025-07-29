def pH_adjustment(
    Path,
    DesiredVariables,
    Dates, 
    Cant, 
    Cant2002,
    PerKgSwTF, 
    Cant_adjusted={},
    Est_pre={}, 
    PredictorMeasurements={}, 
    OutputCoordinates={}, 
    C={},  
    Uncertainties_pre={}, 
    DUncertainties_pre={},
    **kwargs
):

    """
    Adjusting pH for anthropogenic carbon in the second of three Cant adjustment
        scripts

    Inputs:
        Path: User-defined computer path
        DesiredVariables: List of desired variables to estimate
        Dates: List of dates for measurements
        Cant: Numpy array of anthropogenic carbon estimates for each estimate
        Cant2002: Numpy array of anthropogenic carbon estimates for each
            estimate for the year 2002
        PerKgSwTF: Boolean for whether measurements were in molar or molal units
        Cant_adjusted: Dictionary of estimates adjusted for Cant
        Est_pre: Dictionary of estimates for each combination
        PredictorMeasurements: Dictionary of measurements for estimates
        OutputCoordinates: Dictionary of geographic coordinates for estimates
        C: Dictionary of adjusted coordinates for estimates
        Uncertainties_pre: Dictionary of measurement uncertainties for estimates
        DUncertainties_pre: Dictionary of default measurement uncertainties for
            estimates
        **kwargs: Please see README

    Output:
        Cant_adjusted: Dictionary of adjusted values for Cant for each combination
    """

    import numpy as np
    import seawater as sw
    import PyCO2SYS as pyco2
    from PyESPER.inputdata_organize import inputdata_organize
    from PyESPER.temperature_define import temperature_define
    from PyESPER.iterations import iterations
    from PyESPER.fetch_data import fetch_data
    from PyESPER.input_AAinds import input_AAinds
    from PyESPER.coefs_AAinds import coefs_AAinds
    from PyESPER.interpolate import interpolate
    from PyESPER.organize_data import organize_data
        
    # Finding the estimates within the dictionary
    combos2 = list(Est_pre.keys())
    values2 = list(Est_pre.values())
        
    # Setting the default warning message boolean and Verbose
    YouHaveBeenWarnedCanth = False
    VerboseTF = kwargs.get("VerboseTF", True)

    # Proceed only if dates are provided and DIC or pH is requested
    if "EstDates" in kwargs and ("DIC" in DesiredVariables or "pH" in DesiredVariables):
        if "pH" in DesiredVariables:
            warning = []
            for combo, values in zip(combos2, values2):
                if combo.startswith("pH"):
                    # Rerunning LIRs for when pH is requested to obtain TA estimates
                    # based only on salinity input
                    salinity = PredictorMeasurements["salinity"]
                    PM_pH = {"salinity": salinity}
                    eq = [16]
                    InputAll2 = inputdata_organize(
                        Dates,
                        C,
                        PM_pH,
                        Uncertainties_pre
                    )
                    PredictorMeasurements2, InputAll2 = temperature_define(
                        ["TA"],   
                        PM_pH,
                        InputAll2,
                        **kwargs
                    )
                    code, unc_combo_dict, dunc_combo_dict = iterations(
                        ["TA"],
                        eq,
                        PerKgSwTF,
                        C,
                        PredictorMeasurements2,
                        InputAll2,
                        Uncertainties_pre,
                        DUncertainties_pre
                    )
                    LIR_data = fetch_data(["TA"], Path)
                    AAdata, Elsedata = input_AAinds(C, code)
                    Gdf, CsDesired = coefs_AAinds(eq, LIR_data)
                    aaLCs, aaInterpolants_pre, elLCs, elInterpolants_pre = interpolate(
                        Gdf,  
                        AAdata,
                        Elsedata
                    )
                    alkest, _ = organize_data(
                        aaLCs,
                        elLCs,    
                        aaInterpolants_pre,
                        elInterpolants_pre,
                        Gdf,
                        AAdata,
                        Elsedata
                    )
                    EstAlk = np.array(alkest["TA16"])
                    EstAlk = np.transpose(EstAlk)
                    EstSi = EstP = [0] * len(EstAlk)
                    # Calculating pressure using the sw package
                    Pressure = sw.pres(OutputCoordinates["depth"], OutputCoordinates["latitude"])
                    Est = np.array(values)
                    Est = np.transpose(Est)
                    Est = Est[0]
                    temperature = np.array(PredictorMeasurements["temperature"])
                    # Now using CO2SYS to calculate DIC based on the above LIR results for TA
                    # CO2SYS calculations
                    kwargCO2 = {
                        "par1":EstAlk,
                        "par2":Est,
                        "par1_type":1,
                        "par2_type":3,
                        "salinity":salinity,  
                        "temperature":temperature,
                        "temperature_out":temperature,
                        "pressure":Pressure,
                        "pressure_out":Pressure,
                        "total_silicate":EstSi,
                        "total_phosphate":EstP,
                        "opt_total_borate":2}
                    Out = pyco2.sys(**kwargCO2)
                    DICadj = Out["dic"] + Cant - Cant2002
                    
                    # Using this DIC to calculate pH from CO2SYS
                    kwargCO2_2 = {
                        "par1":EstAlk,
                        "par2":DICadj,
                        "par1_type":1,
                        "par2_type":2,
                        "salinity":salinity,
                        "temperature":temperature,
                        "temperature_out":temperature,
                        "pressure":Pressure,
                       "pressure_out":Pressure,
                        "total_silicate":EstSi,
                        "total_phosphate":EstP,
                        "opt_total_borate":2}
                    OutAdj = pyco2.sys(**kwargCO2_2)
                    pHadj = OutAdj["pH"]
                        
                    # Check for convergence warnings
                    if np.isnan(pHadj).any():   
                        warning_message = (
                            "Warning: CO2SYS took >20 iterations to converge. The corresponding estimate(s) will be NaN. "
                            "This typically happens when ESPER_LIR is poorly suited for estimating water with the given properties "
                            "(e.g., very high or low salinity or estimates in marginal seas)."
                        )
                        warning.append(warning_message)
                    # Append to adjusted Cant dictionary
                    Cant_adjusted[combo] = pHadj.tolist()
                        
                # Print warnings if any
                if warning:
                    print(warning[0]) 
                        
    elif "EstDates" not in kwargs and ("DIC" or "pH" in DesiredVariables) and not VerboseTF and YouHaveBeenWarnedCanth:
        print("Warning: DIC or pH is a requested output but the user did not provide dates for the desired esimtates. The estimates "
            "will be specific to 2002.0 unless the optional EstDates input is provided (recommended).")

        YouHaveBeenWarnedCanth = True
                        
    return Cant_adjusted
