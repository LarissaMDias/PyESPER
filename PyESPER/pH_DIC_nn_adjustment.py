def pH_DIC_nn_adjustment(Path, DesiredVariables, Estimates, YouHaveBeenWarnedCanth, OutputCoordinates={}, PredictorMeasurements={}, **kwargs):

    """
    Calculating the anthropogenic carbon component for DIC and pH, if needed

    Inputs:
        Path: Path for user computer
        DesiredVariables: List of variables for which estimates are requested
        Estimates: Dictionary of estimates for each equation-variable combination
        YouHaveBeenWarnedCanth: Boolean for avoiding repeat warnings
        OutputCoordinates: Dictionary of coordinates for requested estimates
        PredictorMeasurements: Dictionary of measurements for calculations
        **kwargs: Please see README for full information

    Outputs:
        Cant_adjusted: Dictionary of estimates adjusted for anthropogenic carbon
            for each equation-variable combination
    """

    from PyESPER.simplecantestimatelr import simplecantestimatelr
    import numpy as np
    from PyESPER.defaults import defaults
    from PyESPER.lir_uncertainties import measurement_uncertainty_defaults
    from PyESPER.inputdata_organize import inputdata_organize
    from PyESPER.temperature_define import temperature_define
    from PyESPER.iterations import iterations
    from PyESPER.define_polygons import define_polygons
    from PyESPER.run_nets import run_nets
    from PyESPER.process_netresults import process_netresults
    import seawater as sw
    import PyCO2SYS as pyco2

    # Interpreting kwargs and setting defaults if needed
    VerboseTF = kwargs.get("VerboseTF", True)
    n = max(len(v) for v in OutputCoordinates.values())
    if "EstDates" in kwargs:
        d = np.array(kwargs["EstDates"])
        EstDates = (
            [item for sublist in [kwargs["EstDates"]] * (n + 1) for item in sublist]
            if len(d) != n else list(d)
        )
    else:
        EstDates = [2002.0] * n

    # Predefining empty dictionary for results and parsing out estimates
    Cant_adjusted = {}
    combos2 = list(Estimates.keys())
    values2 = list(Estimates.values())

    # Only if dates are available and if needed, continue
    if "EstDates" in kwargs and ("DIC" in DesiredVariables or "pH" in DesiredVariables):
        if not YouHaveBeenWarnedCanth:   
            if VerboseTF:
                print("Estimating anthropogenic carbon for PyESPER_NN.")
            longitude = np.mod(OutputCoordinates["longitude"], 360)
            latitude = np.array(OutputCoordinates["latitude"])
            depth = np.array(OutputCoordinates["depth"])
            Cant, Cant2002 = simplecantestimatelr(EstDates, longitude, latitude, depth)
            YouHaveBeenWarnedCanth = True
    
        for combo, a in zip(combos2, values2):
            a = np.array(a, dtype=float)
            if combo.startswith("DIC"):
                a = np.where(np.isnan(a), "nan", a + Cant - Cant2002)
                Cant_adjusted[combo] = a.tolist()
            else:
                Cant_adjusted[combo] = a.tolist()
        
        # Extra perturbation for pH
        if "pH" in DesiredVariables:   
            warning = []
            for combo, values in zip (combos2, values2):
                if combo.startswith("pH"):
                    salinity = PredictorMeasurements["salinity"]
                    PM_pH = {"salinity": salinity}
                    eq = [16]
                    _, n, _, EstDates, C, PerKgSwTF, MeasUncerts = defaults(
                        ["TA"],
                        PM_pH,
                        OutputCoordinates,
                        **kwargs
                    )
                    U_pre, DU_pre = measurement_uncertainty_defaults(
                        n,
                        PM_pH,
                        MeasUncerts
                    )
                    InputAll = inputdata_organize(
                        EstDates,
                        C,
                        PM_pH,
                        U_pre
                    )
                    PM_pH, InputAll = temperature_define(
                        ["TA"],
                        PM_pH,
                        InputAll,
                        **kwargs
                    )
                    code, unc_combo_dict, dunc_combo_dict = iterations(
                        ["TA"],
                        eq,
                        PerKgSwTF,
                        C,
                        PM_pH,
                        InputAll,
                        U_pre,
                        DU_pre 
                    )
                    df = define_polygons(C)
                    EstAtl, EstOther = run_nets(
                        ["TA"],
                        eq,
                        code
                    )
                    alkest = process_netresults(
                        eq,
                        code,
                        df,
                        EstAtl,
                        EstOther
                     )
                    EstAlk = np.array(alkest["TA16"])
                    EstSi = EstP = [0] * len(EstAlk)
                    Pressure = sw.pres(OutputCoordinates["depth"], OutputCoordinates["latitude"])
                    Est = np.array(values)
                        
                    # CO2SYS calculations
                    Out = pyco2.sys(
                        par1=EstAlk,
                        par2=Est,
                        par1_type=1,
                        par2_type=3,
                        salinity=salinity,
                        temperature=PredictorMeasurements["temperature"],
                        temperature_out=PredictorMeasurements["temperature"],
                        pressure=Pressure,
                        pressure_out=Pressure,
                        total_silicate=EstSi,
                        total_phosphate=EstP,
                        opt_total_borate= 2
                    )
                    DICadj = Out["dic"] + Cant - Cant2002
                    OutAdj = pyco2.sys(
                        par1=EstAlk,
                        par2=DICadj,
                        par1_type=1,
                        par2_type=2,
                        salinity=salinity,
                        temperature=PredictorMeasurements["temperature"],
                        temperature_out=PredictorMeasurements["temperature"],
                        pressure=Pressure,
                        pressure_out=Pressure,
                        total_silicate=EstSi,
                        total_phosphate=EstP,
                        opt_total_borate=2
                    )   
                    pHadj = OutAdj["pH"] 
                    
                    if any(np.isnan(pHadj)):
                        warning_message = (
                            "Warning: CO2SS took >20 iterations to converge. The correcponding estimate(s) will be NaN. "
                            "This typically happens when ESPER_NN is pooorly suited for estimating water with the given properties "
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
        print("Warning: DIC or pH is a requested output but the user did not provide dates for the desired estimates. The estimates will be specific to 2002.0 unless the optional EstDates input is provided (recommended).")
        YouHaveBeenWarnedCanth = True
                        
    if kwargs.get("pHCalcTF") and "pH" in DesiredVariables:
        if VerboseTF:
            print("Recalculating the pH to be appropriate for pH values calculated from TA and DIC.")
        for combo, pH_values in zip(combos2, values2):
            if combo.startswith("pH"):
                pH_adjcalc_Est = [(pH + 0.3168) / 1.0404 for pH in pH_values]
                Cant_adjusted[combo] = pH_adjcalc_Est
                        
    return Cant_adjusted

