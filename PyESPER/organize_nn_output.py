def organize_nn_output(Path, DesiredVariables, OutputCoordinates={}, PredictorMeasurements={},  **kwargs):

    """
    Iteratively calculate uncertainties

    Inputs:
        Path: User-defined computer path
        Desired Variables: List of variables requested
        OutputCoordinates: Dictionary of geographic coordinates
        PredictorMeasurements: Dictionary of predictor measurements
        **kwargs: Please see README

    Outputs:
        Uncertainties: Dictionary of uncertainty estimates for each
            equation-variable combination
    """

    import numpy as np
    from PyESPER.defaults import defaults
    from PyESPER.lir_uncertainties import measurement_uncertainty_defaults
    from PyESPER.inputdata_organize import inputdata_organize
    from PyESPER.temperature_define import temperature_define
    from PyESPER.iterations import iterations
    from PyESPER.define_polygons import define_polygons
    from PyESPER.run_nets import run_nets
    from PyESPER.process_netresults import process_netresults
    from PyESPER.emlr_nn import emlr_nn

    # Predefine output lists
    PD_final, DPD_final, Unc_final, DUnc_final = [], [], [], []
    emlr = []

    # Rerun nets for each variable
    for d, var in enumerate(DesiredVariables):
        Pertdv, DPertdv, Unc, DUnc = [], [], [], []
        var = [var] # Wrap single variable in a list
        keys = ["sal_u", "temp_u", "phosphate_u", "nitrate_u", "silicate_u", "oxygen_u"]

        Equations, n, VerboseTF, EstDates, C, PerKgSwTF, MeasUncerts = defaults(
            var,
            PredictorMeasurements,
            OutputCoordinates,
            **kwargs
        )
        Uncertainties_pre, DUncertainties_pre = measurement_uncertainty_defaults(
            n,
            PredictorMeasurements,
            MeasUncerts
        )
    
        InputAll = inputdata_organize( 
            EstDates,
            C,
            PredictorMeasurements,
            Uncertainties_pre
        )
    
        PredictorMeasurements, InputAll = temperature_define(
            var,
            PredictorMeasurements,
            InputAll,
            **kwargs
        )
            
        code, unc_combo_dict, dunc_combo_dict = iterations(
            var,
            Equations,
            PerKgSwTF,
            C,
            PredictorMeasurements,
            InputAll,
            Uncertainties_pre,
            DUncertainties_pre
        )

        df = define_polygons(C)
            
        EstAtl, EstOther = run_nets(var, Equations, code)
            
        Estimate = process_netresults(
            Equations,
            code,
            df,
            EstAtl,
            EstOther
        )
            
        EMLR = emlr_nn(
            Path,
            var,
            Equations,
            OutputCoordinates,
            PredictorMeasurements
        )
      
        emlr.append(EMLR)
        names = list(PredictorMeasurements.keys())
        PMs = list(PredictorMeasurements.values())

        # Replace "nan" with 0 in PMs using list comprehensions
        PMs_nonan = [[0 if val == "nan" else val for val in pm] for pm in PMs]
        
        # Transpose PMs_nonan
        PMs = np.transpose(PMs_nonan)
            
        PMs3, DMs3 = {}, {}
            
        for pred in range(len(PredictorMeasurements)):
            num_coords = len(OutputCoordinates["longitude"])
            num_preds = len(PredictorMeasurements)
            
            # Initialize perturbation arrays
            Pert = np.zeros((num_coords, num_preds))
            DefaultPert = np.zeros((num_coords, num_preds))
            
            # Populate perturbation arrays
            Pert[:, pred] = Uncertainties_pre[keys[pred]]
            DefaultPert[:, pred] = DUncertainties_pre[keys[pred]]
            
            # Apply perturbations
            PMs2 = PMs + Pert
            DMs2 = PMs + DefaultPert
        
            # Update PMs3 and DMs3 dictionaries
            for col, name in enumerate(names):
                PMs3[name] = PMs2[:, col].tolist()
                DMs3[name] = DMs2[:, col].tolist()
        
            # Run preprocess_applynets for perturbed and default data
            VTF = False
            Eqs2, n2, VerbTF2, EstDates2, C2, PerKgSwTF2, MeasUncerts2 = defaults(
                var,
                PMs2,
                OutputCoordinates,
                Equations=Equations,
                EstDates=EstDates,
                VerboseTF = VTF,
            )
            U_pre2, DU_pre2 = measurement_uncertainty_defaults(
                n2,
                PMs3,
                MeasUncerts2
            )
            InputAll2 = inputdata_organize(
                EstDates2,
                C2,
                PMs3,
                U_pre2
            )
            InputAll3 = inputdata_organize(   
                EstDates2,
                C2,
                DMs3,
                U_pre2
            )
            PMs3, InputAll2 = temperature_define(
                var,
                PMs3,
                InputAll2,
                **kwargs
            )
            DMs3, InputAll3 = temperature_define(
                var,
                DMs3,
                InputAll3,
                **kwargs
            )
            code2, unc_combo_dict2, dunc_combo_dict2 = iterations(
                var,
                Eqs2,
                PerKgSwTF2,
                C2,  
                PMs3, 
                InputAll2,
                U_pre2,
                DU_pre2   
            )
            code3, unc_combo_dict3, dunc_combo_dict3 = iterations(
                var,  
                Eqs2,
                PerKgSwTF2,
                C2, 
                DMs3,
                InputAll3,
                U_pre2, 
                DU_pre2
            )
            df2 = define_polygons(C2)
            EstAtl2, EstOther2 = run_nets(
                var,
                Eqs2,   
                code2
            )
            EstAtl3, EstOther3 = run_nets(
                var, 
                Eqs2,
                code3
            )
            PertEst = process_netresults(
                Eqs2,  
                code2,    
                df2,
                EstAtl2,
                EstOther2
            )
            DefaultPertEst = process_netresults(
                Eqs2,
                code3,
                df2,
                EstAtl3,
                EstOther3
            )
            
            # Extract estimates and perturbation results
            estimates = [np.array(v) for v in Estimate.values()]
            pertests = [np.array(v) for v in PertEst.values()]
            defaultpertests = [np.array(v) for v in DefaultPertEst.values()]
             
            # Initialize result lists
            PertDiff, DefaultPertDiff, Unc_sub2, DUnc_sub2 = [], [], [], []

            for c in range(len(Equations)):
            # Compute differences and squared differences using numpy
                PD = estimates[c] - pertests[c]
                DPD = estimates[c] - defaultpertests[c]
                Unc_sub1 = (estimates[c] - pertests[c])**2
                DUnc_sub1 = (estimates[c] - defaultpertests[c])**2
                
               # Append results
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
        DUnc_final.append(DUnc)
            
    # Compute final uncertainty propagation
            
    est = [np.array(v) for v in Estimate.values()]
    emlr_combined = {k: v for d in emlr for k, v in d.items()}
    Uncertainties = {}
    for dv in range(0, len(DesiredVariables)):
        dvu = []
        for eq in range(0, len(Equations)):
            sumu = []
            name = DesiredVariables[dv] + str(Equations[eq])
            for n in range(0, len(est[0])):
                # Collect uncertainty contributions from each perturbation
                u =  np.array([Unc_final[dv][pre][eq][n] for pre in range(len(PredictorMeasurements))])
                du = np.array([DUnc_final[dv][pre][eq][n] for pre in range(len(PredictorMeasurements))])
                eu = emlr_combined[name][n]
                # Final uncertainty formula
                total_uncertainty = np.sqrt(np.sum(u) - np.sum(du) + eu**2)
                sumu.append(total_uncertainty)
            dvu.append(sumu)

            Uncertainties[name] = sumu

    return Uncertainties

