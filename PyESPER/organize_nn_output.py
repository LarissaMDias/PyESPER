def organize_nn_output(Path, DesiredVariables, OutputCoordinates={}, PredictorMeasurements={},  **kwargs):

    """
    Some near-final data organization
    """

    import numpy as np
    from PyESPER.defaults import defaults
    from PyESPER.lir_uncertainties import measurement_uncertainty_defaults
    from PyESPER.inputdata_organize import inputdata_organize
    from PyESPER.temperature_define import temperature_define
    from PyESPER.iterations import iterations
    from PyESPER.fetch_polys_NN import fetch_polys_NN
    from PyESPER.define_polygons import define_polygons
    from PyESPER.run_nets import run_nets
    from PyESPER.process_netresults import process_netresults
    from PyESPER.emlr_nn import emlr_nn

    PD_final, DPD_final, Unc_final, DUnc_final = [], [], [], []

    for d, var in enumerate(DesiredVariables):
        Pertdv, DPertdv, Unc, DUnc = [], [], [], []
        var = [var] # Wrap single variable in a list
        keys = ["sal_u", "temp_u", "phosphate_u", "nitrate_u", "silicate_u", "oxygen_u"]

        Equations, n, e, p, VerboseTF, EstDates, C, PerKgSwTF, MeasUncerts = defaults(
            var, 
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
     
        NN_data = fetch_polys_NN(Path, var)

        df = define_polygons(C)

        EstAtl, EstOther = run_nets(DesiredVariables, Equations, code)

        Estimate, no_equations = process_netresults(
            Equations,
            code,
            df,
            EstAtl,
            EstOther
        )

        EMLR = emlr_nn(
            Path,
            DesiredVariables,
            Equations,
            OutputCoordinates,
            PredictorMeasurements
        )

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
            Eqs2, n2, e2, p2, VerbTF2, EstDates2, C2, PerKgSwTF2, MeasUncerts2 = defaults(
                var, 
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
            NN_data2 = fetch_polys_NN(
                Path,
                var
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
            PertEst, no_equations2 = process_netresults(
                Eqs2, 
                code2,
                df2,
                EstAtl2,
                EstOther2
            )
            DefaultPertEst, no_equations3 = process_netresults(
                Eqs2,
                code3,
                df2,
                EstAtl3, 
                EstOther3
            ) 

            # Extract estimates and perturbation results
            combo, estimates = list(Estimate.keys()), list(Estimate.values())
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
        DUnc_final.append(DUnc)

    est = list(Estimate.values())
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
   
    return Uncertainties
