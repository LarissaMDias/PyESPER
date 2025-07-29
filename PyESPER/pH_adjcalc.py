def pH_adjcalc(DesiredVariables, VerboseTF, Est_pre={}, Cant_adjusted={}, **kwargs):

    """
    Calculating pH adjustment if units are not in molal format

    Inputs:
        DesiredVariables: List of variables requested
        VerboseTF: Boolean option to suppress warnings
        Est_pre: Dictionary of estimates thus far for each variable equation combination
        Cant_adjusted: Dictionary of estimates adjusted for anthropogenic carbon for each
            combination
        **kwargs: Please see README for full information

    Outputs:
        Cant_adjusted: Dictionary of estimates adjusted for anthropogenic carbon
        combos2: List of equation case-desired variable combinations
        values2: List of estimates
    """

    combos2 = list(Est_pre.keys())
    values2 = list(Est_pre.values())

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

    return Cant_adjusted, combos2, values2

