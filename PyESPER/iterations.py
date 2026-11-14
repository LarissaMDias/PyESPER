def iterations(
    DesiredVariables,
    Equations,
    PerKgSwTF,
    C={},
    PredictorMeasurements={},
    InputAll={},
    Uncertainties={},
    DUncertainties={},
):
    """
    A function to iterate and define equation inputs depending upon
    Equation, DesiredVariable, and other user specifications.

    Inputs:
        DesiredVariables: List of desired variables for estimates
        Equations: List of equations to be used for estimates
        PerKgSwTF: Boolean indicating whether user input was in molal (default) or molar units
        C: Dictionary of pre-processed user geographic coordinates
        PredictorMeasurements: Dictionary of all input, with temperature processed as needed
        InputAll: Dictionary of deafault and user inputs with order stamp
        Uncertainties: Dictionary of user-defined uncertainties (or default if missing)
        DUncertainties: Dictionary of default uncertainties

    Output:
        code: Dictionary of filled-in combinations of predictor measurements relative to the reques$
            variable combination
        unc_combo_dict: Dictionary of filled-in uncertainty combinations for the requisite equation$
        dunc_combo_dict: Dictionary of filled-in default uncertainty combinations, as above

    NOTE: This function uses the now-deprecated seawater package for some calculations. This aligns with the
        current version of ESPERs (v1.0); however, this will be updated to the Gibb's seawater package in the
        next release.
    """

    import numpy as np
    import PyESPER.eos80_jit as sw  # grab local copies of NJIT'd seawater

    print("Organzing user input.")

    n = max(len(v) for v in C.values())

    depth = np.asarray(C["depth"])  # using arrays for speed boost
    latitude = np.asarray(C["latitude"])
    salinity = np.asarray(PredictorMeasurements["salinity"])

    temp = (
        np.asarray(PredictorMeasurements["temperature"])
        if "temperature" in PredictorMeasurements
        else np.full(n, 10)
    )
    temp_sw = sw.ptmp(salinity, temp, sw.pres(depth, latitude), pr=0)
    temperature_processed = temp_sw

    if "oxygen" in PredictorMeasurements:
        oxyg = np.asarray(PredictorMeasurements["oxygen"])
        oxyg_sw = sw.satO2(salinity, temp_sw) * 44.6596 - oxyg
        oxyg_sw[np.abs(oxyg_sw) < 0.0001] = 0
    else:
        oxyg_sw = np.full(n, np.nan)
    oxygen_processed = oxyg_sw

    nutrients = ["phosphate", "nitrate", "silicate"]
    proc_nut = {
        p: np.asarray(
            PredictorMeasurements.get(p, np.full(n, np.nan)), dtype=np.float64
        )
        for p in nutrients
    }

    if not PerKgSwTF:
        densities = (
            sw.dens(salinity, temperature_processed, sw.pres(depth, latitude))
            / 1000
        )
        for p in nutrients:
            proc_nut[p] /= densities

    phosphate_processed = proc_nut["phosphate"]
    nitrate_processed = proc_nut["nitrate"]
    silicate_processed = proc_nut["silicate"]

    data_cols = (
        np.full(n, np.nan),
        salinity,
        temperature_processed,
        phosphate_processed,
        nitrate_processed,
        silicate_processed,
        oxygen_processed,
    )

    def get_u(d, k):
        return np.asarray(d.get(k, np.full(n, np.nan)), dtype=np.float64)

    U_cols = (
        np.full(n, np.nan),
        get_u(Uncertainties, "sal_u"),
        get_u(Uncertainties, "temp_u"),
        get_u(Uncertainties, "phosphate_u"),
        get_u(Uncertainties, "nitrate_u"),
        get_u(Uncertainties, "silicate_u"),
        get_u(Uncertainties, "oxygen_u"),
    )

    DU_cols = (
        np.full(n, np.nan),
        get_u(DUncertainties, "sal_u"),
        get_u(DUncertainties, "temp_u"),
        get_u(DUncertainties, "phosphate_u"),
        get_u(DUncertainties, "nitrate_u"),
        get_u(DUncertainties, "silicate_u"),
        get_u(DUncertainties, "oxygen_u"),
    )

    Final_U_cols = tuple(np.maximum(u, du) for u, du in zip(U_cols, DU_cols))

    NeededForProperty = {
        "TA": [1, 2, 4, 6, 5],
        "DIC": [1, 2, 4, 6, 5],
        "pH": [1, 2, 4, 6, 5],
        "phosphate": [1, 2, 4, 6, 5],
        "nitrate": [1, 2, 3, 6, 5],
        "silicate": [1, 2, 3, 6, 4],
        "oxygen": [1, 2, 3, 4, 5],
    }

    VarVec = {
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
    }

    EqsString = [str(e) for e in Equations]
    code = {}
    unc_combo_dict = {}
    dunc_combo_dict = {}

    meta_cols = [
        "Order",
        "Dates",
        "Longitude",
        "Latitude",
        "Depth",
        "Salinity_u",
        "Temperature_u",
        "Phosphate_u",
        "Nitrate_u",
        "Silicate_u",
        "Oxygen_u",
    ]
    metadata_dict = {col: np.asarray(InputAll[col]) for col in meta_cols}
    unc_names = ["US", "UT", "UA", "UB", "UC"]

    for d in DesiredVariables:
        dv = NeededForProperty[d]
        for e in EqsString:
            vv = VarVec[e]
            p_vec = [v * req for v, req in zip(vv, dv)]
            prename = d + e

            p_dict = {
                "S": data_cols[p_vec[0]],
                "T": data_cols[p_vec[1]],
                "A": data_cols[p_vec[2]],
                "B": data_cols[p_vec[3]],
                "C": data_cols[p_vec[4]],
            }
            p_dict.update(metadata_dict)
            code[prename] = p_dict

            unc_combo_dict[prename] = {
                name: Final_U_cols[p_vec[i]]
                for i, name in enumerate(unc_names)
            }
            dunc_combo_dict[prename] = {
                name: DU_cols[p_vec[i]] for i, name in enumerate(unc_names)
            }

    return code, unc_combo_dict, dunc_combo_dict
