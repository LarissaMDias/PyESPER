def emlr_nn(Path, DesiredVariables, Equations, OutputCoordinates={}, PredictorMeasurements={}, **kwargs):

    """
    Estimating EMLR for neural networks.
    Returns a dictionary with (DesiredVariable, Equation) as keys and Uncertainties as values.
    """

    from PyESPER.fetch_polys_NN import fetch_polys_NN
    import numpy as np
    from scipy.interpolate import griddata

    EMLR = {}

    for dv in DesiredVariables:
        DV = f"{dv}"
        NN_data = fetch_polys_NN(Path, [DV])

        data_arrays = [
            np.nan_to_num(np.array([
                NN_data[1][i][c][b][a]
                for a in range(16)
                for b in range(11)
                for c in range(8)
            ]))
            for i in range(4)
        ]

        # Create Dictionary of predetermined Uncertainties
        UGridArray = {
            'UDepth': data_arrays[0],
            'USal': data_arrays[1],
            'Eqn': data_arrays[2],
            'RMSE': data_arrays[3],
        }

        UGridPoints = (UGridArray['UDepth'], UGridArray['USal'], UGridArray['Eqn'])
        UGridValues = UGridArray['RMSE']

        for eq in Equations:
            name = dv + str(eq)
            eq_array = np.full_like(OutputCoordinates['depth'], eq, dtype=float)

            # Perform estimation for each equation
            EM = griddata(
                UGridPoints,
                UGridValues,
                (OutputCoordinates['depth'], PredictorMeasurements['salinity'], eq_array),
                method='linear'
            )

            EMLR[name] = EM

    return EMLR

