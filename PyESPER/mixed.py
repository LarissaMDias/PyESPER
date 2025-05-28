def PyESPER_Mixed(DesiredVariables, Path, OutputCoordinates={}, PredictorMeasurements={}, **kwargs):
    
    """
    Python interpretation of ESPER_Mixedv1.1

    Empirical Seawater Property Estimation Routines: Estimates seawater properties and estimate uncertainty from combinations of other parameter
    measurements.  PYESPER_Mixed refers specifically to code that averages estunated from PyESPER_NN and PyESPER_LIR. See either subfunction for
    comments. The input arguments are the same for this function and for both subfunctions.
    
    *************************************************************************
    Please send questions or related requests about PyESPER to lmdias@uw.edu.
    ************************************************************************* 
    """
    import time

    import numpy as np
    import pandas as pd

    from .lir import PyESPER_LIR
    from .nn import PyESPER_NN
    
    tic = time.perf_counter()

    # Fetch estimates and uncertainties from PyESPER_LIR and PyESPER_NN
    EstimatesLIR, UncertaintiesLIR, _ = PyESPER_LIR(DesiredVariables, Path, OutputCoordinates, PredictorMeasurements, **kwargs)
    EstimatesNN, UncertaintiesNN = PyESPER_NN(DesiredVariables, Path, OutputCoordinates, PredictorMeasurements, **kwargs)

    Estimates, Uncertainties = {}, {}
    for est_type in EstimatesLIR.keys():
        estimates_lir = np.array(EstimatesLIR[est_type])
        estimates_nn = np.array(EstimatesNN[est_type])
        uncertainties_lir = np.array(UncertaintiesLIR[est_type])
        uncertainties_nn = np.array(UncertaintiesNN[est_type])

        Estimates[est_type] = np.mean([estimates_lir, estimates_nn], axis=0).tolist()
        Uncertainties[est_type] = np.min([uncertainties_lir, uncertainties_nn], axis=0).tolist()

    toc = time.perf_counter()
    print(f"PyESPER_Mixed took {toc - tic:0.4f} seconds, or {(toc-tic)/60:0.4f} minutes to run")    

    return pd.DataFrame(Estimates), pd.DataFrame(Uncertainties)
