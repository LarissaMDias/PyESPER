def mixed(DesiredVariables, Path, OutputCoordinates={}, PredictorMeasurements={}, **kwargs):
    
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
    from .lir import lir
    from .nn import nn
    
    tic = time.perf_counter()

    # Fetch estimates and uncertainties from PyESPER_LIR and PyESPER_NN
    EstimatesLIR, _, UncertaintiesLIR = lir(DesiredVariables, Path, OutputCoordinates, PredictorMeasurements, **kwargs)
    # nn() has no coefficient output, so this kwarg is LIR-only.
    kwargs.pop("want_coefficients", None)
    EstimatesNN, UncertaintiesNN = nn(DesiredVariables, Path, OutputCoordinates, PredictorMeasurements, **kwargs)

    Estimates, Uncertainties = {}, {}
    # Uncertainties are None when the caller passed compute_uncertainties=False (which
    # the gridded xr_methods path always does). Previously this indexed them
    # unconditionally, so mixed_xr raised TypeError on every call.
    have_uncertainties = UncertaintiesLIR is not None and UncertaintiesNN is not None

    for est_type in EstimatesLIR.keys():
        # flatten(): the LIR path returns per-combination arrays that may carry a
        # trailing length-1 axis, unlike the NN path.
        estimates_lir = np.asarray(EstimatesLIR[est_type], dtype=np.float64).flatten()
        estimates_nn = np.asarray(EstimatesNN[est_type], dtype=np.float64)
        Estimates[est_type] = 0.5 * (estimates_lir + estimates_nn)
        if have_uncertainties:
            Uncertainties[est_type] = np.minimum(
                np.asarray(UncertaintiesLIR[est_type], dtype=np.float64),
                np.asarray(UncertaintiesNN[est_type], dtype=np.float64),
            )

    if not have_uncertainties:
        Uncertainties = None

    toc = time.perf_counter()
    print(f"PyESPER_Mixed took {toc - tic:0.4f} seconds, or {(toc-tic)/60:0.4f} minutes to run")    

    return Estimates, Uncertainties
