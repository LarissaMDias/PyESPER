def fetch_polys_NN(Path, DesiredVariables):

    """
    Loads the uncertainty polygons for NNs
    """

    from scipy.io import loadmat
    import os

    for v in DesiredVariables:
        fname = os.path.join(Path, "data", "Uncertainty_Polys", f"NN_files_{v}_Unc_Poly.mat")
        NNs = loadmat(fname)
        Polys, UncGrid = NNs["Polys"][0][0], NNs["UncGrid"][0][0]

    NN_data = [Polys, UncGrid]
    return NN_data
