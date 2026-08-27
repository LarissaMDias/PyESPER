def fetch_polys_NN(Path, DesiredVariables):

    """
    Loads the uncertainty polygons for NNs
    """

    from scipy.io import loadmat
    import os

    from PyESPER.paths import data_root

    Path = data_root(Path)
    for v in DesiredVariables:
        fname = os.path.join(Path, f"Uncertainty_Polys/NN_files_{v}_Unc_Poly.mat")
        NNs = loadmat(fname)
        Polys, UncGrid = NNs["Polys"][0][0], NNs["UncGrid"][0][0]

    NN_data = [Polys, UncGrid]
    return NN_data
