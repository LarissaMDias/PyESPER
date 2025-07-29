def fetch_data (DesiredVariables, Path):
    
    """
    Gathers the necessary LIR files that were pre-trained in MATLAB ESPERs

    Inputs:
        DesiredVariables: List of desired output estimate variables
        Path: User-defined computer path of locations of files

    Outputs:
        LIR_data: List of dictionaries of LIR data
    """

    from scipy.io import loadmat
    import os
    import numpy as np

    # Predefine dictionaries of output
    AAIndsCs, GridCoords, Cs = {}, {}, {}

    # Load necessary files
    for v in DesiredVariables:
        fname1 = os.path.join(Path, f"Mat_fullgrid/LIR_files_{v}_fullCs1.mat")
        fname2 = os.path.join(Path, f"Mat_fullgrid/LIR_files_{v}_fullCs2.mat")
        fname3 = os.path.join(Path, f"Mat_fullgrid/LIR_files_{v}_fullCs3.mat")
        fname4 = os.path.join(Path, f"Mat_fullgrid/LIR_files_{v}_fullGrids.mat")

        Cs1 = loadmat(fname1)
        Cs2 = loadmat(fname2)
        Cs3 = loadmat(fname3)
        Grid = loadmat(fname4)

        # Extract and store all arrays
        UncGrid = np.array(Grid["UncGrid"][0][0])
        GridCoodata = np.array(Grid["GridCoords"])
        AAInds = np.array(Grid["AAIndsM"])

        Csdata1 = np.array(Cs1["Cs1"])
        Csdata2 = np.array(Cs2["Cs2"])
        Csdata3 = np.array(Cs3["Cs3"])

        # Store as NumPy arrays
        AAIndsCs[v] = AAInds
        GridCoords[v] = GridCoodata

        # Combine along axis 1, then store each layer in list
        Csdata = np.concatenate((Csdata1, Csdata2, Csdata3), axis=1)
        Cs[v] = [Csdata[:, :, i] for i in range(Csdata.shape[2])]

    # Store all in one list
    LIR_data = [GridCoords, Cs, AAIndsCs, UncGrid]

    return LIR_data

