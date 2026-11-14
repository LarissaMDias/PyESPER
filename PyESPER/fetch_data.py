def fetch_data(DesiredVariables, Path):
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

        Cs1 = loadmat(
            os.path.join(Path, "data", "Mat_fullgrid", f"LIR_files_{v}_fullCs1.mat"),
            squeeze_me=True,
        )
        Cs2 = loadmat(
            os.path.join(Path, "data", "Mat_fullgrid", f"LIR_files_{v}_fullCs2.mat"),
            squeeze_me=True,
        )
        Cs3 = loadmat(
            os.path.join(Path, "data", "Mat_fullgrid", f"LIR_files_{v}_fullCs3.mat"),
            squeeze_me=True,
        )
        Grid = loadmat(
            os.path.join(Path, "data", "Mat_fullgrid", f"LIR_files_{v}_fullGrids.mat")
        )

        # Extract and store all arrays
        UncGrid = Grid["UncGrid"][0][0]
        GridCoords[v] = Grid["GridCoords"]
        AAIndsCs[v] = Grid["AAIndsM"]

        # Combine along axis 1, then store each layer in list
        Csdata = np.concatenate((Cs1["Cs1"], Cs2["Cs2"], Cs3["Cs3"]), axis=1)
        Cs[v] = [Csdata[:, :, i] for i in range(Csdata.shape[2])]

    return [GridCoords, Cs, AAIndsCs, UncGrid]
