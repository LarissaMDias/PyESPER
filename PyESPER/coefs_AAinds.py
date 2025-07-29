def coefs_AAinds(Equations, LIR_data):

    """
    Separates coefficients from MATLAB ESPERv1 into Atlantic and Arctic or other regions.

    Inputs:
        Equations: List of equations for use in ESPERs
        LIR_data: List of dictionaries of data from MATLAB trainings (pre-processed)

    Outputs:
        Gdf: Dictionary of pre-trained and processed LIR data for grid of coordinates
        CsDesired: Dictionary of equation coefficients based on user-defined output
    """

    import numpy as np

    # Use boolean for AA or Else to separate coefficients into Atlantic or not
    GridCoords, Cs, AAInds = LIR_data[:3]
    DVs, CsVs = list(Cs.keys()), list(Cs.values())
    ListVars = np.arange(len(AAInds))
    GridValues = np.array(list(GridCoords.values())[0], dtype=float)
    AAIndValues = np.array(list(AAInds.values())[0], dtype=float)

    lon_grid, lat_grid, d2d_grid, aainds = np.array((GridValues[:,0])), np.array((GridValues[:,1])), \
       np.array(GridValues[:,2])/25, np.array(AAIndValues[:,0])

    names = ['lon', 'lat', 'd2d', "C_alpha", "C_S", "C_T", "C_A", "C_B", "C_C", 'AAInds']
    Gdf, CsDesired = {}, {}

    # Moving data into pre-defined dictionaries
    for lvar, name in zip(ListVars, DVs):
        Cs2 = CsVs[:][lvar][:]
        for e in Equations:
            CsName = f'Cs{name}{e}'
            CsDesired[CsName] = Cs2[e-1][:]
            Cs3 = Cs2[e-1][:]
            C_alpha, C_S, C_T, C_A, C_B, C_C = np.array(Cs3[:,0]), np.array(Cs3[:,1]), np.array(Cs3[:,2]), np.array(Cs3[:,3]), \
                np.array(Cs3[:,4]), np.array(Cs3[:,5])
            Gdf[f"{name}{e}"] = {
                names[0]: np.array(lon_grid),
                names[1]: np.array(lat_grid),
                names[2]: np.array(d2d_grid),
                names[3]: np.array(C_alpha),
                names[4]: np.array(C_S),
                names[5]: np.array(C_T),
                names[6]: np.array(C_A),
                names[7]: np.array(C_B),
                names[8]: np.array(C_C),
                names[9]: np.array(aainds)
            }

    return Gdf, CsDesired

