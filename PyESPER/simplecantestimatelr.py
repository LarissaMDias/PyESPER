def simplecantestimatelr(EstDates, longitude, latitude, depth):

    """
    Simple estimate of contribution of anthropogenic carbon to pH and DIC estimates.

    Inputs:
        EstDates: List of dates for which estimates will be made
        longitude:

    Ouptuts:
        CantMeas: List of anthropogenic carbon estimates
        Cant2002: List of anthropogenic carbon estimates for 2002
    """

    import numpy as np
    import pandas as pd
    from scipy.interpolate import griddata

    # Load interpolation points and values
    CantIntPoints = pd.read_csv('SimpleCantEstimateLR_full.csv')
    pointsi = (
        CantIntPoints['Int_long'] * 0.25,
        CantIntPoints['Int_lat'],
        CantIntPoints['Int_depth'] * 0.025,
    )
    values = CantIntPoints['values']

    # Scale input coordinates
    pointso = (
        np.array(longitude) * 0.25,
        np.array(latitude),
        np.array(depth) * 0.025,
    )

    # Interpolate and compute Cant2002
    Cant2002 = griddata(pointsi, values, pointso, method='linear')

    # Adjust for estimation dates
    EstDates = np.asarray(EstDates)
    CantMeas = Cant2002 * np.exp(0.018989 * (EstDates - 2002))

    return CantMeas, Cant2002

