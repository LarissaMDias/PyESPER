def interpolate(Gdf={}, AAdata={}, Elsedata={}):

    """
    This LIR function performs the interpolation on user-defined data

    Inputs:
        Gdf: Dictionary of pre-trained data for ESPER v1 (processed)
        AAdata: Dictionary of user input for Atlantic or Arctic
        Elsedata: Dictionary of user input not for Atlantic/Arctic

    Outputs:
        aaLCs: List of points to be interpolated within the Atlantic or Arctic
            regions
        aaInterpolants_pre: Scipy interpolant for Atlantic/Arctic region
        elLCs: List of points to be inteprolated outside of Atlantic/Arctic
        elInterpolants_pre: Scipy interpolant for outside of Atlantic/Arctic
    """

    import numpy as np
    from scipy.spatial import Delaunay
    import scipy.interpolate

    # Obtain data from the dictionaries
    Gvalues = list(Gdf.values())
    AAOvalues, ElseOvalues = list(AAdata.values()), list(Elsedata.values())

    def process_grid(grid_values, data_values):

        """
        A functionto help process data from grid and user data for interpolations
            and interpolate based upon a Delaunay triangulation, using scipy's
            LinearNDInterpolator
        """

        results = []
        for i in range(len(grid_values)):
            grid = grid_values[i]
            points = np.array([list(grid['lon']), list(grid['lat']), list(grid['d2d'])]).T
            tri = Delaunay(points)

            values = np.array([
                list(grid['C_alpha']),
                list(grid['C_S']),
                list(grid['C_T']),
                list(grid['C_A']),
                list(grid['C_B']),
                list(grid['C_C'])
            ]).T
            interpolant = scipy.interpolate.LinearNDInterpolator(tri, values)

            data = data_values[i]
            points_to_interpolate = (list(data['Longitude']), list(data['Latitude']), list(data['d2d']))
            results.append(interpolant(points_to_interpolate))
           
        return results, interpolant
            
    # Process AA (Atlantic/Arctic) and Else grids
    aaLCs, aaInterpolants_pre = process_grid(Gvalues, AAOvalues)
    elLCs, elInterpolants_pre = process_grid(Gvalues, ElseOvalues)
        
    return aaLCs, aaInterpolants_pre, elLCs, elInterpolants_pre

