import numpy as np
from scipy.interpolate import RegularGridInterpolator, NearestNDInterpolator
import matplotlib
import warnings

from PyESPER.kernels import grid_cache


def build_interpolants(Gdf):
    """Build the two region interpolants for one set of (variable, equation) grids.

    Point-independent: this is a pure function of the pre-trained ESPER grids and which
    combinations were requested, which is why callers memoise it (see
    PyESPER.kernels.grid_cache). Split out of ``interpolate`` so the coefficient table it
    produces can also be handed to the numba kernel in ``PyESPER.kernels.lir_forward``
    without duplicating any of this construction -- the kernel interpolates byte-
    identical data to what scipy was given.
    """
    Gvalues = list(Gdf.values())
    grid = Gvalues[0]

    lon_array = np.asarray(grid["lon"], dtype=np.float64) % 360.0
    lat_array = np.asarray(grid["lat"], dtype=np.float64)
    grid_points = np.column_stack((lon_array, lat_array))

    polygons = [
        np.array(
            [
                [300.0, 0.0],
                [260.0, 20.0],
                [240.0, 67.0],
                [260.0, 40.0],
                [361.0, 40.0],
                [361.0, 0.0],
                [298.0, 0.0],
            ]
        ),
        np.array(
            [
                [298.0, 0.0],
                [292.0, -40.01],
                [361.0, -40.01],
                [361.0, 0.0],
                [298.0, 0.0],
            ]
        ),
        np.array(
            [
                [-1.0, 50.0],
                [40.0, 50.0],
                [40.0, 0.0],
                [-1.0, 0.0],
                [-1.0, 50.0],
            ]
        ),
        np.array(
            [
                [-1.0, 0.0],
                [20.0, 0.0],
                [20.0, -40.0],
                [-1.0, -40.0],
                [-1.0, 0.0],
            ]
        ),
        np.array(
            [
                [361.0, 40.0],
                [361.0, 91.0],
                [-1.0, 91.0],
                [-1.0, 50.0],
                [40.0, 50.0],
                [40.0, 40.0],
                [104.0, 40.0],
                [104.0, 67.0],
                [240.0, 67.0],
                [280.0, 40.0],
                [361.0, 40.0],
            ]
        ),
        np.array(
            [[0.5, -39.9], [0.99, -39.9], [0.99, -40.001], [0.5, -40.001]]
        ),
    ]

    aa_bool = np.zeros(grid_points.shape[0], dtype=np.bool_)
    for poly in polygons:
        aa_bool |= matplotlib.path.Path(poly).contains_points(grid_points)

    else_bool = ~aa_bool

    u_lon = np.unique(grid["lon"])
    u_lat = np.unique(grid["lat"])
    u_depth = np.unique(grid["d2d"])

    # probably doesn't do anything, but may prevent interp. overshoots
    # at geographic margins, like matlab's scattered interpolant algorithm
    u_lon_pad = np.concatenate(([-1e10], u_lon, [1e10]))
    u_lat_pad = np.concatenate(([-1e10], u_lat, [1e10]))
    u_depth_pad = np.concatenate(([-1e10], u_depth, [1e10]))

    cols = ["C_alpha", "C_S", "C_T", "C_A", "C_B", "C_C"]
    number_of_equations = len(Gvalues)
    number_of_coefficients = len(cols)

    # organize coords for interpolation
    points_3d = np.column_stack((grid["lon"], grid["lat"], grid["d2d"]))
    raw_values_all = np.empty(
        (len(points_3d), number_of_equations, number_of_coefficients),
        dtype=np.float64,
    )

    for i, grid_entry in enumerate(Gvalues):
        for j, name in enumerate(cols):
            data_series = grid_entry[name]
            raw_values_all[:, i, j] = (
                data_series.values
                if hasattr(data_series, "values")
                else np.asarray(data_series)
            )

    mesh_lon, mesh_lat, mesh_depth = np.meshgrid(
        u_lon, u_lat, u_depth, indexing="ij"
    )
    query_points = np.column_stack(
        (mesh_lon.ravel(), mesh_lat.ravel(), mesh_depth.ravel())
    )

    def build_interpolant(mask):
        # prevent polygons' coefs from contaminating one another
        # this makes a significant difference for marginal seas
        # or thin landmasses
        if not np.any(mask):
            return None

        masked_points = points_3d[mask]
        masked_values = raw_values_all[mask]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_vals = np.nanmean(masked_values, axis=0)

        # nearest interp is only for initializing the grid
        # and preventing first interpolation from contaminating second
        nearest_interp = NearestNDInterpolator(masked_points, masked_values)
        filled_flat = nearest_interp(query_points)

        # make sure grid has correct dim order
        filled_3d = filled_flat.reshape(
            (
                len(u_lon),
                len(u_lat),
                len(u_depth),
                number_of_equations,
                number_of_coefficients,
            )
        )

        pad_shape = (
            len(u_lon_pad),
            len(u_lat_pad),
            len(u_depth_pad),
            number_of_equations,
            number_of_coefficients,
        )
        pad_arr = np.empty(pad_shape, dtype=np.float64)
        pad_arr[...] = mean_vals
        pad_arr[1:-1, 1:-1, 1:-1, :, :] = filled_3d

        # regular grid instead of scattered grid performs well and much faster
        return RegularGridInterpolator(
            (u_lon_pad, u_lat_pad, u_depth_pad),
            pad_arr,
            bounds_error=False,
            fill_value=np.nan,
        )

    return build_interpolant(aa_bool), build_interpolant(else_bool)


def interpolate(Path, Gdf={}, AAdata={}, Elsedata={}, verbose=False):
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

    Retained for callers that still want scipy-evaluated coefficients split by region.
    ``lir()`` no longer goes through here: it calls
    ``PyESPER.kernels.lir_forward.lir_estimates``, which evaluates the same interpolants
    with a numba kernel and skips the region split entirely.
    """
    if verbose:
        print("Performing local interpolation.")

    interp_aa, interp_else = grid_cache.interpolants(
        Path, Gdf, lambda: build_interpolants(Gdf)
    )
    number_of_equations = len(Gdf)

    def process_grid(data_values, interpolant):
        """
        A function to help process data from grid and user data for interpolations
            and interpolate based upon gridded interpolation instead of the
            original scattered interpolation via linearndinterpolator
        """
        if not data_values or interpolant is None:
            return [], interpolant
        first_key = list(data_values.keys())[0]

        longitude_array = data_values[first_key]["Longitude"]
        latitude_array = data_values[first_key]["Latitude"]
        depth_array = data_values[first_key]["d2d"]

        points_to_interpolate = np.column_stack(
            [longitude_array, latitude_array, depth_array]
        )
        raw_results = interpolant(points_to_interpolate)
        results = [raw_results[:, i, :] for i in range(number_of_equations)]

        return results, interpolant

    aaLCs, aaInterpolants_pre = process_grid(AAdata, interp_aa)
    elLCs, elInterpolants_pre = process_grid(Elsedata, interp_else)

    return aaLCs, aaInterpolants_pre, elLCs, elInterpolants_pre
