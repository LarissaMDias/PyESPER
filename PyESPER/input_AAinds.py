import numpy as np
from matplotlib.path import Path


def _polygons():
    """The ESPERv1 Atlantic/Arctic polygons, in (longitude, latitude) degrees."""
    LNAPoly = np.array(
        [
            [300.0, 0.0],
            [260.0, 20.0],
            [240.0, 67.0],
            [260.0, 40.0],
            [361.0, 40.0],
            [361.0, 0.0],
            [298.0, 0.0],
        ]
    )
    LSAPoly = np.array(
        [
            [298.0, 0.0],
            [292.0, -40.01],
            [361.0, -40.01],
            [361.0, 0.0],
            [298.0, 0.0],
        ]
    )
    LNAPolyExtra = np.array(
        [[-1.0, 50.0], [40.0, 50.0], [40.0, 0.0], [-1.0, 0.0], [-1.0, 50.0]]
    )
    LSAPolyExtra = np.array(
        [[-1.0, 0.0], [20.0, 0.0], [20.0, -40.0], [-1.0, -40.0], [-1.0, 0.0]]
    )
    LNOPoly = np.array(
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
    )
    xtra = np.array(
        [[0.5, -39.9], [0.99, -39.9], [0.99, -40.001], [0.5, -40.001]]
    )

    return [LNAPoly, LSAPoly, LNAPolyExtra, LSAPolyExtra, LNOPoly, xtra]


def atlantic_mask(longitude, latitude):
    """Boolean mask: True where a point falls in the ESPERv1 Atlantic/Arctic region.

    Split out of :func:`input_AAinds` so callers that only need the classification --
    ``lir()`` selects each point's coefficient set from it, rather than physically
    splitting every input column in two -- do not have to build the split dicts.
    """
    longitude_array = np.asarray(longitude, dtype=np.float64) % 360.0
    latitude_array = np.asarray(latitude, dtype=np.float64)
    points = np.column_stack((longitude_array, latitude_array))

    mask = np.zeros(points.shape[0], dtype=np.bool_)
    for polygon in _polygons():
        mask |= Path(polygon).contains_points(points)
    return mask


def input_AAinds(C={}, code={}, verbose=False):
    """
    Separates user-defined inpus into Atlantic and Arctic regions or other
        regions, defined as in ESPERv1 for MATLAB.

    Inputs:
        C: Dictionary of pre-adjusted grid coordinates
        code: Dictionary of iterated equation-case scenario inputs for
            user-requested variable-equation cases

    Outputs:
        AAdata: Dictionary of code data separated for areas encompassed by the
            Atlantic and Arctic Oceans only
        Elsedata: Dictionary of code data separated for areas not encompassed by
            the Atlantic and Arctic Oceans

    ``lir()`` no longer uses this -- it classifies points with :func:`atlantic_mask` and
    lets the kernel pick a coefficient set per point -- but ``pH_adjustment`` still does.
    """
    if verbose:
        print("Classifying inputs by ocean basin.")

    aa_bool = atlantic_mask(C["longitude"], C["latitude"])
    else_bool = ~aa_bool
    aa_inds_int = aa_bool.astype(np.int8)

    keys_to_split = [
        "Latitude",
        "Longitude",
        "S",
        "T",
        "A",
        "B",
        "C",
        "Order",
        "Salinity_u",
        "Temperature_u",
        "Phosphate_u",
        "Nitrate_u",
        "Silicate_u",
        "Oxygen_u",
    ]

    AAdata = {}
    Elsedata = {}

    for eqn_name, data_dict in code.items():
        data_dict["AAInds"] = aa_inds_int

        depth_array = np.asarray(data_dict["Depth"], dtype=np.float64)
        scaled_depth = depth_array / 25.0

        aa_eq = {"d2d": scaled_depth[aa_bool], "AAInds": aa_inds_int[aa_bool]}
        else_eq = {
            "d2d": scaled_depth[else_bool],
            "AAInds": aa_inds_int[else_bool],
        }

        for key in keys_to_split:
            if key in data_dict:
                raw_array = np.asarray(data_dict[key], dtype=np.float64)
                aa_eq[key] = raw_array[aa_bool]
                else_eq[key] = raw_array[else_bool]

        AAdata[eqn_name] = aa_eq
        Elsedata[eqn_name] = else_eq

    return AAdata, Elsedata
