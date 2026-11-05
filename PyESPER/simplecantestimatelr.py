import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from pathlib import Path

_interpolator_cache = {}
_DATA_FILENAME = "SimpleCantEstimateLR_full.csv"


def _resolve_data_file(data_path=None):
    package_file = Path(__file__).resolve().parent / _DATA_FILENAME

    if data_path is None:
        data_file = package_file
    else:
        supplied = Path(data_path).expanduser()
        if supplied.is_file():
            data_file = supplied
        elif supplied.is_dir():
            candidates = (
                supplied / _DATA_FILENAME,
                supplied / "PyESPER" / _DATA_FILENAME,
                package_file,
            )
            data_file = next((p for p in candidates if p.is_file()), None)
            if data_file is None:
                raise FileNotFoundError(
                    f"Could not find {_DATA_FILENAME!r} under: "
                    + ", ".join(str(p) for p in candidates)
                )
        else:
            data_file = supplied

    if not data_file.is_file():
        raise FileNotFoundError(f"SimpleCant data file not found: {data_file}")

    return data_file.resolve()


def simplecantestimatelr(EstDates, longitude, latitude, depth, data_path=None):
    data_file = _resolve_data_file(data_path)
    cache_key = str(data_file)

    if cache_key not in _interpolator_cache:
        CantIntPoints = pd.read_csv(data_file)

        u_lon, lon_idx = np.unique(
            CantIntPoints["Int_long"], return_inverse=True
        )
        u_lat, lat_idx = np.unique(
            CantIntPoints["Int_lat"], return_inverse=True
        )
        u_depth, depth_idx = np.unique(
            CantIntPoints["Int_depth"], return_inverse=True
        )

        grid_values = np.empty((len(u_depth), len(u_lat), len(u_lon)))
        grid_values[depth_idx, lat_idx, lon_idx] = CantIntPoints["values"]

        _interpolator_cache[cache_key] = RegularGridInterpolator(
            (u_depth * 0.025, u_lat, u_lon * 0.25),
            grid_values,
            bounds_error=False,
            fill_value=np.nan,
        )

    pointso = np.column_stack(
        (
            np.asarray(depth) * 0.025,
            np.asarray(latitude),
            np.asarray(longitude) * 0.25,
        )
    )

    Cant2002 = _interpolator_cache[cache_key](pointso)

    EstDates = np.asarray(EstDates)
    CantMeas = Cant2002 * np.exp(0.018989 * (EstDates - 2002.0))

    return CantMeas, Cant2002
