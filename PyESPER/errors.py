def errors(OutputCoordinates={}, PredictorMeasurements={}):
    """
    Custom error messages for PyESPER that check inputs and ensure that
    formatting and other requirements are met. Mostly custom errors and
    warnings.
    """
    import numpy as np
    import warnings

    class CustomError(Exception):
        pass

    required_coords = ("longitude", "latitude", "depth")
    for coord_name in required_coords:
        if coord_name not in OutputCoordinates:
            raise CustomError(f"Missing {coord_name} in OutputCoordinates.")

    if "salinity" not in PredictorMeasurements:
        raise CustomError(
            "Error: Missing salinity measurements. Salinity is a required input."
        )

    if (
        "oxygen" in PredictorMeasurements
        and "temperature" not in PredictorMeasurements
    ):
        raise CustomError(
            "Error: Missing temperature measurements. Temperature is required when oxygen is provided."
        )

    if "temperature" in PredictorMeasurements:
        temp_arr = np.asarray(PredictorMeasurements["temperature"])
        if np.any((temp_arr < -5) | (temp_arr > 50)):
            warnings.warn(
                "Temperatures below -5°C or above 50°C found. PyESPER is not designed for seawater with these properties. Ensure temperatures are in Celsius."
            )

    sal_arr = np.asarray(PredictorMeasurements["salinity"])
    if np.any((sal_arr < 5) | (sal_arr > 50)):
        warnings.warn(
            "Salinities less than 5 or greater than 50 have been found. ESPER is not intended for seawater with these properties."
        )

    depth_arr = np.asarray(OutputCoordinates["depth"])
    if np.any(depth_arr < 0):
        warnings.warn("Depth cannot be negative.")

    if np.any(np.isnan(depth_arr)):
        warnings.warn("Depth cannot be nan.")

    lat_arr = np.asarray(OutputCoordinates["latitude"])
    if np.any(np.abs(lat_arr) > 90):
        warnings.warn(
            "A latitude >90 deg (N or S) has been detected. Verify latitude is entered correctly as an input."
        )

    missing_flags = np.array([-9999, -99, -1e20])
    if np.any(np.isin(lat_arr, missing_flags)):
        warnings.warn(
            "A common non-NaN missing data indicator (e.g., -999, -9, -1e20) was detected in the input measurements provided. Missing data should be replaced with NaNs. Otherwise, ESPER will interpret your inputs at face value and give terrible estimates."
        )

    warnings.warn(
        "Please note that, for consistency with MATLAB ESPERv1, the now-deprecated seawater package is used. This will be replaced with gsw in future updates.",
        PendingDeprecationWarning,
    )
