def errors(OutputCoordinates={}, PredictorMeasurements={}):

    """
    Custom error messages for PyESPER that check inputs and ensure that formatting and other requirements are met
    """

    # Checking for presence of required input parameters and raising a custom error message if needed
    class CustomError(Exception):
        pass

    required_coords = ("longitude", "latitude", "depth")
    for coord_name in required_coords:
        if coord_name not in OutputCoordinates:
            raise CustomError(f"Warning: Missing {coord_name} in OutputCoordinates.")

    if "salinity" not in PredictorMeasurements: 
        raise CustomError("Warning: Missing salinity measurements. Salinity is a required input.")
            
    if "oxygen" in PredictorMeasurements and "temperature" not in PredictorMeasurements:
        raise CustomError("Warning: Missing temperature measurements. Temperature is required when oxygen is provided.")

    # Check temperature sanity and print a warning for out-of-range values
    if "temperature" in PredictorMeasurements and any(t < -5 or t > 50 for t in PredictorMeasurements["temperature"]):
        print("Warning: Temperatures below -5°C or above 50°C found. PyESPER is not designed for seawater with these properties. Ensure temperatures are in Celsius.")
                
    if any(s < 5 or s > 50 for s in PredictorMeasurements["salinity"]):
        print("Warning: Salinities less than 5 or greater than 50 have been found. ESPER is not intended for seawater with these properties.")
        
    if any(d < 0 for d in OutputCoordinates["depth"]):
        print("Warning: Depth can not be negative.")
        
    if any(l > 90 for l in OutputCoordinates["latitude"]):
        print("Warning: A latitude >90 deg (N or S) has been detected. Verify latitude is entered correctly as an input.")
    
    # Checking for commonly used missing data indicator flags. Consider adding your commonly used flags here.  
    if any(l == -9999 or l == -9 or l == -1e20 for l in OutputCoordinates["latitude"]):
           print("Warning: A common non-NaN missing data indicator (e.g., -999, -9, -1e20) was detected in the input measurements provided. Missing data should be replaced with NaNs. Otherwise, ESPER will interpret your inputs at face value and give terrible estimates.")  
 
    print("Please note that, for consistency with MATLAB ESPERv1, the now-deprecated sw package is used. This will be replaced with gsw in future updates.")
