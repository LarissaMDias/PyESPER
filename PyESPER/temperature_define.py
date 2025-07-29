def temperature_define(
    DesiredVariables, 
    PredictorMeasurements, 
    InputAll, 
    **kwargs
):

    """ 
   A small function to define temperature as needed and adjust
        InputAll and PredictorMeasurements acccordingly.

    Inputs:
        DesiredVariables: List of desired output variables
        PredictorMeasurements: Dictionary of user input measurements for predictors
        InputAll: Dictionary of pre-adjusted and combined input and defined data
        **kwargs: Please see README for more information

    Ouputs:
        PredictorMeasurements: Dictionary of user input measurements, adjusted for temperature
            as needed
        InputAll: Combined and processed necessary data to run the LIR, adjusted now for
            temperature as needed
    """

    import numpy as np

    n = max(len(v) for v in PredictorMeasurements.values()) # Recalculating number of required estimates

    # Printing a custom warning if temperature is absent but needed
    if "EstDates" in kwargs and "pH" in DesiredVariables:
        if "temperature" not in PredictorMeasurements:
            print(
                "Warning: Carbonate system calculations will be used to adjust the pH, but no temperature is"
                "specified so 10 C will be assumed. If this is a poor estimate for your region, consider supplying"
                "your own value in the PredictorMeasurements input."
            )
            # Assuming temperature is 10 C if not defined
            Temperature = np.full(n, 10)
        else:
            # Reading in temperature if defined
            Temperature = np.array(InputAll["Temperature"])

        # Adjusting dictionaries for the updated temperature definition
        PredictorMeasurements["temperature"] = Temperature
        InputAll["temperature"] = Temperature

    return PredictorMeasurements, InputAll
    
