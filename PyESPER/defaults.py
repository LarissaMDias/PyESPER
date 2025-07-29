def defaults (DesiredVariables, PredictorMeasurements={}, OutputCoordinates={}, **kwargs):

    """
    Set default values and bookkeep inputs.

    Inputs: 
        DesiredVariables: List of desired output variables (user-requested)
        PredictorMeasurements: Dictionary of user-provided predictor mesasurements (salinity, etc.)
        OutputCoordinates: Dictionary of user-provided coordinates
        **kwargs: Please see README for more information

    Outputs:
        Equations: numpy array of equations (either user-defined or default)
        n: Scalar representing number of required estimates for each variable-equation comination
        e: Scalar representing number of requested equations
        p: Scalar representing number of requested output variables
        VerboseTF: Boolean read-in of whether user wants to suppress optional warnings
        C: Dictionary of processed geographic coordinates
        PerKgSwTF: Boolean representing whether user input is in molal or molar units
        MeasUncerts: Dictionary of user input measurement uncertainty values or empty 
            dictionary if not provided   
    """

    import numpy as np

    # Check and define Equations based on user-defined kwargs, or use default values
    Equations = kwargs.get("Equations", list(range(1, 17)))
    
    # Reading dimensions of user input
    n = max(len(v) for v in OutputCoordinates.values()) 
                
    # Checking kwargs for presence of VerboseTF and EstDates, and Equations, and defining defaults, as needed
    VerboseTF = kwargs.get("VerboseTF", True)
        
    # Set EstDates based on kwargs, defaulting to 2002.0 if not provided
    if "EstDates" in kwargs:
        d = np.array(kwargs["EstDates"])
        if len(d) != n:
            EstDates = np.tile(d, (n + 1, 1)).reshape(-1)
        else:
            EstDates = d
    else:
        EstDates = np.full(n, 2002.0)
        
    # Bookkeeping coordinates
    C = {}
    longitude = np.array(OutputCoordinates["longitude"])
    longitude[longitude > 360] = np.remainder(longitude[longitude > 360], 360)
    longitude[longitude < 0] = longitude[longitude<0] + 360
    C["longitude"] = longitude
    C["latitude"] = OutputCoordinates["latitude"]
    C["depth"] = OutputCoordinates["depth"]   
    
    # Defining or reading in PerKgSwTF
    PerKgSwTF = kwargs.get("PerKgSwTF", True)

    # Defining Measurement Uncertainties
    MeasUncerts = kwargs.get("MeasUncerts", {})

    # Validate MeasUncerts dimensions
    if MeasUncerts:
        if max(len(v) for v in MeasUncerts.values()) != n:
            if min(len(v) for v in MeasUncerts.values()) != 1:
                raise CustomError(
                    "MeasUncerts must be undefined, a vector with the same number of elements as "
                    "PredictorMeasurements has columns, or a matrix of identical dimension to PredictorMeasurements."
                )
        if len(MeasUncerts) != len(PredictorMeasurements):
            print("Warning: Different numbers of input uncertainties and input measurements.")

    return Equations, n, VerboseTF, EstDates, C, PerKgSwTF, MeasUncerts
