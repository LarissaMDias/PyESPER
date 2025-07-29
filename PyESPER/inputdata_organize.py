def inputdata_organize(
    EstDates, 
    C={}, 
    PredictorMeasurements={}, 
    Uncertainties={} 
):

    """
  This function preprocesses data into a dictionary for easier referencing

    Inputs:
        EstDates: List of preprocessed dates
        C: Dictionary of preprocessed geographic coordinates
        PredictorMeasurements: Dictionary of preprocessed inputs
        Uncertainties: Dictionary of preprocessed uncertainties

    Ouputs:
        InputAll: Dictionary of preprocessed coordinates, measurements, uncertainties, dates, and an indexing term
    """

    import numpy as np

    n = max(len(v) for v in C.values()) # Simply recalculating number of rows out

    # Redefining and organizing all data thus far, and adding an order/indexing stamp
    order = np.arange(n)
    InputAll = {
        "Order": order,
        "Longitude": np.array(C["longitude"]),
        "Latitude": np.array(C["latitude"]),
        "Depth": np.array(C["depth"]),
        "Salinity": PredictorMeasurements["salinity"],
        "Dates": EstDates,
        "Salinity_u": Uncertainties["sal_u"],
        "Temperature_u": Uncertainties["temp_u"],
        "Phosphate_u": Uncertainties["phosphate_u"],
        "Nitrate_u": Uncertainties["nitrate_u"],
        "Silicate_u": Uncertainties["silicate_u"],
        "Oxygen_u": Uncertainties["oxygen_u"]
    }

   # Map PredictorMeasurements keys to InputAll keys
    for key, label in {
        "temperature": "Temperature",
        "phosphate": "Phosphate",
        "nitrate": "Nitrate",
        "silicate": "Silicate",
        "oxygen": "Oxygen"
    }.items():
        if key in PredictorMeasurements:
            InputAll[label] = np.array(PredictorMeasurements[key])

    return InputAll



