def input_completion(OutputCoordinates={}, PredictorMeasurements={}):
    
    """
    Checks for the presence of required input parameters and raises custom 
error messages if needed.
    
    Inputs:
        OutputCoordinates (n by 3 dict): Same as input for PyESPER 
function call, dictionary with 'longitude', 'latitude', 
            and 'depth' coordinates as lists
        PredictorMeasurements (n by y dict): Same as input for PyESPER 
function call, dictionary with input parameters 
            ('salinity', 'temperature', 'phosphate', 'nitrate', 
'silicate', and 'oxygen') as lists
    
    Outputs:
        Custom error for missing but necessary input data.           
    """      
    class CustomError(Exception):
        pass
    
    required_coords = ('longitude', 'latitude', 'depth')
    for coord_name in required_coords:
        if coord_name not in OutputCoordinates:
            raise CustomError(f'Warning: Missing {coord_name} in 
OutputCoordinates.')
            
    if "salinity" not in PredictorMeasurements: 
        raise CustomError('Warning: Missing salinity measurements. 
Salinity is a required input.')
            
    if "oxygen" in PredictorMeasurements and 'temperature' not in 
PredictorMeasurements:
        raise CustomError('Warning: Missing temperature measurements. 
Temperature is required when oxygen is provided.')
