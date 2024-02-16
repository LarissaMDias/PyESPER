def input_sanity(OutputCoordinates={}, PredictorMeasurements={}): 
    
    """
    Checks the sanity of input values and prints warnings for erroneous 
input.
    
    Inputs:
        OutputCoordinates (n by 3 dict): Same as input for PyESPER 
function call, dictionary with 'longitude', 'latitude', 
            and 'depth' coordinates as lists
        PredictorMeasurements (n by y dict): Same as input for PyESPER 
function call, dictionary with input parameters 
            ('salinity', 'temperature', 'phosphate', 'nitrate', 
'silicate', and 'oxygen') as lists
    
    Outputs: 
        Custom warning message for possibly erroneous data.            
    """   
    if "temperature" in PredictorMeasurements:
        if any(t < -5 or t > 50 for t in 
PredictorMeasurements['temperature']):
               print("Warning: Temperatures less than -5 C or greater than 
100 C have been found. ESPER is not intended for \
               seawater with these properties. Note that ESPER expects 
temperatures in Centigrade.")
                
    if any(s < 5 or s > 50 for s in PredictorMeasurements['salinity']):
        print('Warning: Salinities less than 5 or greater than 50 have 
been found. ESPER is not intended for seawater with \
            these properties.')
        
    if any(d < 0 for d in OutputCoordinates['depth']):
        print('Warning: Depth can not be negative.')
        
    if any(l > 90 for l in OutputCoordinates['latitude']):
        print("Warning: A latitude >90 deg (N or S) has been detected. 
Verify latitude is entered correctly as an input.")
    
    #Checking for commonly used missing data indicator flags. Consider 
adding your commonly used flags here.  
    if any(l == -999 or l == -9 or l == -1e20 for l in 
OutputCoordinates['latitude']):
           print("Warning: A common non-NaN missing data indicator (e.g., 
-999, -9, -1e20) was detected in the input \
           measurements provided. Missing data should be replaced with 
NaNs. Otherwise, PyESPER_LIR will interpret your \
           inputs at face value and give terrible estimates.")    
