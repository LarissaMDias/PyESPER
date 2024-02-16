from PyESPER_NN.determine_AAInds_input import determine_AAInds_input
from PyESPER_NN.fetch_data import fetch_data
from PyESPER_NN.input_completion import input_completion
from PyESPER_NN.input_sanity import input_sanity
from PyESPER_NN.iterations_list import iterations_list
from PyESPER_NN.neural_nets import neural_nets
from PyESPER_NN.regional_smoothing import regional_smoothing
import numpy as np
import pandas as pd

def PyESPER_NN(DesiredVariables, OutputCoordinates={}, 
PredictorMeasurements={}, **kwargs):
    """
    Single function call to bring in all other functions and 
user-generated input. Missing data should be indicated 
        with NaN. THIS IS UPDATING THE BELOW CODE PIECE BY PIECE 
(FOLLOWING MATLAB)
 
    Input Parameters:
        OutputCoordinates (n by 3 dict): Same as input for PyESPER 
function call, dictionary with 'longitude', 'latitude', 
            and 'depth' coordinates as lists
        PredictorMeasurements (n by y dict): Same as input for PyESPER 
function call, dictionary with input parameters 
            ('salinity', 'temperature', 'phosphate', 'nitrate', 
'silicate', and 'oxygen') as lists
        PredictorMeasurements = Dictionary of input parameter measurements 
in uol/kg that will be used to estimate alkalinity, 
            including salinity (required), temperature (required if using 
O2), phosphate, nitrate, silicate, and oxygen. Names 
            are sal_in, temp_in, phosphate_in, nitrate_in, silicate_in, 
and oxygen_in, respectively.
        kwargs = see intro for more detail. Series of kwarg options for 
EstDates, Equations, MeasUncerts, SaveDataTF, 
            pHCalcTF, and PerKgSwTF
    
    Outputs* 
        OutputEstimates = nxe pandas DataFrame of estimates in ummol/kg
        UncertaintyEstimates = nxe pandas DataFrame of uncertainty 
estimates specific to coordinates
        CoefficientsUsed = nx6xe pandas DataFrame of coefficients specific 
to coordinates and provided as inputs
        
        *Option to save results to .csv in user-defined directory by 
deleting # at end of code. 
    """
   
    if "VerboseTF" in kwargs:
        if kwargs.get('VerboseTF') is False:
            VerboseTF = False
    else:
        VerboseTF = True
    
    tic = time.perf_counter() # starting the timer  
    
    input_completion(OutputCoordinates, PredictorMeasurements) # checking 
for all required inputs
    input_sanity(OutputCoordinates, PredictorMeasurements) # printing 
warnings as in MATLAB version
    
    Equations = []
    if "Equations" in kwargs: # formatting equations as in MATLAB version        
        for v in kwargs.get('Equations'):
            Equations.append(v)
    else:
        Equations = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 
16]
                  
    n = max(len(v) for v in OutputCoordinates.values()) # number of rows 
out
    e = len(Equations) # number of Equations
    p = len(DesiredVariables) # number of Variables 
    
    class CustomError(Exception):
            pass

    if "MeasUncerts" in kwargs:  
        
        if max(len(v) for v in MeasUncerts.values()) != n:
            if min(len(v) for v in MeasUncerts.values()) != 1: 
                raise CustomError('MeasUncerts must be undefined, a vector 
with the same number of elements as \
                PredictorMeasurements has columns, or a matrix of 
identical dimension to PredictorMeasurements.')
        if len(MeasUncerts) != len(PredictorMeasurements):
            print('Warning: There are different numbers of columns of 
input uncertainties and input measurements.')
               
        if 'sal_u' in MeasUncerts:
            sal_u = np.array(MeasUncerts['sal_u'])
            sal_u = np.tile(sal_u, n)
        else:
            sal_u = np.tile(0.003, n)

        if 'temp_u' in MeasUncerts:
            temp_u = np.array(MeasUncerts['temp_u'])
            temp_u = np.tile(temp_u, n)
        else:
            if 'temperature' in PredictorMeasurements:
                temp_u = np.tile(0.003, n)
            else:
                temp_u = np.tile('nan', n)
        
        def process_uncert(param, default_factor):
            if f"{param}_u" in MeasUncerts:
                result = np.tile(MeasUncerts[f"{param}_u"], n)
            else:
                if param in PredictorMeasurements:
                    result = np.array([i * default_factor for i in 
PredictorMeasurements[param]])
                else: 
                    result = np.tile('nan', n)
            MeasUncerts[f"{param}_u"] = result
        
        process_uncert('phosphate', 0.02)
        phosphate_u = np.array(MeasUncerts['phosphate_u'])
        process_uncert('nitrate', 0.02)
        nitrate_u = np.array(MeasUncerts['nitrate_u'])
        process_uncert('silicate', 0.02)
        silicate_u = np.array(MeasUncerts['silicate_u'])
        process_uncert('oxygen', 0.01)   
        oxygen_u = np.array(MeasUncerts['oxygen_u'])
                
    else:
        MeasUncerts = {}
        sal_u = np.tile(0.003, n)
        if 'temperature' in PredictorMeasurements:
            temp_u = np.tile(0.003, n)
        else: 
            temp_u = np.tile('nan', n)
        if 'phosphate' in PredictorMeasurements:
            phosphate_u = np.array([i * 0.02 for i in 
PredictorMeasurements['phosphate']])
        else:
            phosphate_u = np.tile('nan', n)
        if 'nitrate' in PredictorMeasurements:
            nitrate_u = np.array([i * 0.02 for i in 
PredictorMeasurements['nitrate']])
        else: 
            nitrate_u = np.tile('nan', n)
        if 'silicate' in PredictorMeasurements:
            silicate_u = np.array([i * 0.02 for i in 
PredictorMeasurements['silicate']])
        else:
            silicate_u = np.tile('nan', n)
        if 'oxygen' in PredictorMeasurements:
            oxygen_u = np.array([i * 0.01 for i in 
PredictorMeasurements['oxygen']])
        else:
            oxygen_u = np.tile('nan', n)  
    MeasUncerts['sal_u'] = sal_u
    MeasUncerts['temp_u'] = temp_u
    MeasUncerts['phosphate_u'] = phosphate_u
    MeasUncerts['nitrate_u'] = nitrate_u
    MeasUncerts['silicate_u'] = silicate_u
    MeasUncerts['oxygen_u'] = oxygen_u

       
    keys = ["sal_u", "temp_u", "phosphate_u", "nitrate_u", "silicate_u", 
"oxygen_u"]
    Uncerts = np.column_stack((sal_u, temp_u, phosphate_u, nitrate_u, 
silicate_u, oxygen_u))
    Uncertainties = [pd.DataFrame(Uncerts, columns=keys)]

    longitude = np.array(OutputCoordinates['longitude'])
    longitude[longitude > 360] = np.remainder(longitude[longitude > 360], 
360)
    longitude[longitude < 0] = longitude[longitude<0] + 360
    OutputCoordinates['longitude'] = longitude
    
    if 'EstDates' in kwargs:
        d = np.array(kwargs.get('EstDates'))
        if len(d) != n:
            Dates=[kwargs.get('EstDates') for x in range(n)]
            EstDates = [element for sublist in Dates for element in 
sublist]
        else:
            EstDates = list(kwargs.get('EstDates'))
    else:
        EstDates = list(2002.0 for x in range(n))
    
    order = list(range(n))
    input_data = {
        'Order': order,
        'Longitude': OutputCoordinates['longitude'],
        'Latitude': OutputCoordinates['latitude'],
        'Depth': OutputCoordinates['depth'],
        'Salinity': PredictorMeasurements['salinity'],
        'Dates': EstDates,
        'Salinity_u': sal_u,
        'Temperature_u': temp_u,
        'Phosphate_u': phosphate_u,
        'Nitrate_u': nitrate_u,
        'Silicate_u': silicate_u,
        'Oxygen_u': oxygen_u
    }
    
    if 'temperature' in PredictorMeasurements:
        input_data['Temperature'] = PredictorMeasurements['temperature']
    if 'phosphate' in PredictorMeasurements:
        input_data['Phosphate'] = PredictorMeasurements['phosphate']
    if 'nitrate' in PredictorMeasurements:
        input_data['Nitrate'] = PredictorMeasurements['nitrate']
    if 'silicate' in PredictorMeasurements:
        input_data['Silicate'] = PredictorMeasurements['silicate']
    if 'oxygen' in PredictorMeasurements:
        input_data['Oxygen'] = PredictorMeasurements['oxygen']

    InputAll = pd.DataFrame(input_data).dropna(subset=['Longitude', 
'Latitude', 'Depth'])
    # created a dataframe with order stamp and dropped all nans from a 
replicate dataframe

    if "PerKgSwTF" in kwargs:
        if kwargs.get('PerKgSwTF') is False:
            PerKgSwTF = False
    else:
        PerKgSwTF = True
        
    if "EstDates" in kwargs:
        if 'pH' in DesiredVariables:
            if 'temperature' not in PredictorMeasurements:
                print('Warning: Carbonate system calculations will be used 
to adjust the pH, but no temperature is \
                    specified so 10 C will be assumed. If this is a poor 
estimate for your region, consider supplying \
                    your own value in the PredictorMeasurements input.')           
                Temperature = []
                for i in range(0, n):
                    Temperature.append(10)
            else:
                Temperature = InputAll['Temperature']
            PredictorMeasurements['temperature'] = Temperature
            InputAll['temperature'] = Temperature
            
    need, code = iterations_list(DesiredVariables, Equations, PerKgSwTF, 
InputAll)

    NN_data = fetch_data(DesiredVariables)
    
    df = determine_AAInds_input(InputAll)
    EstAtl, EstOther = neuralnets(DesiredVariables, Equations, code)

    for i in code:
        code[i]['AAInds'] = df['AAInds']
        code[i]['BeringInds'] = df['BeringInds']
        code[i]['SAtlInds'] = df['SAtlInds']
        code[i]['SoAfrInds'] = df['SoAfrInds']
    Est = regional_smoothing(Equations, EstAtl, EstOther, code)
    
    return Est
