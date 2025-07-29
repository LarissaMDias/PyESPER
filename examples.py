# %% EXAMPLE
import glodap

from PyESPER.lir import lir
from PyESPER.nn import nn
from PyESPER.mixed import mixed
 
data = glodap.atlantic() 

L = slice(250, 300)
PredictorMeasurements = {
    k: data[k][L].values.tolist()
    for k in [
        "salinity",
        "temperature",
        "phosphate",
        "nitrate",
        "silicate",
        "oxygen",
    ]
}
OutputCoordinates = {
    k: data[k][L].values.tolist()
    for k in [
        "longitude",
        "latitude",
        "depth",
    ]
}

# Optional Unhash the following and customize your
# measuring uncertainties
#MeasUncerts = {
#    'sal_u': [0.001], 
#    'temp_u': [0.3], 
#    'phosphate_u': [0.14], 
#    'nitrate_u':[0.5], 
#    'silicate_u': [0.03], 
#    'oxygen_u': [0.025]
#}

EstDates = data.year[L].values.tolist()
Path = "" 
# Path works relative to the location of this script, 
# customize as needed
             
EstimatesLIR, CoefficientsLIR, UncertaintiesLIR = lir(
    ['TA'], 
    Path, 
    OutputCoordinates, 
    PredictorMeasurements, 
    EstDates=EstDates,
    Equations=[1]
    )

EstimatesNN, UncertaintiesNN = nn(
    ['DIC', 'phosphate'], 
    Path, 
    OutputCoordinates, 
    PredictorMeasurements, 
    EstDates=EstDates
    )

EstimatesMixed, UncertaintiesMixed = mixed(
    ['pH'], 
    Path,
    OutputCoordinates, 
    PredictorMeasurements,
    EstDates=EstDates,
    Equations=[15]
    )

# DEBUG, unhash as needed
print(EstimatesLIR['TA1']) 
#print(CoefficientsLIR['TA1']['Coef A'][5:10])
#print(EstimatesNN['DIC1'])
#print(UncertaintiesNN['phosphate1'])
#print(UncertaintiesLIR['TA1'])

# Optional format to pandas and save
#import pandas as pd
#(pd.DataFrame.from_dict(data=EstimatesLIR, orient='index')
#    .to_csv('TA1LIR.csv'))
#(pd.DataFrame.from_dict(data=EstimatesNN, orient='index')
#    .to_csv('EstNN.csv'))
