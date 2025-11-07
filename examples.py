# %% EXAMPLE

# Basic example first

from PyESPER.lir import lir
from PyESPER.nn import nn
from PyESPER.mixed import mixed

import numpy as np
import scipy.io

OutputCoordinates = {
    'longitude': [0, -140, 67],
    'latitude': [0, -30, -40],
    'depth': [0, 1000, 100]
}

PredictorMeasurements = {
    'salinity': [35, 33, 32],
    'temperature': [5, 2, 0],
    'phosphate': [0.5, 0.6, 0.4],
    'nitrate': [9, 8, 10],
    'silicate': [21, 20, 19],
    'oxygen': [198, 215, 200]
}


EstDates = [1980, 2002, 2030]

Path = ""

# To estimate TA with equation 2 using only LIR
EstimatesLIR1, CoefficientsLIR1, UncertaintiesLIR1 = lir(
     ['TA'],
     Path,
     OutputCoordinates,
     PredictorMeasurements,
     EstDates=EstDates,
     Equations=[2]
)

# To estimate in situ pH on the total scale 
# with equation 5 using only NN
EstimatesNN1, UncertaintiesNN1 = nn(
     ['pH'],
     Path,
     OutputCoordinates,
     PredictorMeasurements,
     EstDates=EstDates,
     Equations=[5]
)

# To estimate phosphate and nitrate using both methods
# with equations 1 and 16
EstimatesMixed1, UncertaintiesMixed1 = mixed(
    ['phosphate', 'nitrate'],
    Path,
    OutputCoordinates,
    PredictorMeasurements,
    EstDates=EstDates,
    Equations=[1, 16]
)

# DEBUG 
print(EstimatesLIR1['TA2'])
print(UncertaintiesNN1['pH5'])
print(EstimatesMixed1['phosphate16'])
print(UncertaintiesMixed1['nitrate1'])

# More advanced example uses the glodap package 
# from https://github.com/mvdh7/glodap (see for
# instructions)
import glodap

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
# measurement uncertainties
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
