# %% EXAMPLE
import glodap

from PyESPER.lir import PyESPER_LIR
from PyESPER.mixed import PyESPER_Mixed
from PyESPER.nn import PyESPER_NN

data = glodap.atlantic()

L = slice(50, 1000)
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
MeasUncerts = {
    "sal_u": [0.001],
    "temp_u": [0.3],
    "phosphate_u": [0.14],
    "nitrate_u": [0.5],
    "silicate_u": [0.03],
    "oxygen_u": [0.025],
}
EstDates = data.year[L].values.tolist()
Path = ""  # paths fixed to work as relative from the location of this script

EstimatesNN, UncertaintiesNN = PyESPER_NN(
    ["DIC"],
    Path,
    OutputCoordinates,
    PredictorMeasurements,
    EstDates=EstDates,
    Equations=[1],
)
EstimatesLIR, UncertaintiesLIR, CoefficientsLIR = PyESPER_LIR(
    ["pH"],
    Path,
    OutputCoordinates,
    PredictorMeasurements,
    EstDates=EstDates,
    Equations=[1],
)
EstimatesMixed, UncertaintiesMixed = PyESPER_Mixed(
    ["phosphate", "TA"],
    Path,
    OutputCoordinates,
    PredictorMeasurements,
    VerboseTF=True,
    EstDates=EstDates,
    MeasUncerts=MeasUncerts,
    Equations=[1, 2],
)
print(EstimatesNN[90:100], UncertaintiesLIR[10:20], EstimatesMixed[0:10])
