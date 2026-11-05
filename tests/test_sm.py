# smoke test for PR#13
# validating against GLODAP and MATLAB ESPERs
"""
Fix string short-circuit in DesiredVariables checks
Submitted by CatManJr 
Aug 26, 2026

Summary: (“pH” or “DIC”) in DesiredVariables short-circuits to “pH” in 
          DesiredVariables, so DIC-only requests silently return the 
          unadjusted (2002-baseline) DIC even though pH_DIC_nn_adjustment 
          computed the Cant - Cant2002 adjustment. The same short-circuit 
          pattern in the no-EstDates warning branches (pH_DIC_NN_adjustment.py, 
          pH_adjustment.py) makes those warnings fire even when neither DIC nor 
          pH is requested.
        
Fix: Check both variables explicitly in all three conditions:
    
PyESPER/final_formatting.py: “pH” in DesiredVariables or “DIC” in DesiredVariables
PyESPER/pH_DIC_nn_adjustment.py: same explicit check in the no-EstDates branch
PyESPER/pH_adjustment.py: same explicit check in the no-EstDates Branch

Verification: Reproduction with hardcoded inputs (lon -60, lat -50, depth 500, 
            S 34.2, T 3.5, O2 220, EstDates 2024, equation 7
            
Before:
    
DesiredVariables=[“DIC”, “TA”] → DIC7 = 2193.447 (unadjusted)
DesiredVariables-[“DIC”, ”TA”, “pH”] → DIC7 = 2209.569 (adjusted)

After:
    
DesiredVariables=[“DIC”, “TA”] → DIC7 = 2209.569 (adjusted)
DesiredVariables-[“DIC”, ”TA”, “pH”] → DIC7 = 2209.569 (adjusted)
	
The 16.122 delta is exactly Cant (2024) - Cant(2002)
"""

from PyESPER.lir import lir
from PyESPER.nn import nn
from PyESPER.mixed import mixed
import glodap
import pandas as pd
import numpy as np
from scipy.io import savemat

# Setting the project root
from pathlib import Path
import os

REPO_ROOT = Path(__file__).resolve().parents[1]
os.chdir(REPO_ROOT)

print("Working directory:", Path.cwd())

# First recreating the PR test
print("PR Test #13, recreating the results")

# Create the dictionary of coordinates
OutputCoordinates = {
    "longitude": [-60, -60, -60],
    "latitude": [-50, -50, -50],
    "depth": [500, 500, 500]
    }

# Create the dictionary of predictors
PredictorMeasurements = {
    "salinity": [34.2, 34.2, 34.2],
    "temperature": [3.5, 3.5, 3.5],
    "oxygen": [220, 220, 220]
    }

EstDates = [2024, 2024, 2024]

# Define the path
p = str(REPO_ROOT)

Scenario1 = ["DIC", "TA"]
Scenario2 = ["DIC", "TA", "pH"]

# Now try PyESPERs LIR
             
EstimatesLIR1, CoefficientsLIR, UncertaintiesLIR = lir(
    Scenario1, 
    p, 
    OutputCoordinates, 
    PredictorMeasurements, 
    Equation=7,
    EstDates=EstDates
    )

print('LIR Estimates for DIC-only scenario:')
print(EstimatesLIR1['DIC7']) # I got 2207.9676

EstimatesLIR2, CoefficientsLIR, UncertaintiesLIR = lir(
    Scenario2, 
    p, 
    OutputCoordinates, 
    PredictorMeasurements, 
    Equation=7,
    EstDates=EstDates
    )

print('LIR Estimates for DIC + pH scenario:')
print(EstimatesLIR2['DIC7']) # I got 2207.9676

print('MATLAB ESPER_LIR is 2211.9') # within the error for LIR-interpolation differences

# Now try PyESPERs NN
             
EstimatesNN1, UncertaintiesNN = nn(
    Scenario1, 
    p, 
    OutputCoordinates, 
    PredictorMeasurements, 
    Equation=7,
    EstDates=EstDates
    )

print('NN Estimates for DIC-only scenario:')
print(EstimatesNN1['DIC7']) # I got 2209.5694, same as PR scenario

EstimatesNN2, UncertaintiesNN = nn(
    Scenario2, 
    p, 
    OutputCoordinates, 
    PredictorMeasurements, 
    Equation=7,
    EstDates=EstDates
    )

print('NN Estimates for DIC + pH scenario:')
print(EstimatesNN2['DIC7']) # I got 2209.5694, same as PR scenario

print('MATLAB ESPER_NN is 2213.3') # within the error for LIR-interpolation differences

# Now try PyESPERs Mixed
             
EstimatesMixed1, UncertaintiesMixed = mixed(
    Scenario1, 
    p, 
    OutputCoordinates, 
    PredictorMeasurements, 
    Equation=7,
    EstDates=EstDates
    )

print('Mixed Estimates for DIC-only scenario:')
print(EstimatesMixed1['DIC7']) # I got 2208.7685

EstimatesMixed2, UncertaintiesMixed = mixed(
    Scenario2, 
    p, 
    OutputCoordinates, 
    PredictorMeasurements, 
    Equation=7,
    EstDates=EstDates
    )

print('Mixed Estimates for DIC + pH scenario:')
print(EstimatesMixed2['DIC7']) # I got 2208.7685

print('MATLAB ESPER_Mixed is 2212.6') # within the error for LIR-interpolation differences

