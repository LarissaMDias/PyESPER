# smoke test for PR#13
# validating against GLODAP and MATLAB ESPERs
"""
Testing ESPER results
"""

from PyESPER.lir import lir
from PyESPER.nn import nn
from PyESPER.mixed import mixed
import glodap
import pandas as pd
import numpy as np
from scipy.io import savemat
import json

# Setting the project root
from pathlib import Path
import os

REPO_ROOT = Path(__file__).resolve().parents[1]
os.chdir(REPO_ROOT)

print("Working directory:", Path.cwd())

def test_small():

    # Create a dictionary of simulated OutputCoordinates
    OutputCoordinates = {
        "longitude": [-60, 145, -150, 20, -35, 115, -90, 165, 60, -10],
        "latitude":  [-50, -30,  25, -35,  20, -45,  10, -20, -25,  45],
        "depth":     [500,  800, 1200, 700, 1500, 900, 600, 2000, 1100, 400]
    }

    # Create a dictionary of simulated PredictorMeasurements
    PredictorMeasurements = {
        "salinity": [
            34.2, 35.1, 35.8, 34.7, 36.1,
            33.9, 35.5, 34.4, 36.3, 35.0
        ],

        "temperature": [
            3.5, 8.2, 18.6, 5.1, 22.4,
            2.3, 12.7, 6.8, 1.9, 15.4
        ],

        "oxygen": [
            220, 248, 198, 275, 165,
            235, 210, 285, 145, 260
        ],

        "nitrate": [
            25.0, 12.4, 1.8, 31.5, 0.6,
            34.2, 8.7, 22.1, 38.0, 4.3
        ],

        "silicate": [
            80.0, 42.5, 5.2, 96.0, 2.1,
            115.0, 18.4, 67.3, 125.0, 9.6
        ],

        "phosphate": [
            2.0, 1.1, 0.15, 2.5, 0.05,
            2.9, 0.7, 1.8, 3.1, 0.3
        ]
    }

    EstDates = [2000, 2024, 2019, 2016, 2022, 2021, 2019, 2014, 2004, 2017]

    # Define the path
    p = str(REPO_ROOT)

    scenarios = {
        "Scenario1": ["DIC", "TA"],
        "Scenario2": ["DIC", "TA", "pH"],
        "Scenario3": ["pH"],
        "Scenario4": ["oxygen"],
        "Scenario5": ["TA", "DIC", "pH", "nitrate", "phosphate", "silicate", "oxygen"],
    }

    results = {}

    for scenario_name, scenario in scenarios.items():
        # First: test equation 7 independently as a scalar.
        results[(scenario_name, "equation_7_LIR")] = lir(
            scenario,
            p,
            OutputCoordinates,
            PredictorMeasurements,
            Equation=7,
            EstDates=EstDates,
        )

        # Second: pass equations 1 through 16 simultaneously as a list.
        results[(scenario_name, "equations_1_to_16_LIR")] = lir(
            scenario,
            p,
            OutputCoordinates,
            PredictorMeasurements,
            Equation=list(range(1, 17)),
            EstDates=EstDates,
        )
        
        results[(scenario_name, "equation_7_NN")] = nn(
            scenario,
            p,
            OutputCoordinates,
            PredictorMeasurements,
            Equation=7,
            EstDates=EstDates,
        )
        
        results[(scenario_name, "equations_1_to_16_NN")] = nn(
             scenario,
             p,
             OutputCoordinates,
             PredictorMeasurements,
             Equation=list(range(1, 17)),
             EstDates=EstDates,
        )
        
        results[(scenario_name, "equation_7_Mixed")] = mixed(
            scenario,
            p,
            OutputCoordinates,
            PredictorMeasurements,
            Equation=7,
            EstDates=EstDates,
        )
        
        results[(scenario_name, "equations_1_to_16_Mixed")] = mixed(
             scenario,
             p,
             OutputCoordinates,
             PredictorMeasurements,
             Equation=list(range(1, 17)),
             EstDates=EstDates,
        )

    expected_results = len(scenarios) * 2 * 3
    assert len(results) == expected_results


    NNEstimatesScenario3, NNUncertaintiesScenario3 = results[
        ("Scenario3", "equation_7_NN")
    ]
    print("pH equation 7:", NNEstimatesScenario3["pH7"])
    
    MixedEstimatesAll, MixedUncertaintiesAll = results[
        ("Scenario3", "equations_1_to_16_Mixed")
    ]
    print("pH equation 7 from simultaneous run:", MixedEstimatesAll["pH7"])
    
    with open("esper_comparison_results.json", encoding="utf-8") as f:
        data = json.load(f)

    print(data["results"].keys())

    matlab_pH = np.asarray(
        data["results"]["Scenario3"]["equation_7NN"]["estimates"]["pH"],
        dtype=float,
    )

    print(matlab_pH)

test_small()