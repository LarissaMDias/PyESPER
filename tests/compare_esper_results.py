"""Compare MATLAB ESPER JSON results with Python ESPER results.

Expected MATLAB input: JSON produced by save_esper_comparison.m.

Use this script after defining build_python_results() so it returns a
nested dictionary with the same scenario/method/case organization as the
MATLAB export, or adapt the small adapter in that function.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any
from PyESPER.lir import lir
from PyESPER.nn import nn
from PyESPER.mixed import mixed
import numpy as np
import re
import sys
import pandas as pd

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent

# Make the PyESPER package importable when running this file from Spyder.
sys.path.insert(0, str(REPO_ROOT))

MATLAB_JSON = TESTS_DIR / "esper_comparison_results.json"
DIFF_CSV = TESTS_DIR / "esper_comparison_differences.csv"
DIFF_JSON = TESTS_DIR / "esper_comparison_differences.json"

LIR_FILE = REPO_ROOT / "Mat_fullgrid" / "LIR_files_DIC_fullCS1.mat"

assert LIR_FILE.exists(), f"Missing LIR file: {LIR_FILE}"
assert MATLAB_JSON.exists(), f"Missing MATLAB JSON file: {MATLAB_JSON}"

# MATLAB result-field names -> common comparison labels.
MATLAB_CASE_MAP = {
    "equation_7LIR": ("LIR", "equation_7"),
    "equations_1_to_16LIR": ("LIR", "equations_1_to_16"),
    "equation_7NN": ("NN", "equation_7"),
    "equations_1_to_16NN": ("NN", "equations_1_to_16"),
    "equation_7Mixed": ("Mixed", "equation_7"),
    "equations_1_to_16Mixed": ("Mixed", "equations_1_to_16"),
}

# Python result-key labels -> common comparison labels.
PYTHON_CASE_MAP = {
    "equation_7_LIR": ("LIR", "equation_7"),
    "equations_1_to_16_LIR": ("LIR", "equations_1_to_16"),
    "equation_7_NN": ("NN", "equation_7"),
    "equations_1_to_16_NN": ("NN", "equations_1_to_16"),
    "equation_7_Mixed": ("Mixed", "equation_7"),
    "equations_1_to_16_Mixed": ("Mixed", "equations_1_to_16"),
}


def load_json(filename: str | Path) -> dict[str, Any]:
    with Path(filename).open(encoding="utf-8") as f:
        return json.load(f)


def build_python_results() -> dict[tuple[str, str], Any]:
    """Return Python results keyed by (scenario, case).

    Replace this body with the loop you already use to call lir(), nn(), and
    mixed(). Each value should be the complete function return tuple, e.g.
    (estimates, uncertainties) or (estimates, coefficients, uncertainties).
    """
    
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
    
    return results

def to_array(value: Any) -> np.ndarray:
    """Convert MATLAB-JSON/Python values, including nested dicts, to arrays."""
    if isinstance(value, dict):
        # MATLAB struct fields are handled outside this function; this gives
        # a useful error if a whole result struct is passed accidentally.
        raise TypeError(f"Expected numeric data, got fields: {list(value)}")
    return np.asarray(value, dtype=float)


def extract_estimates(result: Any) -> Any:
    """Extract estimates from a Python return tuple or direct estimate value."""
    if isinstance(result, tuple):
        return result[0]
    return result


def extract_named_estimates(value: Any, case: str) -> dict[str, np.ndarray]:
    """Extract estimates while preserving equation identity."""

    if not isinstance(value, dict):
        raise TypeError(
            "Expected estimates to be a dictionary keyed by variable name."
        )

    output = {}

    for name, data in value.items():
        name = str(name)

        if case == "equation_7":
            # Python keys may be DIC7, TA7, pH7, etc.
            if name.endswith("7"):
                variable = name[:-1]
                output[variable] = to_array(data)

        elif case == "equations_1_to_16":
            # Do not strip equation numbers here.
            output[name] = to_array(data)

    return output


def matlab_estimate_fields(case_data: dict[str, Any]) -> dict[str, np.ndarray]:
    estimates = case_data["estimates"]
    if not isinstance(estimates, dict):
        raise TypeError("MATLAB JSON estimates are not a struct/object.")
    return {name: to_array(value) for name, value in estimates.items()}


def compare_arrays(matlab: np.ndarray, python: np.ndarray) -> dict[str, Any]:
    if matlab.shape != python.shape:
        return {
            "status": "shape_mismatch",
            "matlab_shape": list(matlab.shape),
            "python_shape": list(python.shape),
        }

    valid = np.isfinite(matlab) & np.isfinite(python)
    equal_nan = np.isnan(matlab) & np.isnan(python)
    comparable = valid | equal_nan
    differences = np.full(matlab.shape, np.nan, dtype=float)
    differences[valid] = matlab[valid] - python[valid]

    if np.any(valid):
        signed_diff = differences[valid]
        abs_diff = np.abs(signed_diff)
        mean_difference = float(np.mean(signed_diff))
        min_difference = float(np.min(signed_diff))
        max_difference = float(np.max(signed_diff))
        mean_abs = float(np.mean(abs_diff))
        rmse = float(np.sqrt(np.mean(signed_diff ** 2)))
    else:
        mean_difference = min_difference = max_difference = float("nan")
        mean_abs = rmse = float("nan")

    return {
        "status": "ok",
        "shape": list(matlab.shape),
        "n_comparable": int(np.count_nonzero(comparable)),
        "n_nan_mismatch": int(np.count_nonzero(np.isnan(matlab) ^ np.isnan(python))),
        # Differences are MATLAB - Python.
        "mean_difference": mean_difference,
        "bias": mean_difference,
        "min_difference": min_difference,
        "max_difference": max_difference,
        "mean_abs_difference": mean_abs,
        "rmse": rmse,
        "matlab_values": matlab.tolist(),
        "python_values": python.tolist(),
        "difference_matlab_minus_python": differences.tolist(),
    }



def main() -> None:
    matlab_data = load_json(MATLAB_JSON)
    python_results = build_python_results()
    comparison: list[dict[str, Any]] = []

    matlab_scenarios = matlab_data["results"]
    for scenario, scenario_data in matlab_scenarios.items():
        for matlab_field, (method, case) in MATLAB_CASE_MAP.items():
            if matlab_field not in scenario_data:
                continue

            python_key = (scenario, f"{case}_{method}")
            if python_key not in python_results:
                comparison.append({
                    "scenario": scenario,
                    "method": method,
                    "case": case,
                    "status": "missing_python_result",
                })
                continue

            matlab_vars = matlab_estimate_fields(scenario_data[matlab_field])
            python_vars = extract_named_estimates(
                extract_estimates(python_results[python_key]),
                case,
            )
            for variable, matlab_values in matlab_vars.items():
                if variable not in python_vars:
                    comparison.append({
                        "scenario": scenario,
                        "method": method,
                        "case": case,
                        "variable": variable,
                        "status": "missing_python_variable",
                    })
                    continue

                summary = compare_arrays(matlab_values, python_vars[variable])
                comparison.append({
                    "scenario": scenario,
                    "method": method,
                    "case": case,
                    "variable": variable,
                    **summary,
                })

    Path(DIFF_JSON).write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    fieldnames = [
        "scenario", "method", "case", "variable", "status", "shape",
        "n_comparable", "n_nan_mismatch", "max_abs_difference",
        "mean_abs_difference", "rmse", "matlab_shape", "python_shape",
    ]
    with Path(DIFF_CSV).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(comparison)

    summary = pd.DataFrame(comparison)

    summary = summary.loc[
        summary["status"].eq("ok"),
        [
            "scenario",
            "method",
            "case",
            "variable",
            "mean_difference",
            "min_difference",
            "max_difference",
            "bias",
            "rmse",
        ],
    ]

    print("\nDifference summary:")
    print(summary.to_string(index=False))

    print(f"Compared {len(comparison)} variable/case/scenario combinations.")
    print(f"Wrote {DIFF_CSV}")
    print(f"Wrote {DIFF_JSON}")


if __name__ == "__main__":
    main()
