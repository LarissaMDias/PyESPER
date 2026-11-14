# PyESPER

[![Python application](https://github.com/d-sandborn/PyESPER/actions/workflows/python-app.yml/badge.svg)](https://github.com/d-sandborn/PyESPER/actions/workflows/python-app.yml)

PyESPER is a Python implementation of MATLAB Empirical Seawater Property Estimation Routines ([ESPERs](https://github.com/BRCScienceProducts/ESPER)), and the present version consists of a preliminary package which implements these routines. These routines provide estimates of seawater biogeochemical properties at user-provided sets of coordinates, depth, and available biogeochemical properties. 

This package is developed in parallel with [TRACE-Python](https://github.com/d-sandborn/TRACE).

---

## Installation

To install PyESPER, clone this repository and navigate to the PyESPER folder. It is recommended that you create a virtual environment and install all packages listed in the requirements.txt file. Open a terminal in the PyESPER directory and run

`pip install PyESPER`

to install the package and required dependencies listed in `requirements.txt`. *Note: Examples rely on the GLODAPv2.2023 dataset, which requires the separate `glodap` package.*

---

## Algorithms

PyESPER offers three algorithms to predict desired variables:

1.  **PyESPER_LIR:** Interpolated linear networks (LIRv.3 / ESPERv1.1).
2.  **PyESPER_NN:** Neural network estimations (ESPERv1.1).
3.  **PyESPER_Mixed:** An averaged ensemble of LIR and NN estimates.

### Estimation Logic

The routines calculate coefficients and intercepts across up to 16 equation combinations. The base predictors are always Salinity (S) and Temperature (T). The remaining predictors (A, B, C) shift depending on the target variable.

| Desired Variable | Predictor A | Predictor B | Predictor C |
| :--- | :--- | :--- | :--- |
| **TA**, **DIC**, **pH**, **phosphate** | Nitrate | Oxygen | Silicate |
| **nitrate** | Phosphate | Oxygen | Silicate |
| **silicate** | Phosphate | Oxygen | Nitrate |
| **oxygen** | Phosphate | Nitrate | Silicate |

**Equation Options:**

1. S, T, A, B, C
2. S, T, A, C
3. S, T, B, C
4. S, T, C
5. S, T, A, B
6. S, T, A
7. S, T, B
8. S, T
9. S, A, B, C
10. S, A, C
11. S, B, C
12. S, C
13. S, A, B
14. S, A
15. S, B
16. S

---

## Usage

```python
import PyESPER

outputs = PyESPER.mixed(
    DesiredVariables=["TA", "DIC"],
    OutputCoordinates={
        "longitude": [0.0, 180.0], 
        "latitude": [85.0, -20.0], 
        "depth": [10, 1000]
    },
    PredictorMeasurements={
        "salinity": [35.0, 34.1],
        "temperature": [0.1, 10.0],
        "oxygen": [202.3, 214.7],
        "silicate": [15.0, 45.2],
        "nitrate": [1.2, 30.5]
    },
    EstDates=[2020.5, 2020.5],
    Equations=[1, 2, 8, 16],
    PerKgSwTF=True
)
```

### Required Parameters

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `DesiredVariables` | `list[str]` | Variables to return. Options: `"TA"`, `"DIC"`, `"pH"`, `"phosphate"`, `"nitrate"`, `"silicate"`, `"oxygen"`. |
| `Path` | `str` | Absolute or relative path to the downloaded LIR `.mat` files. |
| `OutputCoordinates` | `dict` | Keys: `"longitude"` (°E), `"latitude"` (°N), `"depth"` (m). Values must be arrays of length `n`. |
| `PredictorMeasurements` | `dict` | Keys: `"salinity"`, `"temperature"`, `"phosphate"`, `"nitrate"`, `"silicate"`, `"oxygen"`. Values are arrays of length `n`. Unmeasured variables can be omitted or filled with `NaN`. |

### Optional Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `EstDates` | `list[float]` | `2002.0` | Decimal dates for the estimates. Crucial for DIC and pH accuracy. |
| `Equations` | `list[int]` | `1:16` | Specific equation permutations to run (1-16). |
| `MeasUncerts` | `dict` | WOCE Defaults | Keys: `"sal_u"`, `"temp_u"`, `"phosphate_u"`, `"nitrate_u"`, `"silicate_u"`, `"oxygen_u"`. |
| `pHCalcTF` | `bool` | `False` | Recalculates pH as if derived from TA/DIC rather than measured via dye. |
| `PerKgSwTF` | `bool` | `True` | Set to `False` if your inputs are volumetric (µmol/L) rather than molal (µmol/kg). Outputs are always molal. |
| `VerboseTF` | `bool` | `True` | Toggles terminal logging. |

---

## Outputs

PyESPER returns three dictionaries, shape `(n, e)` where `n` is the number of coordinates and `e` is the number of equations computed:

1.  **Estimates:** The computed variables (µmol/kg, except for unitless pH).
2.  **Coefficients:** The equation intercepts and weights used for the estimation (LIR algorithm only).
3.  **Uncertainties:** The propagated uncertainty bounds for the estimate.

*Missing Data Handling: Passing `NaN` as a coordinate or required predictor parameter will cascade and return `NaN` for all dependent equation estimates.*

---

## References

If you use this package, cite the relevant publications:

* **PyESPER Implementation:** Carter et al., 2021 (doi: 10.1002/lom3/10461)
* **LIRv3 / ESPER_NN (ESPERv1.1):** Carter, 2021 (doi: 10.5281/ZENODO.5512697)
* **LIARv1:** Carter et al., 2016 (doi: 10.1002/lom3.10087)
* **LIARv2, LIPHR, LINR:** Carter et al., 2018 (doi: 10.1002/lom3.10232)
* **LIPR, LISIR, LIOR:** Carter et al., 2021 (doi: 10.1002/lom3/10232)
* **Neural Network Inspiration (CANYON-B):** Bittig et al., 2018 (doi: 10.3389/fmars.2018.00328)