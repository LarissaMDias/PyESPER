# PyESPER
version 1.0.2

<ins>Note:</ins>
This is for use of the [PyESPER](https://github.com/LarissaMDias/PyESPER/blob/main) package. This package is being developed in parallel with [pyTRACE](https://github.com/d-sandborn/pyTRACE/tree/main).

## Quick Start
Please see the associated [Examples](https://github.com/LarissaMDias/PyESPER/blob/main/examples.py) for a quick example of use of the preliminary PyESPER. To run this code, you will need to first make sure that you have downloaded the required associated files from the GitHub page as follows. You will also need to ensure that the installed package [requirements](https://github.com/LarissaMDias/PyESPER/blob/main/requirements.txt) are met. The example uses the [GLODAPv2.2023](https://glodap.info) dataset and requires the [glodap](https://github.com/BjerknesClimateDataCentre/glodap/tree/master) package be installed.

To install PyESPER to your remote repository, clone this repository and then simply run the following in your terminal: 

pip install PyESPER

Mat_fullgrid folder: 
Folder of .mat files needed for each variable to be estimated, necessary for PyESPER_LIR or PyESPER_Mixed
    
NeuralNetworks folder:
Folder of .py files needed for each variable to be estimated, necessary for running PyESPER_NN or PyESPER_Mixed. These currently must be unpacked into the main directory. 
    
Uncertainty_Polys folder:
Folder of .mat files needed for ach variable to be estimated, necessary for running PyESPER_NN or PyESPER_Mixed
    
SimpleCantEstimateLR.csv:
File necessary for estimating anthropogenic carbon component for pH or DIC
    
## Introduction
PyESPER is a Python implementation of MATLAB Empirical Seawater Property Estimation Routines ([ESPERs](https://github.com/BRCScienceProducts/ESPER)), and the present version consists of a preliminary package which implements these routines. These routines provide estimates of seawater biogeochemical properties at user-provided sets of coordinates, depth, and available biogeochemical properties. Three algorithm options are available through these routines: 

1. Locally interpolated regressions (LIRs)
2. Neural networks (NNs)
3. Mixed

The routines predict coefficient and intercept values for a set of up to 16 equations, as follows:
(S=salinity, T=temperature, oxygen=dissolved oxygen molecule... see "PredictorMeasurements" for units). 
1.    S, T, A, B, C
2.    S, T, A, C
3.    S, T, B, C
4.    S, T, C
5.    S, T, A, B
6.    S, T, A
7.    S, T, B
8.    S, T
9.    S, A, B, C
10.   S, A, C
11.   S, B, C
12.   S, C
13.   S, A, B
14.   S, A
15.   S, B
16.   S

<ins>DesiredVariable: A, B, C</ins>

-TA: nitrate, oxygen, silicate

-DIC: nitrate, oxygen, silicate

-pH: nitrate, oxygen, silicate

-phosphate: nitrate, oxygen, silicate

-nitrate: phosphate, oxygen, silicate

-silicate: phosphate, oxygen, nitrate

-oxygen: phosphate, nitrate, silicate

### Documentation and citations:
LIARv1: Carter et al., 2016, doi: 10.1002/lom3.10087

LIARv2, LIPHR, LINR citation: Carter et al., 2018, doi: 10.1002/lom3.10232

LIPR, LISIR, LIOR, first described/used: Carter et al., 2021, doi: 10.1002/lom3/10232

LIRv3 and ESPER_NN (ESPERv1.1): Carter, 2021, doi: 10.5281/ZENODO.5512697

PyESPER is a Python implementation is ESPER:
Carter et al., 2021, doi: 10.1002/lom3/10461

ESPER_NN is inspired by CANYON-B, which also uses neural networks:
Bittig et al., 2018, doi: 10.3389/fmars.2018.00328

### PyESPER_LIR 
These are the first version of Python implementation of LIRv.3; ESPERv1.1, which use collections of interpolated linear networks. 

### PyESPER_NN
These are the first version of Python implementation of ESPERv1.1, which uses neural networks. 

### PyESPER_Mixed
These are the first version of Python implementation of ESPERv1.1, which is an average of the LIR and NN estimates. 

## Basic Use

### Requirements
For the present version, you will need to download the repository along with the affiliated neural network files within the [NeuralNetworks](https://github.com/LarissaMDias/PyESPER/tree/main/NeuralNetworks) folder, and the [SimpleCantEstimateLR_full.csv](https://github.com/LarissaMDias/PyESPER/blob/main/SimpleCantEstimateLR_full.csv) file for estimates involving anthropogenic carbon calculations (pH and dissolved inorganic carbon). 

To run the code, you will need numpy, pandas, seawater (now deprecated but necessary for consistency with ESPERv1), scipy, time, matplotlib, PyCO2SYS, importlib, statistics, and os packages.

Please refer to the examples .py file for proper use.

### Organization and Units
The measurements are provided in molar units or if potential temperature or AOU are needed but not provided by the user. Scale differences from TEOS-10 are a negligible component of alkalinity estimate error. PyCO2SYS is required if pH on the total scale is a desired output variable.

#### Input/Output dimensions:
p:    Integer number of desired property estimate types (e.g., TA, pH, NO3-)

n:    Integer number of desired estimate locations

e:    Integer number of equations used at each location

y:    Integer number of parameter measurement types provided by the user

n*e:  Total number of estimates returned as an n by e array

#### Required Inputs:

##### DesiredVariables (required 1 by p list, where p specifies the desired variable(x) in string format): 
List elements specify which variables will be returned. Excepting unitless pH, all outputs are in micromol per kg seawater. Naming of list elements must be exactly as demonstrated below (excamples ["TA"], ["DIC", "phosphate", "oxygen"]). 

<ins>Desired Variable: List Element Name (String Format):</ins>

Total Titration Seawater Alkalinity: TA

Total Dissolved Inorganic Carbon: DIC

in situ pH on the total scale: pH

Phosphate: phosphate

Nitrate: nitrate

Silicate: silicate

Dissolved Oxygen (O<sub>2</sub>):  oxygen

##### Path (required string):
Path directing Python to the location of saved/downloaded LIR files on the user's computer, if not in the current working directory (otherwise blank; e.g., '/Users/lara/Documents/Python' or ''). 

##### OutputCoordinates (required n by 3 dictionary, where n are the number of desired estimate locations and the three dicstionary keys are longitude, latitude, and depth):
Coordinates at which estimates are desired. The keys should be longitude (degrees E), latitude (degrees N), and positive integer depth (m), with dictionary keys named 'longitude', 'latitude', and 'depth' (ex: OutputCoordinates={"longitude": [0, 180, -50, 10], "latitude": [85, -20, 18, 0.5], "depth": [10, 1000, 0, 0]} or OutputCoordinates={"longitude": long, "latitude": lat, "depth": depth} when referring to a set of predefined lists or numpy arrays of latitude, longitude, and depth information. 

##### PredictorMeasurements (required n by y dictionary, where n are the number of desired estimate locations and y are the dictionary keys representing each possible input): 
Parameter measurements that will be used to estimate desired variables. Concentrations should be expressed as micromol per kg seawater unless PerKgSwTF is set to false in which case they should be expressed as micromol per L, temperature should be expressed as degrees C, and salinity should be specified with the unitless convention. NaN inputs are acceptable, but will lead to NaN estimates for any equations that depend on that parameter. The key order (y columns) is arbitrary, but naming of keys must adhere to the following convention (ex: PredictorMeasurements={"salinity": [35, 34.1, 32, 33], "temperature": [0.1, 10, 0.5, 2], "oxygen": [202.3, 214.7, 220.5, 224.2]} or PredictorMeasurements={'salinity': sal, 'temperature: temp, 'phosphate': phos, 'nitrate': nitrogen} when referring to predefined lists or numpy arrays of measurements:

<ins>Input Parameter: Dictionary Key Name</ins>

-Salinity: salinity

-Temperature: temperature

-Phosphate: phosphate

-Nitrate: nitrate

-Silicate: silicate

-O<sub>2</sub>: oxygen

#### Optional Inputs:
All remaining inputs must be specified as sequential input argument pairs (e.g., "EstDates"=EstDates when referring to a predefined list of dates, 'Equations'=[1:16], pHCalcTF=True, etc.)

##### EstDates (optional but recommended n by 1 list or 1 by 1 value, default 2002.0):
A list of decimal dates for the estimates (e.g., July 1 2020 would be 2020.5). If only a single date is supplied that value is used for all estimates. It is highly recommended that date(s) be provided for estimates of DIC and pH. This version of the code will accept 1 by n inputs as well. 

##### Equations (optional 1 by e list, default [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]):
List indicating which equations will be used to estimate desired variables. If [] is input or the input is not specified then all 16 equations will be used. 

#### Optional Inputs

##### MeasUncerts (Optional n by y dictionary or 1 by y dictionary, default: [0.003 S, 0.003 degrees C T or potential temperature, 2% phosphate, 2% nitrate, 2% silicate, 1% AOU or O<sub>2</sub>]): 
Dictionary of measurement uncertainties (see 'PredictorMeasurements' for units). Providing these estimates will improve PyESPER estimate uncertainties. Measurement uncertainties are a small p0art of PyESPER estimate uncertainties for WOCE-quality measurements. However, estimate uncertainty scales with measurement uncertainty, so it is recommended that measurement uncertainties be specified for sensor measurements. If this optional input argument is not provided, the default WOCE-quality uncertainty is assumed. If a 1 by y array is provided then the uncertainty estimates are assumed to apply uniformly to all input parameter measurements. Uncertainties should be presented with the following naming convention:

<ins>Input Uncertainties: Key Name</ins>

-Salinity: sal_u

-Temperature: temp_u

-Phosphate: phosphate_u

-Nitrate: nitrate_u

-Silicate: silicate_u

-Oxygen: oxygen_u

##### pHCalcTF (Optional boolean, default false):
If set to true, PyESPER will recalculate the pH to be a better estimate of what the seawater pH value would be if calculated from TA and DIC instead of measured with purified m-cresol dye. This is arguably also a better estimate of the pH than would be obtained from pre-2011 measurements with impure dyes. See LIPHR paper for details. 

##### PerKgSwTF (Optional boolean, default true):
Many sensors provide measurements in micromol per L (molarity) instead of micromol per kg seawater. Indicate false if provided measurements are expressed in molar units (concentrations must be micromol per L if so). Outputs will remain in molal units regardless. 

##### VerboseTF (Optional boolean, default true):
Setting this to false will reduce the number of updates, warnings, and errors printed by PyESPER. An additional step can be taken before executing the PyESPER function (see examples) that will further reduce updates, warnings, and errors, if desired.

#### Outputs:

##### Estimates: 
An n by e dictionary of estimates specific to the coordinates and parameter measurements provided as inputs. Units are micromoles per kg (equivalent to the deprecated microeq per kg seawater). Column names are the unique desired variable-equation combinations requested by the user. 

##### Coefficients (LIRs only):
An n by e dictionary of dictionaries of equation intercepts and coefficients specific to the coordinates and parameter measurements provided as inputs. Column names are the unique desired variable-equation combinations requested by the user. 

##### Uncertainties: 
An n by e dictionary of uncertainty estimates specific to the coordinates, parameter measurements, and parameter uncertaineis provided. Units are micromoles per kg (equivalent to the deprecated microeq per kg seawater). Column names are the unique desired variable-equation combinations requested by the user. 

#### Missing Data:
Should be indicated with a NaN. A NaN coordinate will yield NaN estimates for all equations at that coordinate. A NaN parameter value will yield NaN esitmates for all equations that require that parameter.  
