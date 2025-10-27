def run_nets(DesiredVariables, Equations, code={}):

    """
    Running neural nets

    Inputs:
        DesiredVariables: List of variables for estimates
        Equations: List of desired equations
        code: Dictionary of preprocessed measurements

    Outputs:
        EstAtl: Dictionary of estimates for the Atlantic and Arctic
            Oceans
        EstOther: Dictionary of estimates for not Altnatic/Arctic
    """

    import importlib
    import numpy as np

    # Predefining dictionaries to populate
    EstAtl, EstOther = {}, {}
    P, Sd, Td, Ad, Bd, Cd = {}, {}, {}, {}, {}, {}

    # Calculating inputs for nets and formatting them
    for name, value in code.items():
        cosd = np.cos(np.deg2rad(value["Longitude"] - 20)).tolist()
        sind = np.sin(np.deg2rad(value["Longitude"] - 20)).tolist()
        lat, depth = value["Latitude"].tolist(), value["Depth"].tolist()
        # Convert columns to lists of floats
        Sd[name] = value["S"].astype(float).tolist()
        Td[name] = value["T"].astype(float).tolist()
        Ad[name] = value["A"].astype(float).tolist()
        Bd[name] = value["B"].astype(float).tolist()
        Cd[name] = value["C"].astype(float).tolist()

   # Define a mapping from equations to the list of variable dictionaries
    equation_map = {
        1: [Sd, Td, Ad, Bd, Cd],
        2: [Sd, Td, Ad, Cd],
        3: [Sd, Td, Bd, Cd],
        4: [Sd, Td, Cd],
        5: [Sd, Td, Ad, Bd],
        6: [Sd, Td, Ad],
        7: [Sd, Td, Bd],
        8: [Sd, Td],
        9: [Sd, Ad, Bd, Cd],
        10: [Sd, Ad, Cd],
        11: [Sd, Bd, Cd],
        12: [Sd, Cd],
        13: [Sd, Ad, Bd],
        14: [Sd, Ad],
        15: [Sd, Bd],
        16: [Sd]
    }

    # Create the correct vector for each equation case
    for e in Equations:
        for v in DesiredVariables:
            name = v + str(e)
            # Get the corresponding variables for the equation
            variables = [var[name] for var in equation_map[e]]
            P[name] = [[[cosd, sind, lat, depth] + variables]]
            netstimateAtl, netstimateOther = [], []
            for n in range(1, 5):   
                fOName = f"NeuralNetworks.ESPER_{v}_{e}_Other_{n}"
                fAName = f"NeuralNetworks.ESPER_{v}_{e}_Atl_{n}"
                moda = importlib.import_module(fAName)
                modo = importlib.import_module(fOName)
                from importlib import reload
                reload(moda)
                reload(modo)   
                # Running the nets
                netstimateAtl.append(moda.PyESPER_NN(P[name]))
                netstimateOther.append(modo.PyESPER_NN(P[name]))
        
            # Process estimates for Atlantic and Other regions
            EstAtlL = [[netstimateAtl[na][0][eatl] for na in range(4)] for eatl in range(len(netstimateAtl[0][0]))]
            EstOtherL = [[netstimateOther[no][0][eoth] for no in range(4)] for eoth in range(len(netstimateOther[0][0]))]
        
            # Store the result
            EstAtl[name] = EstAtlL
            EstOther[name] = EstOtherL
            
    return EstAtl, EstOther

