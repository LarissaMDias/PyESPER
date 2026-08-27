"""Frozen copy of ``PyESPER.run_nets.run_nets`` exactly as it was before the
batched-forward-pass rewrite (see ``PyESPER/run_nets.py``'s module docstring for
why/how it changed).

This exists ONLY so ``test_run_nets_batched.py`` has a known-good, independently
implemented reference to numerically validate the new implementation against --
it is not part of the public API and should not be imported by anything else.
Do not "fix" or optimize this file; if it ever needs to change, the whole point
of the regression test it backs is gone.
"""

import functools


@functools.lru_cache(maxsize=None)
def _load_net(module_name):
    import importlib

    return importlib.import_module(module_name).PyESPER_NN


def run_nets_legacy(DesiredVariables, Equations, code={}):
    import numpy as np

    # Predefining dictionaries to populate
    EstAtl, EstOther = {}, {}
    P, Sd, Td, Ad, Bd, Cd = {}, {}, {}, {}, {}, {}

    cosd = sind = lat = depth = None

    for name, value in code.items():
        if cosd is None:
            cosd = np.cos(np.deg2rad(value["Longitude"] - 20)).tolist()
            sind = np.sin(np.deg2rad(value["Longitude"] - 20)).tolist()
            lat, depth = value["Latitude"].tolist(), value["Depth"].tolist()
        Sd[name] = value["S"].astype(float).tolist()
        Td[name] = value["T"].astype(float).tolist()
        Ad[name] = value["A"].astype(float).tolist()
        Bd[name] = value["B"].astype(float).tolist()
        Cd[name] = value["C"].astype(float).tolist()

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
        16: [Sd],
    }

    for e in Equations:
        for v in DesiredVariables:
            name = v + str(e)
            variables = [var[name] for var in equation_map[e]]
            P[name] = [[[cosd, sind, lat, depth] + variables]]
            netstimateAtl, netstimateOther = [], []
            for n in range(1, 5):
                fOName = f"NeuralNetworks.ESPER_{v}_{e}_Other_{n}"
                fAName = f"NeuralNetworks.ESPER_{v}_{e}_Atl_{n}"
                net_atl = _load_net(fAName)
                net_other = _load_net(fOName)
                netstimateAtl.append(net_atl(P[name]))
                netstimateOther.append(net_other(P[name]))

            EstAtlL = np.stack([netstimateAtl[na][0] for na in range(4)], axis=1)
            EstOtherL = np.stack([netstimateOther[no][0] for no in range(4)], axis=1)

            EstAtl[name] = EstAtlL
            EstOther[name] = EstOtherL

    return EstAtl, EstOther
