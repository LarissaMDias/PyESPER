#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 12:42:48 2026

Fixing an issue in NN files whereby a RuntimeWarning appears after each
tansig_apply function. 

@author: lara
"""

from pathlib import Path

# This script can live in the repository root or in tests/.
HERE = Path(__file__).resolve().parent
ROOT = HERE if (HERE / "NeuralNetworks").is_dir() else HERE.parent
NETWORK_DIR = ROOT / "NeuralNetworks"

# Start with True to preview changes. Set to False after checking the output.
DRY_RUN = False

OLD_LINE = "return 2 / (1 + np.exp(-2 * n)) - 1"
NEW_LINE = "return np.tanh(n)"


def replace_tansig():
    changed = []
    already_fixed = []

    for path in sorted(NETWORK_DIR.glob("ESPER_*.py")):
        text = path.read_text(encoding="utf-8")
        occurrences = text.count(OLD_LINE)

        if occurrences == 0:
            if NEW_LINE in text:
                already_fixed.append(path.name)
            continue

        if occurrences > 1:
            raise RuntimeError(
                f"Found {occurrences} matching lines in {path}; refusing to edit."
            )

        changed.append(path.name)
        if not DRY_RUN:
            path.write_text(
                text.replace(OLD_LINE, NEW_LINE),
                encoding="utf-8",
            )

    action = "Would change" if DRY_RUN else "Changed"
    print(f"{action}: {len(changed)}")
    print(f"Already fixed: {len(already_fixed)}")

    for filename in changed:
        print(f"  {filename}")


if __name__ == "__main__":
    replace_tansig()

    