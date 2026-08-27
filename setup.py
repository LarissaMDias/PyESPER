from setuptools import setup, find_packages

with open("README.md", "r") as f:
    description = f.read()

setup(
    name="PyESPER",
    version="1.1.1",
    description="Python version of ESPERv1",
    author="LMD",
    author_email="lmdias@uw.edu",
    # find_packages() picks up PyESPER, PyESPER.kernels, PyESPER.tests and the
    # NeuralNetworks weight package (all have __init__.py). Note a plain (wheel)
    # install still does NOT carry the Mat_fullgrid/ .mat grids -- they are 288 MB
    # and live outside any package. Use an editable install (pip install -e .),
    # which maps back to this source tree so PyESPER.paths.data_root() finds them
    # automatically, or set PYESPER_DATA_DIR. See PyESPER/paths.py.
    packages=find_packages(),
    package_data={
        "PyESPER": ["*.csv"],
    },
    install_requires=[
        "numpy",
        # "seawater",
        "scipy",
        "matplotlib",
        "PyCO2SYS",
        "pandas",
        "numba",
    ],
    entry_points={
        "console_scripts": [
            "lir = PyESPER:lir",
            "nn = PyESPER:nn",
            "mixed = PyESPER:mixed",
        ],
    },
    long_description=description,
    long_description_content_type="text/markdown",
)
