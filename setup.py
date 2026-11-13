from setuptools import setup, find_packages

with open("README.md", "r") as f:
    description = f.read()

setup(
    name="PyESPER",
    version="1.1.1",
    description="Python version of ESPERv1",
    author="LMD",
    author_email="lmdias@uw.edu",
    packages=["PyESPER"],
    package_data={
        "PyESPER": ["*.csv"],
    },
    include_package_data=True,  # make sure the data files are installed
    install_requires=[
        "numpy",
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
