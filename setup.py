from setuptools import setup, find_packages

with open("README.md", "r") as f:
    description = f.read()

setup(
    name="PyESPER",
    version="1.0.1",
    description="Python version of ESPERv1",
    author="LMD",
    author_email="lmdias@uw.edu",
    packages=["PyESPER"],
    install_requires=[
        "numpy", 
        "seawater", 
        "scipy", 
        "matplotlib", 
        "PyCO2SYS",
        "pandas"
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
