from contextlib import contextmanager
import sys, os

@contextmanager
def suppress_stdout():
    """
    Function to suppress command-line updates.
    """
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

            
            
def verbose_tf(DesiredVariables, OutputCoordinates, PredictorMeasurements, 
**kwargs):
    """
    Reads in the VerboseTF argument and interprets, using the above 
function.
    
    Inputs: 
        VerboseTF: Boolean input whereby True allows default command-line 
updates and False stops printing updates to the command line. 
    """
    with suppress_stdout():
        PyESPER_NN(DesiredVariables, OutputCoordinates, 
PredictorMeasurements, **kwargs)
