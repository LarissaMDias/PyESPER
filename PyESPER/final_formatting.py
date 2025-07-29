def final_formatting(DesiredVariables, Cant_adjusted={}, Est_pre={}):

    """
    Formatting the final data output for estimates

    Inputs:
        DesiredVariables: List of desired variables to estimate
        Cant_adjusted: Dictionary of estimates adjusted for anthropogenic
            carbon for each combination
        Est_pre: Dictionary of estimates for each combination

    Output:
        Estimates: Dictionary of estimates for each combination
    """

    # Conditional to whether anthropogenic carbon was needed or not
    if ("pH" or "DIC") in DesiredVariables:
        Estimates=Cant_adjusted
        print("anthropogenic carbon has been incorporated into some estimates")
    else:
        Estimates=Est_pre
        print("anthropogenic carbon is not considered for these estimates")

    return Estimates

