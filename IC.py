# Set IC
# Note: MATLAB behaves differently in this part.
def setIC(X_Mat, Y_Mat, IC_Flag):
    from numpy import sin, pi, exp
    if IC_Flag == 1:
        U_Mat = sin(pi*X_Mat) + sin(pi*Y_Mat)
    elif IC_Flag == 2:
        U_Mat = exp(-10*(X_Mat**2 + Y_Mat**2))
    else:
        exit('Initial Condition Error!')
    return U_Mat
