# Set IC
# Note: MATLAB behaves differently in this part.
def setIC(X_Mat, Y_Mat, IC_Flag):
    from numpy import sin, pi
    if IC_Flag == 1:
        U_Mat = sin(pi*X_Mat) + sin(pi*Y_Mat)
    else:
        exit('Initial Condition Error!')
    return U_Mat
