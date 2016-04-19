###########################################################
# Governing equation is the convection equation.
#                       u_t = -f_x, f = f(u)
###########################################################
#==========================================================
def RungeKutta54_LS(U4D, dt, Eq, ConvA, ConvB, MeshX, PolyX, MeshY, PolyY):
    from numpy import zeros
    #######################################################
    # Reference:
    # 1994, ...
    #######################################################
    # Low storage Runge-Kutta coefficients
    CoefA_Vec = [0.0, \
        -567301805773.0/1357537059087.0, \
        -2404267990393.0/2016746695238.0, \
        -3550918686646.0/2091501179385.0, \
        -1275806237668.0/842570457699.0]
    CoefB_Vec = [1432997174477.0/9575080441755.0, \
        5161836677717.0/13612068292357.0, \
        1720146321549.0/2090206949498.0, \
        3134564353537.0/4481467310338.0, \
        2277821191437.0/14882151754819.0]
    CoefC_Vec = [0.0, \
        1432997174477.0/9575080441755.0, \
        2526269341429.0/6820363962896.0, \
        2006345519317.0/3224310063776.0, \
        2802321613138.0/2924317926251.0]
    ResU = zeros(U4D.shape)
    for Ind in range(5):
        dF4D = getdF4D(U4D, Eq, ConvA, ConvB, MeshX, PolyX, MeshY, PolyY)
        ResU = CoefA_Vec[Ind] * ResU - dt * (-dF4D)
        U4D = U4D - CoefB_Vec[Ind] * ResU
    return U4D

def getdF4D(U4D, Eq, ConvA, ConvB, MeshX, PolyX, MeshY, PolyY):
    SolDerivX4D = getdF4DX(U4D, Eq, ConvA, MeshX, PolyX)
    SolDerivY4D = getdF4DY(U4D, Eq, ConvB, MeshY, PolyY)
    dF4D = SolDerivX4D + SolDerivY4D
    return dF4D

def getdF4DX(U4D, Eq, ConvA, MeshX, PolyX):
    from numpy import zeros
    SolDerivX_4D_Mat = zeros(U4D.shape)
    Eq.A = ConvA
    for j in range(U4D.shape[1]):
        for l in range(U4D.shape[3]):
            Eq.Sol = U4D[:, j, :, l]
            Eq.Flux = Eq.A * Eq.Sol
            SolDerivX_4D_Mat[:, j, :, l] = Eq.getdF(MeshX, PolyX)
    return SolDerivX_4D_Mat

def getdF4DY(U4D, Eq, ConvB, MeshY, PolyY):
    from numpy import zeros
    SolDerivY_4D_Mat = zeros(U4D.shape)
    Eq.A = ConvB
    for i in range(U4D.shape[0]):
        for k in range(U4D.shape[2]):
            Eq.Sol = U4D[i, :, k, :]
            Eq.Flux = Eq.A * Eq.Sol
            SolDerivY_4D_Mat[i, :, k, :] = Eq.getdF(MeshY, PolyY)
    return SolDerivY_4D_Mat

