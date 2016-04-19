###########################################################
class ConvectionLinearEq1D(object):
###########################################################
# Define Linear Convection equation
# u_t + a * u_x = 0
# a = const
# f(u) = a * u, f is flux
###########################################################
    def __init__(self, ArtDiffuFlag):
        self.Sol = None
        self.Flux = None
        self.A = None
        self.ArtDiffuFlag = ArtDiffuFlag

    def update(self, Sol_Mat):
        self.Sol = Sol_Mat
        self.Flux = self.A * Sol_Mat

###########################################################
    def getSolFace(self, XMesh, BCType=2):
        from numpy import ones
        SolFaceL_Vec = ones((XMesh.CellNMAX+1,))
        SolFaceL_Vec[0:-1] = self.Sol[0, :]
        SolFaceR_Vec = ones((XMesh.CellNMAX+1,))
        SolFaceR_Vec[1:] = self.Sol[-1, :]
        # Set Periodic BC
        if BCType == 1: # Periodic BC
            SolFaceL_Vec[-1] = SolFaceL_Vec[0]
            SolFaceR_Vec[0] = SolFaceR_Vec[-1]
        elif BCType == 2: # Dirichlet BC
            SolFaceL_Vec[-1] = 0.0
            SolFaceR_Vec[0] = 0.0
        else:
            exit('BC Error!')
        return SolFaceL_Vec, SolFaceR_Vec

    def getSolFluxFace(self, XMesh):
        SolFaceL_Vec, SolFaceR_Vec = self.getSolFace(XMesh)
        CoefK = 0.5
        SolFlux_Vec = CoefK * SolFaceR_Vec + (1.0 - CoefK) * SolFaceL_Vec
        SolFluxL_Vec = SolFlux_Vec[0:-1]
        SolFluxR_Vec = SolFlux_Vec[1:]
        return SolFluxL_Vec, SolFluxR_Vec

    def getSolDeriv(self, Poly):
        from numpy import dot
        # Without Correction Procedure, the Solution Derivative is just the following line.
        SolDeriv_Mat = dot(Poly.getLagrangePolyDeriv(), self.Sol)
        return SolDeriv_Mat

    def getSolDeriv_CPR(self, XMesh, Poly):
        from numpy import dot
        SolDeriv_Mat = self.getSolDeriv(Poly)
        # Construct Discontinuous Solution Polynomial
        SolDiscL_Vec = self.Sol[0, :]
        SolDiscR_Vec = self.Sol[-1, :]
        SolFluxL_Vec, SolFluxR_Vec = self.getSolFluxFace(XMesh)
        # Compute the derivative of continuous flux polynomial
        SolDeriv_CPR_Mat = 1.0 / XMesh.getJacobian() * ( SolDeriv_Mat \
            + dot(Poly.getRadauRightPolyDeriv().reshape((Poly.Order+1, 1)), \
                (SolFluxL_Vec - SolDiscL_Vec).reshape((1, XMesh.CellNMAX))) \
            + dot(Poly.getRadauLeftPolyDeriv().reshape((Poly.Order+1, 1)), \
                (SolFluxR_Vec - SolDiscR_Vec).reshape((1, XMesh.CellNMAX)) ) )
        return SolDeriv_CPR_Mat

    def getFluxFluxFace(self, XMesh):
        from numpy import ones, where, isclose
        SolFaceL_Vec, SolFaceR_Vec = self.getSolFace(XMesh)
        # Compute True Flux values at the interfaces
        # Flux Difference Splitting expression
        Flux_A = self.A * ones(SolFaceL_Vec.shape)
        IndexAll = isclose(SolFaceL_Vec, SolFaceR_Vec)
        IndexNotEqual = where(IndexAll == False)[0]
        Flux_A[IndexNotEqual] = \
            (self.A * SolFaceL_Vec[IndexNotEqual] - self.A * SolFaceR_Vec[IndexNotEqual]) \
            / (SolFaceL_Vec[IndexNotEqual] - SolFaceR_Vec[IndexNotEqual])
        Flux_Vec = 0.5 * ( \
            self.A * (SolFaceL_Vec + SolFaceR_Vec) \
            - abs(Flux_A) * (SolFaceL_Vec - SolFaceR_Vec) )
        FluxL_Vec = Flux_Vec[0:-1]
        FluxR_Vec = Flux_Vec[1:]
        return FluxL_Vec, FluxR_Vec

    def getFluxDeriv(self, Poly):
        from numpy import dot
        # Without Correction Procedure, Flux derivative is just the following line.
        FluxDeriv_Mat = dot(Poly.getLagrangePolyDeriv(), self.Flux)
        return FluxDeriv_Mat

    def getFluxDeriv_CPR(self, XMesh, Poly):
        from numpy import dot
        FluxDeriv_Mat = self.getFluxDeriv(Poly)
        # Construct Discontinuous Flux Polynomial
        FluxDiscL_Vec = self.Flux[0, :]
        FluxDiscR_Vec = self.Flux[-1, :]
        FluxL_Vec, FluxR_Vec = self.getFluxFluxFace(XMesh)
        # Compute the derivative of continuous flux polynomial
        FluxDeriv_CPR_Mat = 1.0 / XMesh.getJacobian() * ( FluxDeriv_Mat \
            + dot(Poly.getRadauRightPolyDeriv().reshape((Poly.Order+1, 1)), \
                (FluxL_Vec - FluxDiscL_Vec).reshape((1, XMesh.CellNMAX))) \
            + dot(Poly.getRadauLeftPolyDeriv().reshape((Poly.Order+1, 1)), \
                (FluxR_Vec - FluxDiscR_Vec).reshape((1, XMesh.CellNMAX)) ) )
        return FluxDeriv_CPR_Mat

    def getSolDeriv_CPR_ArtDiffu(self, XMesh, Poly):
        from numpy import ones, dot
        Vandermonde = Poly.getVandermondeLegendre()
        CellSize = XMesh.getCellSize()
        Epsilon_e = self.getSmoothIndicator(Vandermonde, CellSize)
        FluxDeriv_CPR_Mat = self.getSolDeriv_CPR(XMesh, Poly)
        FluxDeriv_CPR_ArtDiffu_Mat = \
            dot( ones((Poly.Order+1,1)), Epsilon_e.reshape((1, Epsilon_e.size)) ) \
            * FluxDeriv_CPR_Mat
        # Epsilon_e = 1E-5
        # FluxDeriv_CPR_ArtDiffu_Mat = Epsilon_e * FluxDeriv_CPR_Mat
        return FluxDeriv_CPR_ArtDiffu_Mat

    def getSolDeriv2_CPR_ArtDiffu(self, XMesh, Poly):
        Temp = self.Sol
        self.Sol = self.getSolDeriv_CPR_ArtDiffu(XMesh, Poly)
        SolDeriv2_CPR_ArtDiffu_Mat = self.getSolDeriv_CPR(XMesh, Poly)
        self.Sol = Temp # Ensure Solution in Eq is not changed.
        return SolDeriv2_CPR_ArtDiffu_Mat

    def getdF(self, XMesh, Poly):
        if self.ArtDiffuFlag == 0:
            f_x = self.getFluxDeriv_CPR(XMesh, Poly)
            # f_x = self.getFluxDeriv(Poly) * 1.0 / XMesh.getJacobian()
            dF = f_x
        elif self.ArtDiffuFlag == 1:
            f_x = self.getFluxDeriv_CPR(XMesh, Poly)
            u_xx = self.getSolDeriv2_CPR_ArtDiffu(XMesh, Poly)
            dF = f_x - u_xx
        else:
            exit('Flag Error!')
        return dF

###########################################################
    def getSmoothIndicator(self, Vandermonde, CellSize):
        from numpy.linalg import solve
        from numpy import dot, inner, log10, shape, where, isinf, nan_to_num, sin, pi, gradient
        SolBasis = solve(Vandermonde, self.Sol)
        # Truncated Solution Matrix
        SolBasis[-1, :] = 0
        SolTrunc = dot(Vandermonde, SolBasis)
        # Smooth/Resolution Indicator
        CellNMAX = shape(self.Sol)[1]
        S_e = [ inner(self.Sol[:, Ind] - SolTrunc[:, Ind], \
                      self.Sol[:, Ind] - SolTrunc[:, Ind]) \
              / inner(self.Sol[:, Ind], self.Sol[:, Ind]) \
            for Ind in range(CellNMAX) ]
        s_e = log10(S_e)
        # clean 's_e' variable: get rid of -Inf values
        # First set of (s_1, k, s_0) = (-6, 1.5, -4)
        # 2nd set: (-4, 1, -2.3)
        s_inf = -6
        s_e[where(isinf(s_e))] = s_inf # this value depends on k and s_0
        # or I can just convert nan to finite value
        # s_e = nan_to_num(s_e)
        # Parameters
        PolyOrder = shape(self.Sol)[0] - 1
        k = 1.5
        s_0 = -4    # s_0 ~ 1/(PolyOrder**4), not sure about this
        Epsilon_0 = CellSize / PolyOrder # * abs(self.A) / 10, not sure about this
        # Ranges
        Range1 = s_e < (s_0 - k)
        Range2 = [ (s_0 - k) < s_e[i] and s_e[i] < (s_0 + k) \
            for i in range(CellNMAX) ]
        Range3 = s_e > (s_0 + k)
        Epsilon_e = Epsilon_0 \
            * ( 0.0 * Range1 \
                + 0.5 * ( 1 + sin( pi*(s_e-s_0) / (2*k) ) ) * Range2 \
                + Range3 )
        return Epsilon_e

###########################################################
class ConvectionNonlinearEq1D(ConvectionLinearEq1D):
###########################################################
# Define Nonlinear Convection equation, currently it is Burgers Equation.
# u_t + u * u_x = 0
# f(u) = u**2 / 2, f is flux
###########################################################
    def __init__(self, X_Mat, Sol_Mat, ArtDiffuFlag):
        self.Sol = None
        self.Flux = None
        self.ArtDiffuFlag = ArtDiffuFlag

    def update(self, Sol_Mat):
        self.Sol = Sol_Mat
        self.Flux = 0.5 * Sol_Mat**2

    def getFluxFluxFace(self, XMesh):
        from numpy import ones, where, isclose
        SolFaceL_Vec, SolFaceR_Vec = self.getSolFace(XMesh)
        # Compute True Flux values at the interfaces
        # Flux Difference Splitting expression
        Flux_A = SolFaceL_Vec
        IndAll = isclose(SolFaceL_Vec, SolFaceR_Vec)
        IndNEq = where(IndAll == False)[0]
        Flux_A[IndNEq] = \
            (0.5*SolFaceL_Vec[IndNEq]**2 - 0.5*SolFaceR_Vec[IndNEq]**2) \
            / (SolFaceL_Vec[IndNEq] - SolFaceR_Vec[IndNEq])
        Flux_Vec = 0.5 * ( \
            0.5 * (SolFaceL_Vec**2 + SolFaceR_Vec**2) \
            - abs(Flux_A) * (SolFaceL_Vec - SolFaceR_Vec) )
        FluxL_Vec = Flux_Vec[0:-1]
        FluxR_Vec = Flux_Vec[1:]
        return FluxL_Vec, FluxR_Vec

    def getFluxDeriv(self, Poly):
        from numpy import dot
        # Without Correction Procedure, Flux derivative is just the following line.
        # f = u**2 / 2, f_x = u * u_x
        FluxDeriv_Mat = dot(Poly.getLagrangePolyDeriv(), self.Sol) * self.Sol
        return FluxDeriv_Mat
###########################################################
class ConvectionLinearEqPseudo2D(ConvectionLinearEq1D):

    def __init__(self, ConvA, ConvB, MeshX, MeshY, PolyX, PolyY, ArtDiffuFlag):
        self.Sol = None
        self.Flux = None
        self.A = ConvA
        self.B = ConvB
        self.MeshX = MeshX
        self.MeshY = MeshY
        self.PolyX = PolyX
        self.PolyY = PolyY
        self.ArtDiffuFlag = ArtDiffuFlag

    def getdF4D(self, U4D):
        SolDerivX4D = self.getdF4DX(U4D)
        SolDerivY4D = self.getdF4DY(U4D)
        dF4D = SolDerivX4D + SolDerivY4D
        return dF4D

    def getdF4DX(self, U4D):
        from numpy import zeros
        SolDerivX_4D_Mat = zeros(U4D.shape)
        for j in range(U4D.shape[1]):
            for l in range(U4D.shape[3]):
                self.update(U4D[:, j, :, l])
                SolDerivX_4D_Mat[:, j, :, l] = self.getdF(self.MeshX, self.PolyX)
        return SolDerivX_4D_Mat

    def getdF4DY(self, U4D):
        from numpy import zeros
        SolDerivY_4D_Mat = zeros(U4D.shape)
        for i in range(U4D.shape[0]):
            for k in range(U4D.shape[2]):
                self.update(U4D[i, :, k, :])
                SolDerivY_4D_Mat[i, :, k, :] = self.getdF(self.MeshY, self.PolyY)
        return SolDerivY_4D_Mat

    def RungeKutta54_LS(self, dt, U4D):
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
            dF4D = self.getdF4D(U4D)
            ResU = CoefA_Vec[Ind] * ResU - dt * (-dF4D)
            U4D = U4D - CoefB_Vec[Ind] * ResU
        return U4D

###########################################################
class ConvectionNonlinearEqPseudo2D(ConvectionNonlinearEq1D):

    def __init__(self, MeshX, MeshY, PolyX, PolyY, ArtDiffuFlag):
        self.Sol = None
        self.Flux = None
        self.MeshX = MeshX
        self.MeshY = MeshY
        self.PolyX = PolyX
        self.PolyY = PolyY
        self.ArtDiffuFlag = ArtDiffuFlag

    def getdF4D(self, U4D):
        SolDerivX4D = self.getdF4DX(U4D)
        SolDerivY4D = self.getdF4DY(U4D)
        dF4D = SolDerivX4D + SolDerivY4D
        return dF4D

    def getdF4DX(self, U4D):
        from numpy import zeros
        SolDerivX_4D_Mat = zeros(U4D.shape)
        for j in range(U4D.shape[1]):
            for l in range(U4D.shape[3]):
                self.update(U4D[:, j, :, l])
                SolDerivX_4D_Mat[:, j, :, l] = self.getdF(self.MeshX, self.PolyX)
        return SolDerivX_4D_Mat

    def getdF4DY(self, U4D):
        from numpy import zeros
        SolDerivY_4D_Mat = zeros(U4D.shape)
        for i in range(U4D.shape[0]):
            for k in range(U4D.shape[2]):
                self.update(U4D[i, :, k, :])
                SolDerivY_4D_Mat[i, :, k, :] = self.getdF(self.MeshY, self.PolyY)
        return SolDerivY_4D_Mat

    def RungeKutta54_LS(self, dt, U4D):
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
            dF4D = self.getdF4D(U4D)
            ResU = CoefA_Vec[Ind] * ResU - dt * (-dF4D)
            U4D = U4D - CoefB_Vec[Ind] * ResU
        return U4D


