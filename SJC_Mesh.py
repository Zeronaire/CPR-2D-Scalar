class Mesh1D(object):

    def __init__(self, Range, OrderNMAX, CellNMAX, QuadType):
        self.Range = Range
        self.PolyOrder = OrderNMAX
        self.NodeInCellNMAX = OrderNMAX + 1
        self.CellNMAX = CellNMAX
        self.QuadType = QuadType

    def getCellSize(self):
        CellSize = (self.Range[1] - self.Range[0]) / self.CellNMAX
        return CellSize

    def getCellCenter(self):
        from numpy import linspace
        Face_Vec = linspace(self.Range[0], self.Range[1], self.CellNMAX+1)
        CellCenter_Vec = ( Face_Vec[0:-1] + Face_Vec[1:] ) / 2.0
        return CellCenter_Vec

    def getJacobian(self):
        Jacobian = self.getCellSize() / 2.0
        return Jacobian

    def getSolutionPoints(self):
        from scipy.special import legendre
        from numpy.polynomial.polynomial import polyroots
        from numpy import zeros
        # Currently, there is only 1 type available.
        # LGL
        ###########################################################
        # Construct legendre polynomials of order OrderNMAX
        Poly = legendre(self.PolyOrder)
        Poly_Vec = Poly.coeffs
        PolyDeriv = Poly.deriv(1)
        # Note: In numpy.polynomial.polynomial module, the first element in the
        #       coefficients array is the constant value. While in scipy.special
        #       module, the coefficient array is arranged in descending order.
        PolyDeriv_Vec = PolyDeriv.coeffs
        # Compute Legendre-Gauss-Lobatto integration points
        # LGL points are local coordinates
        SolPts_Vec = zeros((self.NodeInCellNMAX,))
        SolPts_Vec[0] = -1.0
        SolPts_Vec[-1] = 1.0
        SolPts_Vec[1:-1] = polyroots(PolyDeriv_Vec[::-1])
        return SolPts_Vec

    def getGlobalCoordinates(self):
        from numpy import tile
        Global_Mat = \
            tile(self.getCellCenter(), (self.NodeInCellNMAX, 1)) \
            + tile( self.getJacobian() \
                * self.getSolutionPoints().reshape(self.NodeInCellNMAX, 1), \
                (1, self.CellNMAX) )
        return Global_Mat

