class Poly1D(object):

    def __init__(self, Node_Vec):
        self.Order = len(Node_Vec) - 1
        self.Nodes = Node_Vec

    def getLagrangeBasis(self):
        from numpy.polynomial.polynomial import Polynomial
        #########################
        # Construct Lagrange Basis Functions in the form of the Polynomial class
        # imported from numpy.
        # Output:
        #           List of Polynomial object. Each Polynomial object has 3 default
        #           parameters. The first is the coefficients, second is the domain,
        #           third is the window size. Details about the latter 2 parameters
        #           are in the definition of Polynomial class in numpy.
        x = self.Nodes
        PolyList_List = []
        for j in range(self.Order+1):
            Poly_Poly = 1.0
            for k in range(self.Order+1):
                if k == j:
                    continue
                Poly_Poly *= Polynomial([-x[k], 1.0]) / (x[j] - x[k])
            PolyList_List.append(Poly_Poly)
        return PolyList_List

    def getLagrangePolyDeriv(self):
        from numpy import zeros
        from numpy.polynomial.polynomial import polyval
        ###########################################################
        # Construct Lagrange Polynomial of order OrderNMAX
        NodesNMAX = self.Order + 1
        Basis_List = self.getLagrangeBasis()
        BasisDeriv_List = \
            [Basis_List[i].deriv(1) for i in range(NodesNMAX)]
        BasisDerivCoef_List = \
            [BasisDeriv_List[i].coef for i in range(NodesNMAX)]
        BasisDeriv_Mat = zeros((NodesNMAX, NodesNMAX))
        # Here raises a problem for the arrangement
        for Ind in range(NodesNMAX):
            BasisDeriv_Mat[:, Ind] = \
                polyval(self.Nodes, BasisDerivCoef_List[Ind])
        return BasisDeriv_Mat
        # Each row share the same Nodes Coordinate.

    def getRadauRightPoly(self):
        from scipy.special import legendre
        from numpy.polynomial.polynomial import polyval
        from numpy import insert
        ###########################################################
        # Construct Right Radau Polynomial
        Temp1 = legendre(self.Order).coeffs
        Temp2 = legendre(self.Order+1).coeffs
        RadauRPoly_Vec = (-1)**self.Order / 2.0 * (insert(Temp1, 0, 0) - Temp2)
        RadauRPolyValue_Vec = polyval(self.Nodes, RadauR_Vec[::-1])
        return RadauRPolyValue_Vec

    def getRadauRightPolyDeriv(self):
        from scipy.special import legendre
        from numpy.polynomial.polynomial import polyval
        from numpy import insert
        ###########################################################
        # Construct Right Radau Polynomial
        Temp1 = legendre(self.Order).deriv(1).coeffs
        Temp2 = legendre(self.Order+1).deriv(1).coeffs
        RadauRPolyDeriv_Vec = (-1)**self.Order / 2.0 * (insert(Temp1, 0, 0) - Temp2)
        RadauRPolyDerivValue_Vec = polyval(self.Nodes, RadauRPolyDeriv_Vec[::-1])
        return RadauRPolyDerivValue_Vec

    def getRadauLeftPolyDeriv(self):
        RadauRPolyDerivValue_Vec = self.getRadauRightPolyDeriv()
        return -RadauRPolyDerivValue_Vec[::-1]

    def getVandermondeLegendre(self):
        from numpy.polynomial.legendre import legvander
        return legvander(self.Nodes, self.Order)

class Poly2D(object):

    def __init__(self, NodeX_Mat, NodeY_Mat):
        self.OrderX = NodeX_Mat.shape[0] - 1
        self.OrderY = NodeY_Mat.shape[1] - 1
        self.NodesX = NodeX_Mat
        self.NodesY = NodeY_Mat

    def getLagrangeBasis(self):
        from numpy.polynomial.polynomial import Polynomial
        #########################
        # Construct Lagrange Basis Functions in the form of the Polynomial class
        # imported from numpy.
        # Output:
        #           List of Polynomial object. Each Polynomial object has 3 default
        #           parameters. The first is the coefficients, second is the domain,
        #           third is the window size. Details about the latter 2 parameters
        #           are in the definition of Polynomial class in numpy.
        PolyListX_List = []
        for IndX in range(self.OrderX+1):
            x = self.NodesX[IndX, :]
            for j in range(self.OrderX+1):
                PolyX_Poly = 1.0
                for k in range(self.OrderX+1):
                    if k == j:
                        continue
                    PolyX_Poly *= Polynomial([-x[k], 1.0]) / (x[j] - x[k])
                PolyListX_List.append(PolyX_Poly)
        PolyListY_List = []
        for IndY in range(self.OrderY+1):
            y = self.NodesY[:, IndY]
            for j in range(self.OrderY+1):
                PolyY_Poly = 1.0
                for k in range(self.OrderY+1):
                    if k == j:
                        continue
                    PolyY_Poly *= Polynomial([-y[k], 1.0]) / (y[j] - y[k])
                PolyListY_List.append(PolyY_Poly)
        return PolyListX_List, PolyListY_List

    def getLagrangePolyDeriv(self):
        from numpy import zeros
        from numpy.polynomial.polynomial import polyval
        ###########################################################
        # Construct Lagrange Polynomial of order OrderNMAX
        XN = self.OrderX + 1
        YN = self.OrderY + 1
        BasisX_List, BasisY_List = self.getLagrangeBasis()
        # X
        BasisXDeriv_List = \
            [BasisX_List[i].deriv(1) for i in range(XN*XN)]
        BasisXDerivCoef_List = \
            [BasisXDeriv_List[i].coef for i in range(XN*XN)]
        BasisXDeriv_Mat = zeros((XN, XN, XN))
        # Here raises a problem for the arrangement
        for i in range(XN):
            for j in range(XN):
                BasisXDeriv_Mat[i, :, j] = \
                    polyval(self.NodesX[i, :], BasisXDerivCoef_List[(i+1)*j])
        # Y
        BasisYDeriv_List = \
            [BasisY_List[i].deriv(1) for i in range(YN*YN)]
        BasisYDerivCoef_List = \
            [BasisYDeriv_List[i].coef for i in range(YN*YN)]
        BasisYDeriv_Mat = zeros((YN, YN, YN))
        # Here raises a problem for the arrangement
        for i in range(YN):
            for j in range(YN):
                BasisYDeriv_Mat[i, :, j] = \
                    polyval(self.NodesY[:, i], BasisYDerivCoef_List[(i+1)*j])
        return BasisXDeriv_Mat, BasisYDeriv_Mat
