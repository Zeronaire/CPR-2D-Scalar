from SJC_Mesh import Mesh1D
from SJC_SpectralToolbox import Poly1D
from SJC_Equation import ConvectionLinearEq
from IC import setIC
from SJC_TimeDiscretization import RungeKutta54_LS
from SJC_Utilities import assemble4DTo2D
from numpy import mod, zeros
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
###########################################################

RangeX = [0.0, 2.0]
RangeY = [0.0, 2.0]
OrderXNMAX = 3
OrderYNMAX = 3
CellXNMAX = 8
CellYNMAX = 8
ConvA = 1
ConvB = 1
QuadType = 'LGL'

MeshX = Mesh1D(RangeX, OrderXNMAX, CellXNMAX, QuadType)
MeshY = Mesh1D(RangeY, OrderYNMAX, CellYNMAX, QuadType)
X1D = MeshX.getGlobalCoordinates()
Y1D = MeshY.getGlobalCoordinates()
X4D = zeros((X1D.shape[0], Y1D.shape[0], X1D.shape[1], Y1D.shape[1])) # k, l, i, j
Y4D = zeros((X1D.shape[0], Y1D.shape[0], X1D.shape[1], Y1D.shape[1]))
for j in range(MeshY.CellNMAX):
    for l in range(MeshY.NodeInCellNMAX):
        X4D[:, l, :, j] = X1D
for i in range(MeshX.CellNMAX):
    for k in range(MeshX.NodeInCellNMAX):
        Y4D[k, :, i, :] = Y1D
X2D = assemble4DTo2D(X4D)
Y2D = assemble4DTo2D(Y4D)

PolyX = Poly1D(MeshX.getSolutionPoints())
PolyY = Poly1D(MeshY.getSolutionPoints())

ArtDiffuFlag = 0
Eq = ConvectionLinearEq(ArtDiffuFlag)

ICFlag = 1
U4D = setIC(X4D, Y4D, ICFlag)

CFL = 5E-2
TimeStep = CFL * MeshX.getCellSize() / abs(ConvA)
TimeEnd = (RangeX[1] - RangeX[0]) / abs(ConvA)

fig = plt.figure()
Time = 0.0
TimeInd = 0
while (Time < TimeEnd):
    Time = TimeInd * TimeStep
    if Time > TimeEnd:
        Time = TimeEnd
    if mod(TimeInd, 1) == 0:
        print(('%.4d' % TimeInd) + ': ' + ('%.4f' % Time) + ' / ' + ('%.4f' % TimeEnd))
        ax = fig.gca(projection='3d')
        U2D = assemble4DTo2D(U4D)
        ax.set_zlim(-2.1, 2.1)
        ax.plot_surface(X2D, Y2D, U2D, rstride=1, cstride=1, cmap=cm.coolwarm, \
                       linewidth=0, antialiased=False)
        FigName_Str = ('%03d' % TimeInd) + '.jpg'
        plt.savefig(FigName_Str)
        plt.clf()
    U4D = RungeKutta54_LS(U4D, TimeStep, Eq, ConvA, ConvB, MeshX, PolyX, MeshY, PolyY)
    if U4D.max() > 1E3:
        exit('Divergence!')
    TimeInd = TimeInd + 1

