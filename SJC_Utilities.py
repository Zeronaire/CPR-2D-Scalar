from numpy import zeros

def heaviside(x):
    from numpy import sign
    y = 0.5 * ( sign(x) + 1.0 )
    return y

def assemble3DTo2D(In3D):
    N = (In3D.shape[1]-1)*In3D.shape[2]
    Out2D = zeros((In3D.shape[0], N+1))
    for i in range(Out2D.shape[0]):
        # This line can reduce the error. However, it only relieves the pain.
        # In3D[i, 0, 1:] = (In3D[i, 0, 1:] + In3D[i, -1, 0:-1] ) / 2.0 # Try to minimize the error
        Out2D[i, 0:-1] = In3D[i, 0:-1, :].T.reshape((1, N))
        Out2D[i, -1] = In3D[i, -1, -1]
    return Out2D

def assemble2DTo1D(In2DIn1D, Out1DRowN, Out1DColN):
    from numpy import insert
    Out1D = zeros((Out1DRowN, Out1DColN))
    for i in range(Out1DColN-1):
        Out1D[0:Out1DRowN-1, i] = In2DIn1D[(Out1DRowN-1)*i:(Out1DRowN-1)*(i+1)]
        Out1D[Out1DRowN-1, i] = In2DIn1D[(Out1DRowN-1)*(i+1)]
    i = Out1DColN - 1
    Out1D[:, i] = In2DIn1D[(Out1DRowN-1)*i:]
    return Out1D

def assemble4DTo2D(In4D):
    from numpy import zeros
    RowN = (In4D.shape[1]-1)*In4D.shape[3] + 1 # Y
    ColN = (In4D.shape[0]-1)*In4D.shape[2] + 1 # X
    Out2D = zeros((RowN, ColN))
    for j in range(In4D.shape[3]):
        for l in range(In4D.shape[1]-1):
            for i in range(In4D.shape[2]):
                for k in range(In4D.shape[0]-1):
                    Out2D[j*(In4D.shape[1]-1)+l, i*(In4D.shape[0]-1)+k] = In4D[k, l, i, j]
            Out2D[j*(In4D.shape[1]-1)+l, -1] = In4D[-1, l, -1, j]
    j = In4D.shape[3]-1
    l = In4D.shape[1]-1
    for i in range(In4D.shape[2]):
        for k in range(In4D.shape[0]-1):
            Out2D[j*(In4D.shape[1]-1)+l, i*(In4D.shape[0]-1)+k] = In4D[k, l, i, j]
    Out2D[j*(In4D.shape[1]-1)+l, -1] = In4D[-1, l, -1, j]
    return Out2D
