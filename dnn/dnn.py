from pygraphblas import FP32, binary_op
from . import timing

def relu(Y):
    Y.select(">0", out=Y)
    M = Y.select(">", 32)
    if len(M):
        Y[M] = 32
    return Y

@timing
def dnn(W, B, Y):
    for i, (w, b) in enumerate(zip(W, B)):
        Y = Y @ w
        with FP32.PLUS_PLUS:
            Y.mxm(b, out=Y)
        Y = relu(Y)
    return Y

@timing
def hyperdnn(nlayers, W, B, Y):
    for i in range(nlayers):
        Y @= W
        with ReLU:
            Y @= B
    return Y
