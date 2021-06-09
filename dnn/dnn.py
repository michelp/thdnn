from pygraphblas import FP32, binary_op
from . import timing

@timing
def mad(Y, w, b):
    Y.mxm(w, out=Y)
    with FP32.PLUS_PLUS:
        Y.mxm(b, out=Y)
    return Y

@timing
def relu(Y):
    Y.select(">0", out=Y)
    M = Y.select(">", 32)
    if len(M):
        Y[M] = 32
    return Y

def relu_prime(Y):
    return Y.select('>0').apply(FP32.ONE)

@timing
def dnn(W, B, Y):
    for w, b in zip(W, B):
        Y = relu(mad(Y, w, b))
    return Y

@timing
def hyperdnn(nlayers, W, B, Y):
    for i in range(nlayers):
        Y @= W
        with ReLU:
            Y @= B
    return Y
