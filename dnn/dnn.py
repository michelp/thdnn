from pygraphblas import FP32, binary_op
from . import timing

@timing
def dnn(W, B, Y):
    for i, (w, b) in enumerate(zip(W, B)):  # for every weight, bias matrix
        Y = Y @ w
        with FP32.PLUS_PLUS:  # with PLUS_PLUS semiring:
            Y.mxm(b, out=Y)  # Y = Y @ B
        Y.select(">0", out=Y)  # select all >0 from Y
        M = Y.select(">", 32)  # select all > 32
        if len(M):  # if any > 32
            Y[M] = 32  # truncate to 32
    return Y


@timing
def hyperdnn(nlayers, W, B, Y):
    frames = []
    for i in range(nlayers):
        Y @= W
        with ReLUNeuron_semiring:
            Y @= B
        Y.select(">0", out=Y)
        print(Y.nvals)
    return Y
