from pygraphblas import Matrix, binary_op, unary_op, FP64, FP32
from operator import mul, eq, attrgetter
from functools import reduce
from . import timing
from random import random, uniform


@unary_op(FP32)  
def random_op(x): 
    return random()


@binary_op(FP32)  
def uniform_op(x, y):
    return uniform(x, y)


def permutation_matrix(size, default=1.0):
    P = Matrix.sparse(FP32, size, size)
    P[size - 1, 0] = default
    for i in range(size - 1):
        P[i, i + 1] = default
    return P


def mixed_topo_radix(topos, default=1.0):
    sizes = [reduce(mul, x) for x in topos]
    assert reduce(eq, sizes)
    size = sizes[0]
    layers = []
    P = permutation_matrix(size, default)

    for t in topos:
        place_value = 1
        for n in t:
            layer = Matrix.sparse(FP32, size, size)
            for j in range(n):
                layer += P ** (j * place_value)
            place_value *= n
            layers.append(layer)
    return layers


def ddnn(spec, fill=1.0):
    return [Matrix.dense(FP32, spec[i], spec[i + 1], fill=fill)
            for i in range(len(spec) - 1)]


def radixnet(topos, spec, default=1.0, kron_op=None):
    return [d.kronecker(w, kron_op)
            for d, w in
            zip(mixed_topo_radix(topos, default), ddnn(spec))]


_nrowsgetter = attrgetter("nrows")


@timing
def hypergraph(mt, size=None):
    if size is None:
        size = sum(map(_nrowsgetter, mt)) + mt[-1].nrows
    r = Matrix.sparse(FP32, size, size)
    ioffset = 0
    joffset = 0
    for m in mt:
        joffset += m.nrows
        for c, (i, j, k) in enumerate(m):
            r[i + ioffset, j + joffset] = k
        ioffset += m.nrows
    return r
