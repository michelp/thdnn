from functools import partial
from pathlib import Path
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from pygraphblas import Matrix, Vector, FP32, BOOL

from . import timing

NFEATURES = 60000
BIAS = {1024: -0.3, 4096: -0.35, 16384: -0.4, 65536: -0.45}

@timing
def load_images(neurons, dest):
    fname = "{}/sparse-images-{}.{}"
    binfile = fname.format(dest, neurons, "ssb")
    if Path(binfile).exists():
        return Matrix.from_binfile(binfile.encode("ascii"))
    images = Path(fname.format(dest, neurons, "tsv"))
    with images.open() as i:
        m = Matrix.from_tsv(i, FP32, NFEATURES, neurons)
        m.to_binfile(binfile.encode("ascii"))
        return m

def load_categories(neurons, nlayers, dest):
    fname = "{}/neuron{}-l{}-categories.tsv"
    cats = Path(fname.format(dest, neurons, nlayers))
    result = Vector.sparse(BOOL, NFEATURES)
    with cats.open() as i:
        for line in i.readlines():
            result[int(line.strip()) - 1] = True
    return result

def load_layer(neurons, dest, i):
    fname = "{}/neuron{}/n{}-l{}.{}"
    binfile = fname.format(dest, neurons, neurons, str(i + 1), "ssb")
    if Path(binfile).exists():
        return Matrix.from_binfile(binfile.encode("ascii"))
    l = Path(fname.format(dest, neurons, neurons, str(i + 1), "tsv"))
    with l.open() as f:
        m = Matrix.from_tsv(f, FP32, neurons, neurons)
        m.to_binfile(binfile.encode("ascii"))
        return m

@timing
def load_layers(neurons, dest, nlayers):
    with ThreadPool(cpu_count()) as pool:
        return pool.map(partial(load_layer, neurons, dest), range(nlayers))

@timing
def generate_bias(neurons, nlayers):
    result = []
    for i in range(nlayers):
        bias = Matrix.sparse(FP32, neurons, neurons)
        for i in range(neurons):
            bias[i, i] = BIAS[neurons]
        bias.nvals  # causes async completion
        result.append(bias)
    return result
