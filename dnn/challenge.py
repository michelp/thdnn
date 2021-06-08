from pygraphblas import Vector, BOOL

from . import timing
from . import io
from .dnn import dnn, hyperdnn
from .radix import hypergraph


@timing
def classify(neurons, images, layers, bias, dest):
    breakpoint()
    result = dnn(layers, bias, images)
    r = result.reduce_vector()
    cats = r.apply(BOOL.ONE, out=Vector.sparse(BOOL, r.size))
    truecats = io.load_categories(neurons, len(layers), dest)
    if cats.iseq(truecats):
        print('SUCCESS')
        breakpoint()
    else:
        print('FAIL: Result does not match ground truth categories.')
        breakpoint()

def run(dest, neurons, nlayers):
    if neurons and nlayers:
        images = io.load_images(neurons, dest)
        layers = io.load_layers(neurons, dest, nlayers)
        bias = io.generate_bias(neurons, nlayers)
        classify(neurons, images, layers, bias, dest)
    else:
        for neurons in DEFAULT_NEURONS:
            print("Building layers for %s neurons" % neurons)
            layers = io.load_layers(neurons, 1920, dest)
            bias = io.generate_bias(neurons, 1920)
            images = io.load_images(neurons, dest)
            for nlayers in DEFAULT_LAYERS:
                print(f"Benching {neurons} neurons {nlayers} layers")
                classify(neurons, images, layers[:nlayers], bias[:nlayers], dest)

    
