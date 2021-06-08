from . import io
from . import radix
from . import challenge
from .dnn import dnn

neurons_to_spec = {
    1024: ([[2]*6], [16]*7, (1/16.0), 20)
    }

def train(dest, neurons, nlayers):
    print(f"Training {spec} with {kron} layers to {dest}")
    images = io.load_images(neurons, dest)
    bias = io.generate_bias(neurons, nlayers)
    spec, kron, default, depth = neurons_to_spec[neurons]
    layers = radix.radixnet(spec, kron, default, initializer=radix.random_op)
    result = dnn(layers, bias, images)
    breakpoint()
