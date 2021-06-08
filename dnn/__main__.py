import os
import argparse
from . import challenge
from . import radix
from . import training

DEFAULT_NEURONS = [1024, 4096, 16384, 65536]
DEFAULT_LAYERS = [120, 480, 1920]

DEST = os.getenv("DEST")
NEURONS = os.getenv("NEURONS")
LAYERS = os.getenv("LAYERS")
LOG_LEVEL = os.getenv("LOG_LEVEL")

parser = argparse.ArgumentParser(description="CoinBLAS")
parser.add_argument("mode", default="run", help="run|train")
parser.add_argument("--dest", default=DEST, help="Destination directory")
parser.add_argument("--neurons", default=NEURONS, help="Number of Neurons")
parser.add_argument("--layers", default=LAYERS, help="Number of Layers")
parser.add_argument("--log-level", default=LOG_LEVEL, help="Log level.")
args = parser.parse_args()


neurons = int(args.neurons)
nlayers = int(args.layers)

if args.mode == 'train':
    training.train(args.dest, neurons, nlayers)
elif args.mode == 'run':
    challenge.run(args.dest, neurons, nlayers)

