from setuptools import setup
import os

setup(
    name='thdnn',
    version='0.0.1',
    description='Tropical Hypersparse Deep Neural Networks with pygraphblas.',
    author='Michel Pelletier',
    packages=['dnn'],
    setup_requires=["pygraphblas"],
    install_requires=["pygraphblas"],
)

