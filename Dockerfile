ARG BASE_CONTAINER=graphblas/pygraphblas-notebook:test
FROM ${BASE_CONTAINER}
ADD . /home/jovyan/dnn
WORKDIR /home/jovyan/dnn
USER root
RUN python3 setup.py develop
