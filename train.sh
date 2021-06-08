if [ $# -eq 0 ]
    then
        echo "Usage: ./dnn.sh data_dir [neurons layers]

Example: ./dnn.sh dnn_data 1024 120
"
        exit 1
fi

docker run --rm --env DEST=/$1 --env NEURONS=$2 --env LAYERS=$3 \
       -v `pwd`/$1:/$1 \
       -v `pwd`/dnn:/pygraphblas/dnn/ \
       -it graphblas/pygraphblas-minimal:test ipython -i -m dnn train

