
docker run -p 8888:8888 --env DEST=/$1 --env NEURONS=$2 --env LAYERS=$3 \
       -v `pwd`/$1:/$1 \
       -v `pwd`/:/dnn/ \
       -w /dnn \
       -it graphblas/thdnn:test

