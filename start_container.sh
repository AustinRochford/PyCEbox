#! /bin/bash

PORT=${PORT:-8888}
SRC_DIR=${SRC_DIR:-`pwd`}
NOTEBOOK_DIR=${NOTEBOOK_DIR:-$SRC_DIR/notebooks}

docker build -t pycebox $SRC_DIR
docker run -d \
    -p $PORT:8888 \
    -v $SRC_DIR:/home/jovyan/pycebox/ \
    -v $NOTEBOOK_DIR:/home/jovyan/work/ \
    --name pycebox pycebox \
    start-notebook.sh --NotebookApp.token=''
