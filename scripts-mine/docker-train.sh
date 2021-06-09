#!/bin/bash
IMAGE=fairseq:torch-1.6.0  # docker image name
SHM_SIZE=5g   # share memory size
NAME="fairseq"
USER=1052  # change for different machine
GPU='"device=0,1,2,3"'
TRAIN_FILE=$1
echo "[Docker] Train on $TRAIN_FILE"
docker run --gpus $GPU -it --rm --name=$NAME --shm-size=$SHM_SIZE --ulimit memlock=-1 -v $PWD:/workspace $IMAGE bash $TRAIN_FILE