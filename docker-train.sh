#!/bin/bash
IMAGE=fairseq:torch-1.6.0  # docker image name
SHM_SIZE=5g   # share memory size
NAME="fairseq"
USER=1005  # change for different machine
GPU='"device=1,2"'
TRAIN_FILE=$1
echo "[Docker] Train on $TRAIN_FILE"
docker run --gpus $GPU -it --rm --name=$NAME --shm-size=$SHM_SIZE --user=$USER --ulimit memlock=-1 -v $PWD:/workspace $IMAGE bash $TRAIN_FILE