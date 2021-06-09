FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
# FROM nvcr.io/nvidia/pytorch:20.12-py3
RUN apt-get update && apt-get install -y g++
COPY . /fairseq
WORKDIR /fairseq
RUN pip install --editable ./ && pip install numpy==1.20.0 tensorboard wandb
WORKDIR /workspace
CMD /bin/bash