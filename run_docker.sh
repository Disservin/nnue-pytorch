#!/bin/bash

docker build -t nnue-pytorch .

docker run -it --name nnue-container --gpus all -v .:/workspace/nnue-pytorch -v /mnt/g/stockfish-data:/workspace/data --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 nnue-pytorch