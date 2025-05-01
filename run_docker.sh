#!/bin/bash

docker build -t nnue-pytorch .

if [ "$(docker ps -aq -f name=nnue-container)" ]; then
  if [ ! "$(docker ps -q -f name=nnue-container)" ]; then
    echo "Starting existing container 'nnue-container'..."
    docker start -i nnue-container
  else
    echo "Container 'nnue-container' is already running. Attaching..."
    docker attach nnue-container
  fi
else
  echo "Creating new container 'nnue-container'..."
  docker run -it --name nnue-container \
    --gpus all \
    -v .:/workspace/nnue-pytorch \
    -v /mnt/g/stockfish-data:/workspace/data \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    nnue-pytorch
fi