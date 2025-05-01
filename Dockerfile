FROM nvidia/cuda:11.8.0-base-ubuntu22.04

RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    git \
    cmake \
    python3-venv \
    python3.10 \
    python3-pip && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/nnue-pytorch

WORKDIR /workspace/nnue-pytorch

COPY ./setup_script.sh /workspace/nnue-pytorch/

ENTRYPOINT [ "/workspace/nnue-pytorch/setup_script.sh" ]
CMD ["/bin/bash"]