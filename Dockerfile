FROM nvcr.io/nvidia/pytorch:22.12-py3

RUN apt-get update && apt-get install -y \
    git \
    curl \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/nnue-pytorch

WORKDIR /workspace/nnue-pytorch

COPY setup_script.sh .

ENTRYPOINT ["/workspace/nnue-pytorch/setup_script.sh"]
CMD ["/bin/bash"]