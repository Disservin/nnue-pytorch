FROM nvcr.io/nvidia/pytorch:24.01-py3

RUN apt-get update && apt-get install -y \
    git \
    curl \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/nnue-pytorch

WORKDIR /workspace/nnue-pytorch

COPY setup_script.sh .

CMD ["setup_script.sh"]