FROM nvcr.io/nvidia/pytorch:25.03-py3

RUN apt-get update && apt-get install -y \
    g++ \
    git \
    curl \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    psutil \
    asciimatics \
    GPUtil \
    "python-chess==0.31.4" \
    matplotlib \
    tensorboard \
    numba \
    "numpy<2.0" \
    requests \
    pytorch-lightning

WORKDIR /workspace/nnue-pytorch

COPY setup_script.sh .

RUN chmod +x setup_script.sh

ENTRYPOINT [ "/workspace/nnue-pytorch/setup_script.sh" ]
CMD ["/bin/bash"]