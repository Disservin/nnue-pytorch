#!/bin/bash

uv sync
uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124

# Install Triton
git clone https://github.com/triton-lang/triton.git --depth 1
cd triton
uv pip install ninja cmake wheel pybind11  # build-time dependencies
# uv pip install ninja wheel  # build-time dependencies
uv pip install -e .
cd ..

uv pip install -e .

rm -rf triton