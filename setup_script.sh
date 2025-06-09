#!/bin/bash

if [ ! -d ".venv" ]; then
    python3 -m venv .venv --system-site-packages
    source .venv/bin/activate
    # pip --no-cache-dir install torch --index-url https://download.pytorch.org/whl/cu118
    pip install -r requirements.txt
    python3 -m pip install --upgrade pip
fi

echo "source /workspace/nnue-pytorch/.venv/bin/activate" >> /root/.bashrc
sh compile_data_loader.bat

exec "$@"