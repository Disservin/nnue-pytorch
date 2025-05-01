#!/bin/bash

python3 -m venv .venv --system-site-packages
source .venv/bin/activate
# pip --no-cache-dir install torch --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

exec "$@"