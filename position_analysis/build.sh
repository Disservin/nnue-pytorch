#!/usr/bin/env bash
# Build the binpack position extractor. Needs a C++20 compiler; uses the binpack/chess headers
# from the repo (data_loader/cpp/lib).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIB="${NNUE_LIB_DIR:-$HERE/../data_loader/cpp/lib}"
CXX="${CXX:-g++}"
"$CXX" -std=c++20 -O3 -march=native -DNDEBUG -pthread -I"$LIB" \
    "$HERE/extract_positions.cpp" -o "$HERE/extract_positions"
echo "built $HERE/extract_positions"
