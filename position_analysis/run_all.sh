#!/usr/bin/env bash
# Full pipeline: sample positions from binpacks -> evaluate them with a net -> classify + report.
#
# usage: position_analysis/run_all.sh NET.nnue OUT_DIR file1.binpack [file2.binpack ...]
#
# Environment overrides:
#   COUNT=1000000     positions to sample
#   PER_CHUNK=2000    positions per sampled binpack chunk
#   SEED=1
#   EXTRACT_ARGS=""   extra extract_positions options (e.g. "--filtered" or "--by-size")
#   EVAL_ARGS=""      extra eval_positions.py options (e.g. "--device cpu")
#   CLASSIFY_ARGS=""  extra classify_positions.py options (e.g. "--quiet-only --fails 0.2")
#   PYTHON=python3
set -euo pipefail

if [ $# -lt 3 ]; then
    sed -n '2,15p' "$0"
    exit 2
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NET="$1"; OUT="$2"; shift 2
PYTHON="${PYTHON:-python3}"
mkdir -p "$OUT"

if [ ! -x "$HERE/extract_positions" ] || [ "$HERE/extract_positions.cpp" -nt "$HERE/extract_positions" ]; then
    "$HERE/build.sh"
fi

# shellcheck disable=SC2086
"$HERE/extract_positions" -n "${COUNT:-1000000}" --per-chunk "${PER_CHUNK:-2000}" --seed "${SEED:-1}" \
    ${EXTRACT_ARGS:-} -o "$OUT/positions.tsv" "$@"
printf '%s\n' "$@" | nl -v0 -w1 -s$'\t' > "$OUT/sources.tsv"   # source index -> file

# shellcheck disable=SC2086
"$PYTHON" "$HERE/eval_positions.py" --net "$NET" ${EVAL_ARGS:-} "$OUT/positions.tsv" "$OUT/positions.eval.tsv"

# shellcheck disable=SC2086
"$PYTHON" "$HERE/classify_positions.py" ${CLASSIFY_ARGS:-} --out-dir "$OUT" "$OUT/positions.eval.tsv"

echo "report: $OUT/report.md"
