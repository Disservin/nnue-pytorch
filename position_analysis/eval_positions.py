#!/usr/bin/env python3
"""Evaluate the positions of a TSV (from extract_positions) with a .nnue net.

Adds the columns
  net_eval   the net's static evaluation, side-to-move perspective, internal units (same scale as the
             binpack score; the forward pass fake-quantizes like the engine by default)
  ls_index   raw layer-stack index the data loader assigned to the position
  ls_used    stack whose weights the net uses for it (differs from ls_index when the net routes a rare
             stack to a fallback stack instead of storing it)

This is the only step that needs this repo (model code + compiled data loader library) and torch.

Example:
  python position_analysis/eval_positions.py --net nn-bddd9d57eb87.nnue positions.tsv positions.eval.tsv
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="TSV from extract_positions")
    ap.add_argument("output", help="TSV with net_eval / ls_index added")
    ap.add_argument("--net", required=True, help=".nnue file")
    ap.add_argument("--repo", default=REPO, help="nnue-pytorch checkout (default: parent of this script)")
    ap.add_argument("--features", default=None, help="feature set (default: the repo default)")
    ap.add_argument("--ls-keys", default=None, help="layer-stack keys (default: the repo default)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--no-fake-quantize", action="store_true",
                    help="evaluate the float net instead of mimicking the engine's quantization")
    args = ap.parse_args()

    sys.path.insert(0, args.repo)
    import data_loader
    import model as M

    cfg = M.NNUEConfig()
    features = args.features or cfg.features
    ls_keys = args.ls_keys or cfg.model_config.ls_keys
    # The '^' suffix marks factorized features on the python side only; the loader knows plain names.
    loader_features = features.replace("^", "")

    t0 = time.time()
    with open(args.net, "rb") as f:
        net = M.NNUEReader(f, features, cfg.model_config).model
    net.to(args.device).eval()
    nnue2score = net.quantization.nnue2score
    print(f"loaded {args.net} in {time.time() - t0:.1f}s "
          f"({net.layer_stacks.count}/{net.layer_stacks.raw_count} layer stacks stored)", file=sys.stderr)

    df = pd.read_csv(args.input, sep="\t", keep_default_na=False, na_values={"next_score": [""]})
    fens = df["fen"].tolist()
    n = len(fens)
    evals = np.empty(n, dtype=np.float32)
    ls_idx = np.empty(n, dtype=np.int32)
    fq = not args.no_fake_quantize

    t0 = time.time()
    with torch.no_grad():
        for start in range(0, n, args.batch_size):
            chunk = fens[start:start + args.batch_size]
            k = len(chunk)
            batch = data_loader.get_sparse_batch_from_fens(
                loader_features, chunk, [0] * k, [1] * k, [0] * k, ls_keys=ls_keys
            )
            if not batch:
                raise RuntimeError(
                    f"data loader rejected feature set {loader_features!r} / ls keys {ls_keys!r}; "
                    "is the compiled loader library up to date with the repo?"
                )
            try:
                us, them, wi, bi, _o, _s, ls = batch.contents.get_tensors(args.device)
                out = net.forward(us, them, wi, bi, ls,
                                  fake_quantize_acts=fq, fake_quantize_weights=fq) * nnue2score
                evals[start:start + k] = out.flatten().float().cpu().numpy()
                ls_idx[start:start + k] = ls.cpu().numpy()
            finally:
                data_loader.destroy_sparse_batch(batch)
            done = start + k
            if (sys.stderr.isatty() and (start // args.batch_size) % 20 == 0) or done == n:
                rate = done / max(time.time() - t0, 1e-9)
                print(f"\r{done}/{n} positions ({rate:,.0f}/s)", end="", file=sys.stderr)
    print(file=sys.stderr)

    df["net_eval"] = np.round(evals, 1)
    df["ls_index"] = ls_idx
    fallback = net.layer_stacks.stack_fallback.cpu().numpy()
    df["ls_used"] = fallback[ls_idx]
    df.to_csv(args.output, sep="\t", index=False)
    print(f"wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
