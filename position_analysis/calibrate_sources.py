#!/usr/bin/env python3
"""Check whether the training sources put their search scores on different scales.

For each source (binpack) the script fits the win/draw/loss model that the training loss uses for the target
score to that source's own game results:

    P(win)  = sigmoid(( s - offset) / scaling)
    P(loss) = sigmoid((-s - offset) / scaling)
    P(draw) = 1 - P(win) - P(loss)

(the loss's pf = P(win) + P(draw) / 2 is exactly the trainer's wdl(score, out_offset, out_scaling)).
The fit maximizes the multinomial likelihood of the W/D/L results.

Three fits:

  pooled      one (offset, scaling) for all sources together; compare it with the loss's --out-offset and
              --out-scaling.
  per source  a free (offset, scaling) for each source.
  scale k     the pooled (offset, scaling) with the source's scores multiplied by k. k is the factor that
              puts the source on the pooled scale: k > 1 means its scores are too small for how often its
              games are won (the same advantage is reported with fewer cp), k < 1 means too large. Multiplying
              the source's scores by k is the offline rescale.

Uncertainty: positions of one game share a result, and the games of one binpack chunk are neighbours, so the
positions are not independent. The script resamples whole chunks (a cluster bootstrap) and also fits two
random halves of the chunks separately. extract_positions writes the positions of a chunk as one contiguous
block, so a cluster is a contiguous run of rows from the same source; neighbouring chunks of the same source
merge into one cluster, which only makes the intervals wider.

A source "differs" only when the whole 95% interval of k lies outside 1 +- --tol: statistical significance
alone is not enough, because with ~100k positions per source even tiny, harmless differences are significant.

Input: positions.tsv from extract_positions, or positions.eval.tsv (only fen, score, ply, result, prev_score,
capture, in_check and source are used; no net is needed). Needs numpy, pandas and scipy.

Example:
  python position_analysis/calibrate_sources.py results/positions.tsv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

# ---------------------------------------------------------------------------------------------------------
# W/D/L model


def _probs(s, offset, scaling):
    pw = expit((s - offset) / scaling)
    pl = expit((-s - offset) / scaling)
    pd_ = np.clip(1.0 - pw - pl, 1e-12, None)
    return pw, pl, pd_


def expected_score(s, offset, scaling):
    pw, _pl, pd_ = _probs(np.asarray(s, dtype=float), offset, scaling)
    return pw + 0.5 * pd_


def _nll_terms(s, y, w, offset, scaling, k):
    """Weighted NLL of the W/D/L results y (1, 0, -1) for scores k * s, and the derivatives of the per-position
    log-likelihood with respect to the arguments u = (k s - offset) / scaling and v = (-k s - offset) / scaling."""
    x = k * s
    u = (x - offset) / scaling
    v = (-x - offset) / scaling
    pw, pl = expit(u), expit(v)
    pd_ = np.clip(1.0 - pw - pl, 1e-12, None)
    win, loss = y == 1, y == -1
    draw = ~(win | loss)
    ll = np.where(
        win,
        np.log(np.clip(pw, 1e-12, None)),
        np.where(loss, np.log(np.clip(pl, 1e-12, None)), np.log(pd_)),
    )
    # d log p / du and d log p / dv for the observed outcome
    dpw, dpl = pw * (1 - pw), pl * (1 - pl)
    gu = np.where(win, 1 - pw, np.where(draw, -dpw / pd_, 0.0))
    gv = np.where(loss, 1 - pl, np.where(draw, -dpl / pd_, 0.0))
    return -(w * ll).sum(), w * gu, w * gv, u, v


# The optimizer works on well-scaled parameters (offset / 100, log scaling, log k) with tight tolerances: in
# raw cp units a small move of the optimum changes the objective by less than the default tolerance, and the
# bootstrap refits would stop at their starting point.
_OPTS = {"ftol": 1e-14, "gtol": 1e-10, "maxiter": 500}


def fit_free(s, y, w, start=(270.0, 380.0)):
    """Free (offset, scaling)."""
    W = w.sum()

    def f(p):
        offset, scaling = 100.0 * p[0], np.exp(p[1])
        nll, gu, gv, u, v = _nll_terms(s, y, w, offset, scaling, 1.0)
        # du/doffset = dv/doffset = -1/scaling; du/dscaling = -u/scaling, dv/dscaling = -v/scaling
        g_off = ((gu + gv) / scaling).sum() * 100.0
        g_sc = ((gu * u + gv * v) / scaling).sum() * scaling
        return nll / W, np.array([g_off, g_sc]) / W

    x0 = np.array([start[0] / 100.0, np.log(start[1])])
    r = minimize(
        f,
        x0,
        jac=True,
        method="L-BFGS-B",
        bounds=[(0.01, 30.0), (np.log(10.0), np.log(5000.0))],
        options=_OPTS,
    )
    return np.array([100.0 * r.x[0], np.exp(r.x[1])])


def fit_scale(s, y, w, offset, scaling, start=1.0):
    """Scale k of the scores under a fixed (offset, scaling)."""
    W = w.sum()

    def f(p):
        k = np.exp(p[0])
        nll, gu, gv, _, _ = _nll_terms(s, y, w, offset, scaling, k)
        # du/dk = s/scaling, dv/dk = -s/scaling
        g = -((gu - gv) * s / scaling).sum() * k
        return nll / W, np.array([g]) / W

    r = minimize(
        f,
        np.array([np.log(start)]),
        jac=True,
        method="L-BFGS-B",
        bounds=[(np.log(0.2), np.log(5.0))],
        options=_OPTS,
    )
    return float(np.exp(r.x[0]))


def rescale_mismatch(offset, scaling, k, off0, sc0, lo=0.0, hi=1200.0):
    """Largest gap in expected score over [lo, hi] cp between the source's own curve and the pooled curve
    after multiplying the scores by k: what a single rescale factor cannot fix."""
    s = np.linspace(lo, hi, 241)
    return float(
        np.abs(
            expected_score(s, offset, scaling) - expected_score(k * s, off0, sc0)
        ).max()
    )


# ---------------------------------------------------------------------------------------------------------
# data


def load(path, a):
    cols = [
        "fen",
        "score",
        "ply",
        "result",
        "prev_score",
        "capture",
        "in_check",
        "source",
    ]
    df = pd.read_csv(path, sep="\t", usecols=cols)
    n0 = len(df)
    notes = [f"- input: `{path}`, {n0} positions"]

    # Clusters: contiguous runs of rows of one source (= one chunk, or neighbouring chunks of that source).
    # Assigned before any filtering so that a filtered-out row cannot join two chunks.
    df["cluster"] = (df["source"] != df["source"].shift()).cumsum()

    keep = df["result"].isin([-1, 0, 1])
    notes.append(f"- no game result: {int((~keep).sum())} excluded")
    if not a.all_positions:
        quiet = (df["capture"] == 0) & (df["in_check"] == 0)
        notes.append(
            f"- capture / in-check (not in --filtered training): {int((keep & ~quiet).sum())} excluded"
        )
        keep &= quiet
    # the trainer's placeholder-zero heuristic: score 0 in a decisive game right after |score| > 100
    prv = pd.to_numeric(df["prev_score"], errors="coerce").fillna(0).abs()
    placeholder = (df["score"] == 0) & (df["result"] != 0) & (prv > 100)
    notes.append(f"- placeholder score 0: {int((keep & placeholder).sum())} excluded")
    keep &= ~placeholder
    pieces = df["fen"].str.split(" ", n=1).str[0].str.count(r"[a-zA-Z]")
    small = pieces < a.min_pieces
    notes.append(
        f"- fewer than {a.min_pieces} pieces: {int((keep & small).sum())} excluded"
    )
    keep &= ~small
    big = df["score"].abs() > a.max_score
    notes.append(f"- |score| > {a.max_score}: {int((keep & big).sum())} excluded")
    keep &= ~big

    df = df[keep].reset_index(drop=True)
    notes.append(
        f"- fitted: **{len(df)}** positions, {df['cluster'].nunique()} chunk clusters"
    )
    return df, notes


def source_names(path, input_path):
    p = path or os.path.join(
        os.path.dirname(os.path.abspath(input_path)), "sources.tsv"
    )
    names = {}
    if os.path.exists(p):
        with open(p) as f:
            for line in f:
                parts = line.rstrip("\n").split("\t", 1)
                if len(parts) == 2 and parts[0].strip().isdigit():
                    names[int(parts[0])] = os.path.basename(parts[1])
    return names


# ---------------------------------------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("input", help="positions.tsv or positions.eval.tsv")
    ap.add_argument(
        "--sources",
        default=None,
        help="TSV 'index<TAB>binpack path' (default: sources.tsv next to the input)",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="markdown report (default: calibration.md next to the input)",
    )
    ap.add_argument(
        "--min-pieces",
        type=int,
        default=8,
        help="drop positions with fewer pieces incl. kings (default 8: no <=7-piece endgames)",
    )
    ap.add_argument(
        "--max-score",
        type=int,
        default=2000,
        help="drop |score| above this (mate / TB scores, decided positions; default 2000)",
    )
    ap.add_argument(
        "--all-positions",
        action="store_true",
        help="keep capture and in-check positions (default: quiet only, as with --filtered)",
    )
    ap.add_argument(
        "--bootstrap",
        type=int,
        default=200,
        help="cluster bootstrap samples (default 200)",
    )
    ap.add_argument(
        "--tol",
        type=float,
        default=0.05,
        help="a scale k within 1 +- tol counts as the same scale (default 0.05)",
    )
    ap.add_argument(
        "--out-offset",
        type=float,
        default=270.0,
        help="the loss's out_offset (default 270)",
    )
    ap.add_argument(
        "--out-scaling",
        type=float,
        default=380.0,
        help="the loss's out_scaling (default 380)",
    )
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    names = source_names(a.sources, a.input)
    df, notes = load(a.input, a)
    if df.empty:
        sys.exit("no positions left after filtering")

    s_all = df["score"].to_numpy(float)
    y_all = df["result"].to_numpy()
    w_all = np.ones(len(df))
    off0, sc0 = fit_free(s_all, y_all, w_all)

    sources = sorted(df["source"].unique())
    rows = []
    for src in sources:
        d = df[df["source"] == src]
        s, y = d["score"].to_numpy(float), d["result"].to_numpy()
        cl = pd.factorize(d["cluster"])[0]
        ncl = cl.max() + 1
        ones = np.ones(len(d))

        off, sc = fit_free(s, y, ones, (off0, sc0))
        k = fit_scale(s, y, ones, off0, sc0)

        # cluster bootstrap: each chunk cluster gets a multinomial count
        ks, offs, scs = [], [], []
        for _ in range(a.bootstrap):
            cw = rng.multinomial(ncl, np.full(ncl, 1.0 / ncl)).astype(float)
            w = cw[cl]
            ks.append(fit_scale(s, y, w, off0, sc0, k))
            o_, c_ = fit_free(s, y, w, (off, sc))
            offs.append(o_)
            scs.append(c_)
        k_lo, k_hi = np.percentile(ks, [2.5, 97.5])

        # split half by clusters
        half = rng.permutation(ncl) < ncl // 2
        hm = half[cl]
        k_h1 = fit_scale(s[hm], y[hm], ones[hm], off0, sc0, k) if hm.any() else np.nan
        k_h2 = (
            fit_scale(s[~hm], y[~hm], ones[~hm], off0, sc0, k)
            if (~hm).any()
            else np.nan
        )

        if k_lo > 1 + a.tol or k_hi < 1 - a.tol:
            verdict = "differs"
        elif k_lo >= 1 - a.tol and k_hi <= 1 + a.tol:
            verdict = "same"
        else:
            verdict = "unclear"

        m = (np.abs(s) > 300) & (np.abs(s) <= 1200)
        win_ahead = (
            float((np.sign(y[m]) == np.sign(s[m])).mean()) if m.any() else np.nan
        )
        draw_ahead = float((y[m] == 0).mean()) if m.any() else np.nan
        rows.append(
            {
                "source": src,
                "name": names.get(src, str(src)),
                "n": len(d),
                "clusters": ncl,
                "k": k,
                "k_lo": k_lo,
                "k_hi": k_hi,
                "k_half1": k_h1,
                "k_half2": k_h2,
                "offset": off,
                "offset_lo": np.percentile(offs, 2.5),
                "offset_hi": np.percentile(offs, 97.5),
                "scaling": sc,
                "scaling_lo": np.percentile(scs, 2.5),
                "scaling_hi": np.percentile(scs, 97.5),
                "es300": float(expected_score(300, off, sc)),
                "es600": float(expected_score(600, off, sc)),
                "gap_before": rescale_mismatch(off, sc, 1.0, off0, sc0),
                "gap_after": rescale_mismatch(off, sc, k, off0, sc0),
                "win_300_1200": win_ahead,
                "draw_300_1200": draw_ahead,
                "verdict": verdict,
            }
        )
        print(
            f"source {src}: k={k:.3f} [{k_lo:.3f}, {k_hi:.3f}] {verdict}",
            file=sys.stderr,
        )

    res = pd.DataFrame(rows)

    out = a.out or os.path.join(
        os.path.dirname(os.path.abspath(a.input)), "calibration.md"
    )
    tsv = os.path.splitext(out)[0] + ".tsv"
    res.to_csv(tsv, sep="\t", index=False, float_format="%.4f")

    es_loss = (
        expected_score(300, a.out_offset, a.out_scaling),
        expected_score(600, a.out_offset, a.out_scaling),
    )
    es_pool = (expected_score(300, off0, sc0), expected_score(600, off0, sc0))
    L = ["# Per-source score calibration", ""]
    L += notes
    L += [
        (
            f"- uncertainty: {a.bootstrap} cluster-bootstrap samples (whole chunks resampled), 95% intervals; "
            f"`half 1 / half 2` = k fitted on two random halves of the chunks"
        ),
        (
            f"- verdict: `differs` = the whole 95% interval of k is outside 1 +- {a.tol}, `same` = inside, "
            f"`unclear` = it crosses a bound"
        ),
        "",
    ]
    L += [
        "## Pooled fit",
        "",
        "| | offset | scaling | E[score] at +300 | E[score] at +600 |",
        "|---|---|---|---|---|",
        (
            f"| loss setting (--out-offset / --out-scaling) | {a.out_offset:.0f} | {a.out_scaling:.0f} | "
            f"{es_loss[0]:.3f} | {es_loss[1]:.3f} |"
        ),
        (
            f"| fitted to the game results, all sources | {off0:.0f} | {sc0:.0f} | {es_pool[0]:.3f} | "
            f"{es_pool[1]:.3f} |"
        ),
        "",
    ]
    L += [
        "## Scale k per source (pooled offset and scaling, scores multiplied by k)",
        "",
        "k > 1: the source's cp are too small for how often its games are won; k < 1: too large.",
        "",
        (
            "`gap before / after` = the largest difference in expected score over 0-1200 cp between the "
            "source's own curve (free fit below) and the pooled curve, before and after multiplying the scores "
            "by k. A large gap after means the source's curve has a different shape and one factor can't fix it."
        ),
        "",
        "| source | n | chunks | k | 95% interval | half 1 / half 2 | gap before / after | verdict |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        L.append(
            f"| {r['source']}: {r['name']} | {r['n']} | {r['clusters']} | **{r['k']:.3f}** | "
            f"{r['k_lo']:.3f} - {r['k_hi']:.3f} | {r['k_half1']:.3f} / {r['k_half2']:.3f} | "
            f"{r['gap_before']:.3f} / {r['gap_after']:.3f} | {r['verdict']} |"
        )
    L += [
        "",
        "## Free fit per source",
        "",
        (
            "E[score] = expected score (win + draw / 2) of the side to move at that search score, from the "
            "source's own fit. `win / draw 300-1200` are the raw rates of the side the score favours."
        ),
        "",
        "| source | offset (95%) | scaling (95%) | E at +300 | E at +600 | win 300-1200 | draw 300-1200 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        L.append(
            f"| {r['source']}: {r['name']} | {r['offset']:.0f} ({r['offset_lo']:.0f}-{r['offset_hi']:.0f}) | "
            f"{r['scaling']:.0f} ({r['scaling_lo']:.0f}-{r['scaling_hi']:.0f}) | {r['es300']:.3f} | "
            f"{r['es600']:.3f} | {r['win_300_1200']:.1%} | {r['draw_300_1200']:.1%} |"
        )
    L += [
        "",
        "## Reading it",
        "",
        (
            "- This measures what a score means *in terms of game results*. With lambda < 1 the results enter "
            "the loss, so a source whose results disagree with the pooled curve pulls the net toward its own "
            "curve. With lambda = 1 only the scores enter the loss, and k says how far the source's scores are "
            "from the scale the other sources' results imply."
        ),
        (
            "- A source can differ because its scores are scaled differently, or because its games were played "
            "differently (e.g. games played by one engine and rescored by another, adjudication, opening books, "
            "noise or temperature). The fit can't tell these apart."
        ),
        f"- Machine-readable results: `{os.path.basename(tsv)}`.",
        "",
    ]
    with open(out, "w") as f:
        f.write("\n".join(L))
    print(f"report: {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
