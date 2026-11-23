#!/usr/bin/env python3
"""Classify positions and measure where a net learns the target, where it struggles and where it fails.

Input: a TSV from eval_positions.py (columns fen, move, score, ply, result, next_score, prev_score, capture,
in_check, source, net_eval, ls_index, ls_used). Needs only numpy, pandas and python-chess.

Per position
------------
The net's eval and the search score are both mapped to an expected score (win probability + half the
draw probability) with exactly the conversions the training loss uses:

    q_net    = wdl(net_eval,           in_offset,  in_scaling)
    p_target = wdl(tb_remap(score),    out_offset, out_scaling)
    wdl(x, o, s) = 0.5 * (1 + sigmoid((x - o) / s) - sigmoid((-x - o) / s))

and the error is err = q_net - p_target (> 0: the net is too optimistic for the side to move).
Positions are labelled by |err|:

    learned   |err| <  --learned  (default 0.05)
    struggles |err| <  --fails    (default 0.15)
    fails     otherwise

Comparing in expected-score space means that 50 units of error matter a lot in a balanced position and
hardly at all at +1500, like in training.

Position classes
----------------
Every position gets a set of categorical features (material and imbalance, phase, endgame signature,
pawn structure, king safety, hanging pieces / threats, checks, mobility, the stored move type,
rule-50 counter, how stable the search score is around the position, agreement of score with material
and with the game result, layer stack, ...). For every feature and value the report gives the share of
learned / struggles / fails, the mean signed error (bias) and the share of the total training-style loss
(|err|^2.5) the class accounts for, and ranks the classes by how much more often than average the net
fails on them. The most interesting two-feature combinations are ranked the same way.

Outputs (in --out-dir)
----------------------
  report.md                 the full report
  positions_classified.tsv  every position with all features, err and label (for ad-hoc analysis)
  worst_positions.tsv       the worst positions overall and per hardest class, with FEN, to look at

Example:
  python position_analysis/classify_positions.py positions.eval.tsv --out-dir analysis/
"""

import argparse
import itertools
import os
import sys
import time
from multiprocessing import Pool

import chess
import numpy as np
import pandas as pd

VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
PIECE_LETTER = {chess.KING: "K", chess.QUEEN: "Q", chess.ROOK: "R", chess.BISHOP: "B", chess.KNIGHT: "N",
                chess.PAWN: "P"}
FILE_MASKS = [chess.BB_FILES[f] for f in range(8)]
LIGHT = chess.BB_LIGHT_SQUARES


popcount = int.bit_count  # python >= 3.10


# ---------------------------------------------------------------------------------------------------------
# Board features (python-chess)
# ---------------------------------------------------------------------------------------------------------


def bucket(x, edges, labels):
    for e, lab in zip(edges, labels):
        if x <= e:
            return lab
    return labels[-1]


def passed_pawns(board: chess.Board, color: bool):
    """Squares of the passed pawns of `color`."""
    own = board.pieces_mask(chess.PAWN, color)
    opp = board.pieces_mask(chess.PAWN, not color)
    res = []
    for sq in chess.scan_forward(own):
        f, r = chess.square_file(sq), chess.square_rank(sq)
        files = FILE_MASKS[f]
        if f > 0:
            files |= FILE_MASKS[f - 1]
        if f < 7:
            files |= FILE_MASKS[f + 1]
        if color == chess.WHITE:
            ahead = files & ~((1 << (8 * (r + 1))) - 1)
        else:
            ahead = files & ((1 << (8 * r)) - 1)
        if not (ahead & opp):
            res.append(sq)
    return res


def rel_rank(sq: int, color: bool) -> int:
    """1..8 from the point of view of `color`."""
    r = chess.square_rank(sq)
    return (r if color == chess.WHITE else 7 - r) + 1


def threat_gain(board: chess.Board, victim_color: bool) -> tuple[int, int, int]:
    """Material the opponent of `victim_color` can win by a single capture (crude SEE: full value if the
    victim is undefended, else value minus the cheapest attacker), the number of en-prise pieces and the
    number of completely undefended attacked pieces."""
    att_color = not victim_color
    best = 0
    en_prise = 0
    hanging = 0
    for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        for sq in chess.scan_forward(board.pieces_mask(pt, victim_color)):
            attackers = board.attackers_mask(att_color, sq)
            if not attackers:
                continue
            v = VAL[pt]
            cheapest = min(VAL[board.piece_type_at(a)] or 100 for a in chess.scan_forward(attackers))
            defended = bool(board.attackers_mask(victim_color, sq))
            if not defended:
                gain = v
                hanging += 1
            else:
                gain = v - cheapest
            if gain > 0:
                en_prise += 1
                best = max(best, gain)
    return best, en_prise, hanging


def king_danger(board: chess.Board, color: bool) -> tuple[int, int]:
    """Number of (attacker, square) pairs of the opponent on the king zone of `color`, and the number of
    own pawns sheltering the king (the 3 files around it, the 2 ranks in front)."""
    ksq = board.king(color)
    if ksq is None:
        return 0, 0
    zone = chess.BB_KING_ATTACKS[ksq] | chess.BB_SQUARES[ksq]
    attacks = 0
    for sq in chess.scan_forward(zone):
        attacks += popcount(board.attackers_mask(not color, sq))
    f, r = chess.square_file(ksq), chess.square_rank(ksq)
    shelter = 0
    for df in (-1, 0, 1):
        ff = f + df
        if not 0 <= ff <= 7:
            continue
        for dr in (1, 2):
            rr = r + dr if color == chess.WHITE else r - dr
            if 0 <= rr <= 7:
                p = board.piece_at(chess.square(ff, rr))
                if p is not None and p.piece_type == chess.PAWN and p.color == color:
                    shelter += 1
    return attacks, shelter


def endgame_signature(board: chess.Board, stm: bool) -> str:
    """e.g. 'KRPvKR', stronger side (by material, then side to move) first."""
    def side(c):
        s = "K"
        for pt in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN):
            s += PIECE_LETTER[pt] * popcount(board.pieces_mask(pt, c))
        return s

    def mat(c):
        return sum(VAL[pt] * popcount(board.pieces_mask(pt, c)) for pt in VAL)

    a, b = side(stm), side(not stm)
    if (mat(not stm), len(b)) > (mat(stm), len(a)):
        a, b = b, a
    return f"{a}v{b}"


def imbalance_class(d: dict) -> str:
    """Kind of material imbalance between the two sides (counts are side to move minus opponent)."""
    dq, dr, dm, dp = d["dQ"], d["dR"], d["dN"] + d["dB"], d["dP"]
    if dq == dr == dm == 0:
        if dp == 0:
            if d["dN"] != 0:
                return "equal_B_vs_N"
            return "equal"
        return "pawns_only_" + ("up" if dp > 0 else "down") + ("1" if abs(dp) == 1 else "2+")
    if dq == 0 and dr == 0:
        if abs(dm) == 1 and dp * dm < 0 and abs(dp) >= 2:
            return "minor_vs_pawns"
        return "piece_" + ("up" if dm > 0 else "down")
    if dq == 0 and abs(dr) == 1 and dm == -dr:
        return "exchange_" + ("up" if dr > 0 else "down")
    if dq == 0 and abs(dr) == 1 and dm == -2 * dr:
        return "rook_vs_2minors"
    if dq != 0 and (dr * dq < 0 or dm * dq < 0):
        return "queen_vs_pieces"
    up = 9 * dq + 5 * dr + 3 * dm + dp
    return "other_" + ("up" if up > 0 else "down" if up < 0 else "even")


def board_features(fen: str, move_uci: str) -> dict:
    b = chess.Board(fen)
    stm = b.turn
    opp = not stm
    d: dict = {}

    cnt = {}
    for pt, L in PIECE_LETTER.items():
        cnt[(stm, pt)] = popcount(b.pieces_mask(pt, stm))
        cnt[(opp, pt)] = popcount(b.pieces_mask(pt, opp))
        if pt != chess.KING:
            d["d" + L] = cnt[(stm, pt)] - cnt[(opp, pt)]

    n_pieces = popcount(b.occupied)
    npm = {c: sum(VAL[pt] * cnt[(c, pt)] for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN))
           for c in (stm, opp)}
    mat = {c: npm[c] + cnt[(c, chess.PAWN)] for c in (stm, opp)}
    d["n_pieces"] = n_pieces
    d["material_diff"] = mat[stm] - mat[opp]
    d["npm_total"] = npm[stm] + npm[opp]

    # --- material / phase -------------------------------------------------------------------------------
    d["f_piece_count"] = bucket(n_pieces, [7, 10, 14, 18, 22, 26, 29], ["<=7 (TB)", "8-10", "11-14", "15-18",
                                                                       "19-22", "23-26", "27-29", "30-32"])
    tot = d["npm_total"]
    if tot == 0:
        phase = "pawn_endgame"
    elif tot <= 13:
        phase = "endgame (npm<=13)"
    elif tot <= 30:
        phase = "late_middlegame (npm 14-30)"
    elif tot <= 50:
        phase = "middlegame (npm 31-50)"
    else:
        phase = "opening (npm>50)"
    d["f_phase"] = phase
    md = d["material_diff"]
    d["f_material_diff"] = bucket(md, [-9, -5, -3, -2, -1, 0, 1, 2, 4, 8],
                                  ["<=-9", "-8..-5", "-4..-3", "-2", "-1", "0", "+1", "+2", "+3..+4",
                                   "+5..+8", ">=+9"])
    d["f_imbalance"] = imbalance_class(d)
    qs, qo = cnt[(stm, chess.QUEEN)] > 0, cnt[(opp, chess.QUEEN)] > 0
    d["f_queens"] = {(True, True): "both", (False, False): "none", (True, False): "stm_only",
                     (False, True): "opp_only"}[(qs, qo)]
    bs, bo = cnt[(stm, chess.BISHOP)], cnt[(opp, chess.BISHOP)]
    if bs == 1 and bo == 1:
        sb = b.pieces_mask(chess.BISHOP, stm)
        ob = b.pieces_mask(chess.BISHOP, opp)
        ocb = bool(sb & LIGHT) != bool(ob & LIGHT)
        d["f_bishops"] = "opposite_colored" if ocb else "same_colored"
    elif bs >= 2 and bo < 2:
        d["f_bishops"] = "stm_bishop_pair"
    elif bo >= 2 and bs < 2:
        d["f_bishops"] = "opp_bishop_pair"
    elif bs == 0 and bo == 0:
        d["f_bishops"] = "no_bishops"
    else:
        d["f_bishops"] = "other"
    d["f_endgame"] = endgame_signature(b, stm) if n_pieces <= 8 else "-"

    # --- pawns ----------------------------------------------------------------------------------------
    pawns = b.pieces(chess.PAWN, chess.WHITE) | b.pieces(chess.PAWN, chess.BLACK)
    left = any(chess.square_file(s) < 4 for s in pawns)
    right = any(chess.square_file(s) >= 4 for s in pawns)
    d["f_pawn_wings"] = "both_wings" if left and right else "one_wing" if (left or right) else "no_pawns"
    pp_s, pp_o = passed_pawns(b, stm), passed_pawns(b, opp)
    adv_s = max((rel_rank(s, stm) for s in pp_s), default=0)
    adv_o = max((rel_rank(s, opp) for s in pp_o), default=0)
    d["f_passed_pawns"] = f"stm{min(len(pp_s), 2)}{'+' if len(pp_s) > 2 else ''}_opp{min(len(pp_o), 2)}" \
                          f"{'+' if len(pp_o) > 2 else ''}"
    d["f_passer_stm_most_advanced"] = "none" if adv_s == 0 else ("rank<=4" if adv_s <= 4 else f"rank{adv_s}")
    d["f_passer_opp_most_advanced"] = "none" if adv_o == 0 else ("rank<=4" if adv_o <= 4 else f"rank{adv_o}")

    # --- threats / tactics ----------------------------------------------------------------------------
    gain_stm, eprise_opp, hang_opp = threat_gain(b, opp)  # what stm can grab
    gain_opp, eprise_stm, hang_stm = threat_gain(b, stm)  # what stm must defend
    gb = [0, 2, 4, 8]
    gl = ["0", "1-2 (pawn+)", "3-4 (piece)", "5-8 (rook+)", "9+ (queen)"]
    d["f_stm_can_win"] = bucket(gain_stm, gb, gl)
    d["f_stm_threatened"] = bucket(gain_opp, gb, gl)
    d["f_stm_en_prise"] = bucket(eprise_stm, [0, 1], ["0", "1", "2+"])
    d["f_opp_en_prise"] = bucket(eprise_opp, [0, 1], ["0", "1", "2+"])
    d["gain_stm"] = gain_stm
    d["gain_opp"] = gain_opp
    d["hanging_stm"] = hang_stm
    d["hanging_opp"] = hang_opp

    pinned_stm = sum(1 for sq in chess.scan_forward(b.occupied_co[stm] & ~b.kings) if b.is_pinned(stm, sq))
    pinned_opp = sum(1 for sq in chess.scan_forward(b.occupied_co[opp] & ~b.kings) if b.is_pinned(opp, sq))
    d["f_pins"] = f"stm{min(pinned_stm, 1)}{'+' if pinned_stm > 1 else ''}_opp{min(pinned_opp, 1)}" \
                  f"{'+' if pinned_opp > 1 else ''}"

    legal = list(b.legal_moves)
    n_legal = len(legal)
    n_caps = sum(1 for m in legal if b.is_capture(m))
    n_checks = sum(1 for m in legal if b.gives_check(m))
    d["n_legal"] = n_legal
    d["f_mobility"] = bucket(n_legal, [1, 5, 15, 30, 45], ["1 (forced)", "2-5", "6-15", "16-30", "31-45", "46+"])
    d["f_captures_available"] = bucket(n_caps, [0, 2, 5], ["0", "1-2", "3-5", "6+"])
    d["f_checks_available"] = bucket(n_checks, [0, 2], ["0", "1-2", "3+"])
    d["f_in_check"] = "yes" if b.is_check() else "no"

    # promotion threats (pawn on the 7th, relative)
    p7s = any(rel_rank(s, stm) == 7 for s in b.pieces(chess.PAWN, stm))
    p7o = any(rel_rank(s, opp) == 7 for s in b.pieces(chess.PAWN, opp))
    d["f_pawn_on_7th"] = {(False, False): "none", (True, False): "stm", (False, True): "opp",
                          (True, True): "both"}[(p7s, p7o)]

    # --- king safety ----------------------------------------------------------------------------------
    ka_s, sh_s = king_danger(b, stm)
    ka_o, _ = king_danger(b, opp)
    kb = [0, 2, 5, 9]
    kl = ["0", "1-2", "3-5", "6-9", "10+"]
    d["f_stm_king_attacks"] = bucket(ka_s, kb, kl)
    d["f_opp_king_attacks"] = bucket(ka_o, kb, kl)
    d["f_stm_king_shelter"] = bucket(sh_s, [0, 1, 2], ["0", "1", "2", "3+"])
    ksq = b.king(stm)
    kr = rel_rank(ksq, stm) if ksq is not None else 1
    d["f_stm_king_rank"] = "back_rank" if kr == 1 else "rank2-3" if kr <= 3 else "rank4+"
    d["f_castling_rights"] = "any" if b.castling_rights else "none"

    # --- the stored move ------------------------------------------------------------------------------
    try:
        mv = chess.Move.from_uci(move_uci)
        ok = mv in b.legal_moves
    except ValueError:
        ok = False
    if not ok:
        mt = "invalid"
    elif b.is_castling(mv):
        mt = "castling"
    elif mv.promotion:
        mt = "promotion"
    elif b.is_capture(mv):
        mt = "capture+check" if b.gives_check(mv) else "capture"
    elif b.gives_check(mv):
        mt = "check"
    else:
        pt = b.piece_type_at(mv.from_square)
        mt = "quiet_" + PIECE_LETTER[pt]
    d["f_move_type"] = mt

    d["f_rule50"] = bucket(b.halfmove_clock, [9, 29, 59, 79], ["0-9", "10-29", "30-59", "60-79", "80+"])
    return d


def _features_chunk(rows):
    return [board_features(f, m) for f, m in rows]


# ---------------------------------------------------------------------------------------------------------
# Targets, errors and score-derived features (vectorised)
# ---------------------------------------------------------------------------------------------------------


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def wdl(x, offset, scaling):
    return 0.5 * (1.0 + sigmoid((x - offset) / scaling) - sigmoid((-x - offset) / scaling))


def tb_remap(score, base, scale, decay):
    mate_score, max_mate_ply = 32000, 245
    a = np.abs(score)
    is_mate = a >= mate_score - max_mate_ply
    plies = np.clip(mate_score - a, 0, None)
    rem = base + decay ** plies * scale
    return np.where(is_mate, np.sign(score) * rem, score)


def cut(x, edges, labels):
    """Bucket a float array: labels[i] for edges[i-1] < x <= edges[i]; NaN -> 'n/a'."""
    out = np.full(len(x), "n/a", dtype=object)
    x = np.asarray(x, dtype=float)
    valid = ~np.isnan(x)
    idx = np.searchsorted(np.asarray(edges, dtype=float), x[valid], side="left")
    out[valid] = np.asarray(labels, dtype=object)[idx]
    return out


def add_score_features(df: pd.DataFrame, a) -> pd.DataFrame:
    s = df["score"].astype(float).to_numpy()
    df["f_target_eval"] = cut(np.abs(s), [25, 75, 150, 300, 600, 1200, 9999],
                              ["|s|<=25", "26-75", "76-150", "151-300", "301-600", "601-1200", "1201-9999",
                               ">=10000 (won/TB)"])
    df["f_target_sign"] = np.where(s > 75, "stm_better", np.where(s < -75, "stm_worse", "balanced"))
    df["f_score_exact_zero"] = np.where(s == 0, "yes", "no")

    nxt = df["next_score"].astype(float).to_numpy()
    prv = df["prev_score"].astype(float).to_numpy()
    # compare in expected-score space so that +3000 -> +3500 does not look unstable
    pn = wdl(np.nan_to_num(nxt), a.out_offset, a.out_scaling)
    pp = wdl(np.nan_to_num(prv), a.out_offset, a.out_scaling)
    p0 = wdl(s, a.out_offset, a.out_scaling)
    dn = np.where(np.isnan(nxt), np.nan, np.abs(pn - p0))
    dp = np.where(np.isnan(prv), np.nan, np.abs(pp - p0))
    df["f_score_change_next_ply"] = cut(dn, [0.02, 0.05, 0.1, 0.2],
                                        ["<=0.02", "0.02-0.05", "0.05-0.10", "0.10-0.20", ">0.20"])
    df["f_score_change_prev_ply"] = cut(dp, [0.02, 0.05, 0.1, 0.2],
                                        ["<=0.02", "0.02-0.05", "0.05-0.10", "0.10-0.20", ">0.20"])

    md = df["material_diff"].to_numpy()
    agree = np.full(len(df), "consistent", dtype=object)
    agree[(md >= 2) & (s <= 50)] = "material_up_eval_not_better"
    agree[(md <= -2) & (s >= -50)] = "material_down_eval_not_worse"
    agree[(np.abs(md) <= 1) & (np.abs(s) >= 400) & (np.abs(s) < 10000)] = "big_eval_without_material"
    df["f_material_vs_eval"] = agree

    r = df["result"].to_numpy()
    p = wdl(s, a.out_offset, a.out_scaling)
    res = np.full(len(df), "no_result", dtype=object)
    known = np.isin(r, (-1, 0, 1))
    pred = np.where(p > 0.75, 1, np.where(p < 0.25, -1, 0))
    res[known & (pred == r)] = "agrees"
    res[known & (pred != r) & (np.abs(pred - r) == 1)] = "off_by_one"
    res[known & (np.abs(pred - r) == 2)] = "opposite"
    df["f_score_vs_result"] = res

    df["f_ply"] = cut(df["ply"].astype(float).to_numpy(), [16, 40, 80, 120, 200],
                      ["<=16", "17-40", "41-80", "81-120", "121-200", ">200"])
    if "ls_index" in df:
        df["f_layer_stack_remapped"] = np.where(df["ls_index"] != df["ls_used"], "yes (fallback stack)", "no")
        df["f_layer_stack"] = df["ls_used"].astype(str)
    names = getattr(a, "source_names", {}) or {}
    df["f_source"] = [f"{i}: {names[i]}" if i in names else str(i) for i in df["source"]]
    df["f_position_type"] = position_type(df)
    return df


POSITION_TYPES = [
    ("tactical: in check", "side to move is in check"),
    ("tactical: material in flight", ("the stored move captures/promotes, or one side can win >= a minor "
                                      "piece with a single capture (pending recapture, hanging piece)")),
    ("decided (|score| >= 1200)", "quiet, but the search score is already decisive"),
    ("TB endgame (<= 7 pieces)", ""),
    ("pawn endgame", "only kings and pawns, 8+ pieces"),
    ("endgame (npm <= 13)", ""),
    ("quiet material imbalance", "pieces (not just pawns) are unbalanced, nothing en prise"),
    ("quiet opening (npm > 50)", ""),
    ("quiet middlegame", "everything else"),
]


def position_type(df: pd.DataFrame) -> np.ndarray:
    """One primary class per position; the first matching rule of POSITION_TYPES wins."""
    s = df["score"].abs().to_numpy()
    mt = df["f_move_type"].to_numpy()
    material_in_flight = (np.isin(mt, ["capture", "capture+check", "promotion"])
                          | (df["gain_stm"].to_numpy() >= 3) | (df["gain_opp"].to_numpy() >= 3))
    balanced_pieces = df["f_imbalance"].str.startswith(("equal", "pawns_only")).to_numpy()
    rules = [
        df["f_in_check"].to_numpy() == "yes",
        material_in_flight,
        s >= 1200,
        df["n_pieces"].to_numpy() <= 7,
        df["npm_total"].to_numpy() == 0,
        df["npm_total"].to_numpy() <= 13,
        ~balanced_pieces,
        df["npm_total"].to_numpy() > 50,
    ]
    names = [n for n, _ in POSITION_TYPES]
    return np.select(rules, names[:-1], default=names[-1])


def flag_placeholder_scores(df: pd.DataFrame, report_lines: list) -> np.ndarray:
    """Positions whose score is a placeholder rather than a search result.

    * Some binpacks (e.g. minimized .min ones) store score 0 for every capture / in-check position. This is
      detected per source: if >= 90% of these positions have score 0 they are all flagged.
    * The trainer's own heuristic: score 0 in a decisive game right after a position scored |s| > 100.
    """
    s = df["score"].to_numpy()
    nontrivial = (df["capture"] == 1) | (df["in_check"] == 1)
    bad = np.zeros(len(df), dtype=bool)
    for src, g in df[nontrivial].groupby("source"):
        frac = float((g["score"] == 0).mean()) if len(g) else 0.0
        if len(g) >= 50 and frac >= 0.9:
            m = (df["source"] == src).to_numpy() & nontrivial.to_numpy()
            bad |= m
            report_lines.append(f"- source {src}: {frac:.0%} of capture/in-check positions have score 0 -> "
                                f"treated as placeholders ({int(m.sum())} positions excluded)")
    prv = df["prev_score"].to_numpy(dtype=float)
    heur = (s == 0) & (df["result"].to_numpy() != 0) & (np.abs(np.nan_to_num(prv)) > 100)
    if heur.any():
        report_lines.append(f"- trainer placeholder-zero heuristic: {int((heur & ~bad).sum())} more excluded")
    return bad | heur


# ---------------------------------------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------------------------------------

LABELS = ("learned", "struggles", "fails")


def summarise(g: pd.DataFrame, total_loss: float) -> dict:
    n = len(g)
    lab = g["label"].value_counts()
    return {
        "n": n,
        "learned": lab.get("learned", 0) / n,
        "struggles": lab.get("struggles", 0) / n,
        "fails": lab.get("fails", 0) / n,
        "mean_abs_err": g["abs_err"].mean(),
        "bias": g["err"].mean(),
        "loss_share": g["loss"].sum() / total_loss if total_loss > 0 else 0.0,
    }


def feature_table(df, feat, total_loss, overall_fail, min_n):
    rows = []
    for val, g in df.groupby(feat, sort=False):
        r = summarise(g, total_loss)
        r["value"] = val
        r["share"] = len(g) / len(df)
        r["fail_lift"] = r["fails"] / overall_fail if overall_fail > 0 else float("nan")
        rows.append(r)
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    t["small"] = t["n"] < min_n
    return t.sort_values("fail_lift", ascending=False)


def fmt_pct(x):
    return f"{100 * x:5.1f}%"


def md_table(t: pd.DataFrame, name_col="value", extra_cols=()):
    cols = ["n", "share", "learned", "struggles", "fails", "fail_lift", "mean_abs_err", "bias", "loss_share"]
    head = "| " + " | ".join([name_col, *extra_cols, *cols]) + " |"
    sep = "|" + "---|" * (1 + len(extra_cols) + len(cols))
    lines = [head, sep]
    for _, r in t.iterrows():
        cells = [str(r[name_col]) + (" *" if r.get("small", False) else "")]
        cells += [str(r[c]) for c in extra_cols]
        cells += [
            f"{int(r['n'])}", fmt_pct(r["share"]), fmt_pct(r["learned"]), fmt_pct(r["struggles"]),
            fmt_pct(r["fails"]), f"{r['fail_lift']:.2f}", f"{r['mean_abs_err']:.4f}", f"{r['bias']:+.4f}",
            fmt_pct(r["loss_share"]),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


FEATURE_GROUPS = [
    ("Primary position type", ["f_position_type"]),
    ("Material and phase", ["f_piece_count", "f_phase", "f_material_diff", "f_imbalance", "f_queens",
                            "f_bishops"]),
    ("Pawns", ["f_pawn_wings", "f_passed_pawns", "f_passer_stm_most_advanced", "f_passer_opp_most_advanced",
               "f_pawn_on_7th"]),
    ("Tactics and threats", ["f_in_check", "f_stm_can_win", "f_stm_threatened", "f_stm_en_prise",
                             "f_opp_en_prise", "f_pins", "f_captures_available", "f_checks_available",
                             "f_mobility", "f_move_type"]),
    ("King safety", ["f_stm_king_attacks", "f_opp_king_attacks", "f_stm_king_shelter", "f_stm_king_rank",
                     "f_castling_rights"]),
    ("Target and game context", ["f_target_eval", "f_target_sign", "f_score_exact_zero",
                                 "f_score_change_next_ply", "f_score_change_prev_ply", "f_material_vs_eval",
                                 "f_score_vs_result", "f_rule50", "f_ply"]),
    ("Net / data specifics", ["f_layer_stack_remapped", "f_source"]),
]

FEATURE_HELP = {
    "f_position_type": "one class per position, first matching rule: "
                       + "; ".join(f"{n}" + (f" ({h})" if h else "") for n, h in POSITION_TYPES),
    "f_piece_count": "pieces on the board incl. kings and pawns",
    "f_phase": "non-pawn material of both sides (N=B=3, R=5, Q=9)",
    "f_material_diff": "material of side to move minus opponent (P=1, N=B=3, R=5, Q=9)",
    "f_imbalance": "kind of material imbalance (side to move vs opponent)",
    "f_stm_can_win": "material the side to move can win with one capture (crude SEE)",
    "f_stm_threatened": "material the opponent threatens to win with one capture (crude SEE)",
    "f_stm_en_prise": "side-to-move pieces attacked and undefended or attacked by a cheaper piece",
    "f_opp_en_prise": "opponent pieces attacked and undefended or attacked by a cheaper piece",
    "f_move_type": "the move stored in the binpack (normally the search best move)",
    "f_score_change_next_ply": "|expected score change| of the search score after the stored move "
                               "(large = tactical / unstable / search noise)",
    "f_score_change_prev_ply": "|expected score change| between the previous position and this one",
    "f_material_vs_eval": "does the search score follow the material balance?",
    "f_score_vs_result": "does the search score (W/D/L at expected score 0.25/0.75) match the game result?",
    "f_rule50": "halfmove clock (the net does not see it)",
    "f_layer_stack_remapped": "the position's layer stack is rare and was routed to a fallback stack",
    "f_stm_king_attacks": "opponent (attacker, square) pairs on the side-to-move king zone",
    "f_opp_king_attacks": "side-to-move (attacker, square) pairs on the opponent king zone",
}


def write_report(df, a, notes, out_dir, t_start):
    total_loss = df["loss"].sum()
    overall = summarise(df, total_loss)
    of = overall["fails"]
    min_n = a.min_count
    L = []
    L.append("# Position classification report\n")
    L.append(f"- input: `{a.input}`")
    L.append(f"- positions analysed: **{len(df)}**")
    L.extend(notes)
    L.append(f"- labels by |err| in expected-score space: learned < {a.learned}, struggles < {a.fails}, "
             f"fails >= {a.fails}")
    L.append(f"- err = q_net - p_target, q_net = wdl(net, {a.in_offset}, {a.in_scaling}), "
             f"p_target = wdl(tb_remap(score), {a.out_offset}, {a.out_scaling}); bias > 0 = net too optimistic "
             "for the side to move")
    L.append("- `fail_lift` = fail rate of the class / overall fail rate; `loss_share` = share of the total "
             f"|err|^{a.pow_exp} (the training loss) the class accounts for; rows marked * have fewer than "
             f"{min_n} positions\n")

    L.append("## Overall\n")
    L.append("| learned | struggles | fails | mean abs err | bias | corr(net, score) cp, |score|<2000 | "
             "median abs cp err, |score|<2000 | mean abs err vs game result |")
    L.append("|---|---|---|---|---|---|---|---|")
    m = df["score"].abs() < 2000
    corr = np.corrcoef(df.loc[m, "score"], df.loc[m, "net_eval"])[0, 1] if m.sum() > 2 else float("nan")
    mae_cp = (df.loc[m, "score"] - df.loc[m, "net_eval"]).abs().median()
    known = df["result"].isin([-1, 0, 1])
    res_err = (df.loc[known, "q_net"] - (df.loc[known, "result"] + 1) / 2).abs().mean()
    tgt_res_err = (df.loc[known, "p_target"] - (df.loc[known, "result"] + 1) / 2).abs().mean()
    L.append(f"| {fmt_pct(overall['learned'])} | {fmt_pct(overall['struggles'])} | {fmt_pct(of)} | "
             f"{overall['mean_abs_err']:.4f} | {overall['bias']:+.4f} | {corr:.3f} | {mae_cp:.0f} | "
             f"{res_err:.4f} (search score itself: {tgt_res_err:.4f}) |\n")

    q = np.quantile(df["abs_err"], [0.5, 0.75, 0.9, 0.95, 0.99])
    L.append("|err| quantiles: " + ", ".join(f"p{int(p * 100)}={v:.3f}" for p, v in
                                              zip([0.5, 0.75, 0.9, 0.95, 0.99], q)) + "\n")

    L.append("## By primary position type\n")
    L.append(FEATURE_HELP["f_position_type"] + "\n")
    pt = feature_table(df, "f_position_type", total_loss, of, min_n)
    L.append(md_table(pt))
    L.append("")

    # --- ranking over all classes ---------------------------------------------------------------------
    feats = [f for _, fs in FEATURE_GROUPS for f in fs if f in df.columns]
    if df["f_endgame"].ne("-").any():
        feats_rank = feats + ["f_endgame"]
    else:
        feats_rank = feats
    all_rows = []
    tables = {}
    for f in feats_rank:
        t = feature_table(df, f, total_loss, of, min_n)
        tables[f] = t
        t2 = t.copy()
        t2["feature"] = f[2:]
        all_rows.append(t2)
    allt = pd.concat(all_rows, ignore_index=True)
    big = allt[~allt["small"]]

    L.append("## Where the net fails (classes with the highest fail rate)\n")
    L.append(md_table(big.sort_values("fail_lift", ascending=False).head(a.top), extra_cols=("feature",)))
    L.append("\n## Where the net struggles (highest struggle + fail rate)\n")
    tmp = big.assign(sf=big["struggles"] + big["fails"]).sort_values("sf", ascending=False)
    L.append(md_table(tmp.head(a.top), extra_cols=("feature",)))
    L.append("\n## What the net has learned (classes with the highest learned rate)\n")
    L.append(md_table(big.sort_values("learned", ascending=False).head(a.top), extra_cols=("feature",)))
    L.append("\n## Where the loss comes from (largest share of the total loss)\n")
    L.append(md_table(big.sort_values("loss_share", ascending=False).head(a.top), extra_cols=("feature",)))
    L.append("\n## Most biased classes (|mean signed error|)\n")
    L.append(md_table(big.reindex(big["bias"].abs().sort_values(ascending=False).index).head(a.top),
                      extra_cols=("feature",)))

    # --- two-way interactions -------------------------------------------------------------------------
    L.append("\n## Hardest two-feature combinations\n")
    L.append(f"Cells of two features with at least {min_n} positions, ranked by fail lift; `vs parts` is the "
             "cell's fail rate divided by the larger fail rate of its two single-feature classes (> 1: the "
             "combination is harder than either part alone).\n")
    single_fail = {(f, v): r["fails"] for f, t in tables.items() for _, r in t.iterrows()
                   for v in [r["value"]]}
    combos = []
    pair_feats = [f for f in feats if f not in ("f_source", "f_position_type")]
    for f1, f2 in itertools.combinations(pair_feats, 2):
        grp = df.groupby([f1, f2], sort=False)
        stats = grp.agg(n=("label", "size"), fails=("is_fail", "mean"), mae=("abs_err", "mean"),
                        bias=("err", "mean"), loss=("loss", "sum"))
        stats = stats[stats["n"] >= min_n]
        for (v1, v2), r in stats.iterrows():
            parts = max(single_fail.get((f1, v1), 0), single_fail.get((f2, v2), 0))
            combos.append((f"{f1[2:]}={v1} & {f2[2:]}={v2}", int(r["n"]), r["fails"], r["fails"] / of if of else 0,
                           r["fails"] / parts if parts > 0 else float("nan"), r["mae"], r["bias"],
                           r["loss"] / total_loss))
    ct = pd.DataFrame(combos, columns=["cell", "n", "fails", "fail_lift", "vs_parts", "mae", "bias",
                                       "loss_share"])
    ct = ct[ct["vs_parts"] > 1.15].sort_values("fail_lift", ascending=False)
    # keep the table varied: every single class may appear in at most 3 cells
    used: dict = {}
    keep = []
    for i, cell in ct["cell"].items():
        parts = cell.split(" & ")
        if all(used.get(p, 0) < 3 for p in parts):
            keep.append(i)
            for p in parts:
                used[p] = used.get(p, 0) + 1
        if len(keep) >= a.top:
            break
    ct = ct.loc[keep]
    L.append("| cell | n | fails | fail_lift | vs parts | mean abs err | bias | loss_share |")
    L.append("|---|---|---|---|---|---|---|---|")
    for _, r in ct.iterrows():
        L.append(f"| {r['cell']} | {r['n']} | {fmt_pct(r['fails'])} | {r['fail_lift']:.2f} | "
                 f"{r['vs_parts']:.2f} | {r['mae']:.4f} | {r['bias']:+.4f} | {fmt_pct(r['loss_share'])} |")

    # --- per feature ----------------------------------------------------------------------------------
    L.append("\n## All classes, per feature\n")
    for gname, fs in FEATURE_GROUPS:
        L.append(f"### {gname}\n")
        for f in fs:
            if f not in tables:
                continue
            L.append(f"#### {f[2:]}" + (f" — {FEATURE_HELP[f]}" if f in FEATURE_HELP else "") + "\n")
            L.append(md_table(tables[f]))
            L.append("")
    if "f_endgame" in tables:
        t = tables["f_endgame"]
        t = t[t["value"] != "-"].sort_values("n", ascending=False).head(40).sort_values("fail_lift",
                                                                                         ascending=False)
        L.append("### Endgames (<= 8 pieces), 40 most frequent material signatures\n")
        L.append(md_table(t))
        L.append("")

    L.append(f"\n_generated in {time.time() - t_start:.0f}s_\n")
    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write("\n".join(L))
    return allt, big


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="TSV from eval_positions.py")
    ap.add_argument("--out-dir", default="position_report")
    ap.add_argument("--learned", type=float, default=0.05, help="|err| below this: learned")
    ap.add_argument("--fails", type=float, default=0.15, help="|err| at or above this: fails")
    ap.add_argument("--quiet-only", action="store_true",
                    help="only positions the trainer uses with --filtered (no capture move, not in check)")
    ap.add_argument("--keep-placeholders", action="store_true",
                    help="do not drop positions whose score looks like a placeholder 0")
    ap.add_argument("--min-count", type=int, default=200, help="minimum class size for the rankings")
    ap.add_argument("--top", type=int, default=25, help="rows in the ranking tables")
    ap.add_argument("--worst", type=int, default=30, help="worst positions per hard class in worst_positions.tsv")
    ap.add_argument("--sources", default=None,
                    help="TSV 'index<TAB>binpack path' naming the sources (default: sources.tsv next to the "
                         "input, as written by run_all.sh)")
    ap.add_argument("-j", "--jobs", type=int, default=os.cpu_count())
    # conversion used by the training loss (model/config.py LossParams)
    ap.add_argument("--in-offset", type=float, default=270.0)
    ap.add_argument("--in-scaling", type=float, default=340.0)
    ap.add_argument("--out-offset", type=float, default=270.0)
    ap.add_argument("--out-scaling", type=float, default=380.0)
    ap.add_argument("--tb-remap-base", type=float, default=4000.0)
    ap.add_argument("--tb-remap-scale", type=float, default=4000.0)
    ap.add_argument("--tb-remap-decay", type=float, default=0.85)
    ap.add_argument("--pow-exp", type=float, default=2.5)
    a = ap.parse_args()

    t_start = time.time()
    os.makedirs(a.out_dir, exist_ok=True)
    src_file = a.sources or os.path.join(os.path.dirname(os.path.abspath(a.input)), "sources.tsv")
    a.source_names = {}
    if os.path.exists(src_file):
        with open(src_file) as f:
            for line in f:
                if "\t" in line:
                    i, path = line.rstrip("\n").split("\t", 1)
                    a.source_names[int(i)] = os.path.basename(path)
    df = pd.read_csv(a.input, sep="\t", keep_default_na=False,
                     na_values={"next_score": [""], "prev_score": [""]})
    if "net_eval" not in df:
        sys.exit("input has no net_eval column; run eval_positions.py first")
    notes = []
    n0 = len(df)
    if not a.keep_placeholders:
        bad = flag_placeholder_scores(df, notes)
        df = df[~bad].reset_index(drop=True)
    if a.quiet_only:
        df = df[(df["capture"] == 0) & (df["in_check"] == 0)].reset_index(drop=True)
        notes.append("- restricted to quiet positions (no capture move, not in check)")
    notes.insert(0, f"- read {n0} positions, {n0 - len(df)} excluded")
    if not a.quiet_only:
        notes.append("- includes capture-move and in-check positions; training with --filtered never sees these, "
                     "so failures there are out of distribution (rerun with --quiet-only for the training "
                     "distribution)")

    print(f"computing board features for {len(df)} positions with {a.jobs} processes...", file=sys.stderr)
    rows = list(zip(df["fen"], df["move"]))
    chunk = 2000
    parts = [rows[i:i + chunk] for i in range(0, len(rows), chunk)]
    feats = []
    with Pool(a.jobs) as pool:
        for i, r in enumerate(pool.imap(_features_chunk, parts)):
            feats.extend(r)
            if sys.stderr.isatty() or (i + 1) % 100 == 0 or (i + 1) == len(parts):
                print(f"\r  {min((i + 1) * chunk, len(rows))}/{len(rows)}", end="", file=sys.stderr)
    print(file=sys.stderr)
    df = pd.concat([df, pd.DataFrame(feats)], axis=1)

    score = df["score"].astype(float).to_numpy()
    net = df["net_eval"].astype(float).to_numpy()
    df["q_net"] = wdl(net, a.in_offset, a.in_scaling)
    df["p_target"] = wdl(tb_remap(score, a.tb_remap_base, a.tb_remap_scale, a.tb_remap_decay),
                         a.out_offset, a.out_scaling)
    df["err"] = df["q_net"] - df["p_target"]
    df["abs_err"] = df["err"].abs()
    df["loss"] = df["abs_err"] ** a.pow_exp
    df["label"] = np.where(df["abs_err"] < a.learned, "learned",
                           np.where(df["abs_err"] < a.fails, "struggles", "fails"))
    df["is_fail"] = (df["label"] == "fails").astype(float)
    df = add_score_features(df, a)

    print("writing report...", file=sys.stderr)
    _, big = write_report(df, a, notes, a.out_dir, t_start)

    df.drop(columns=["is_fail"]).to_csv(os.path.join(a.out_dir, "positions_classified.tsv"), sep="\t",
                                        index=False, float_format="%.4f")

    # worst positions: overall and for the hardest classes
    cols = ["fen", "move", "score", "net_eval", "p_target", "q_net", "err", "ply", "result", "next_score"]
    out = [df.nlargest(a.worst * 3, "abs_err")[cols].assign(cls="overall")]
    for _, r in big.sort_values("fail_lift", ascending=False).head(15).iterrows():
        f = "f_" + r["feature"]
        g = df[df[f] == r["value"]]
        out.append(g.nlargest(a.worst, "abs_err")[cols].assign(cls=f"{r['feature']}={r['value']}"))
    pd.concat(out).to_csv(os.path.join(a.out_dir, "worst_positions.tsv"), sep="\t", index=False,
                          float_format="%.4f")
    print(f"done: {a.out_dir}/report.md, positions_classified.tsv, worst_positions.tsv "
          f"({time.time() - t_start:.0f}s)", file=sys.stderr)


if __name__ == "__main__":
    main()
