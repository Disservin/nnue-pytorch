# Position classification: what a net learns, where it struggles, where it fails

Three standalone programs and a wrapper script:

| step | program | needs |
|---|---|---|
| 1. sample positions from binpacks | `extract_positions` (C++, `build.sh`) | C++20 compiler, the headers in `../data_loader/cpp/lib` |
| 2. evaluate them with a net | `eval_positions.py` | this repo (model code + compiled data loader), torch; GPU optional |
| 3. classify + report | `classify_positions.py` | numpy, pandas, python-chess only |
| all of it | `run_all.sh NET OUT_DIR file.binpack...` | |

Steps 1 and 3 do not depend on the training code, so you can run step 3 on any machine once you have the
`positions.eval.tsv` file.

```bash
position_analysis/build.sh
COUNT=1000000 position_analysis/run_all.sh nn-bddd9d57eb87.nnue position_analysis/results_bddd9d57 /data/**/*.binpack
# in-distribution view only (what the trainer with --filtered actually sees):
python position_analysis/classify_positions.py --quiet-only --sources results/sources.tsv \
    --out-dir results/quiet_only results/positions.eval.tsv
```

On this machine (32 cores, RTX 4090), 1M positions take about 3 minutes in total: extraction takes seconds,
evaluation runs at about 300k positions/s, and the board features take about 90 s.

**Data loader library:** `eval_positions.py` uses the repo's compiled `libtraining_data_loader`. The one in
`build/` was older than the current sources (it didn't know the `K32H3` feature). A plain rebuild is
enough:
`cmake -S data_loader/cpp -B data_loader/cpp/build -DCMAKE_BUILD_TYPE=Release && cmake --build data_loader/cpp/build -j --target training_data_loader`.
The loader picks the newest library it finds.

## How it works

**Sampling.** Binpack chunks can be decoded on their own. The extractor indexes every chunk of every file,
visits chunks in random order (by default the same number of chunks from each file in turn; `--by-size`
weights by file size instead) and takes a random `--per-chunk` subset from each chunk. For each position it
also records the search score of the previous and the next ply of the same game, and whether the stored move
is a capture or the side to move is in check.

**Placeholder scores.** Minimized binpacks (`*.min*`) store score 0 for every capture or in-check
position. The classifier detects this per source and drops those positions. It also applies the trainer's
own placeholder-zero heuristic.

**Error.** The net eval and the search score go through the same conversion the training loss uses
(`wdl(net, 270, 340)` and `wdl(tb_remap(score), 270, 380)`, which you can change with options). The error is
`err = q_net - p_target` in expected-score space. A positive error means the net is too optimistic for the
side to move. Labels:

- **learned**: `|err| < 0.05`
- **struggles**: `0.05 <= |err| < 0.15`
- **fails**: `|err| >= 0.15`

You can change the thresholds with `--learned` and `--fails`.

**Position classes.** Each position gets:

- one *primary type*, from the first rule that matches: in check / material in flight / decided / TB
  endgame / pawn endgame / endgame / quiet imbalance / quiet opening / quiet middlegame;
- about 40 categorical features: material and kind of imbalance, phase, endgame signature, pawn
  structure, passed pawns, crude-SEE threats (what the side to move can win and what it must defend),
  pieces en prise, pins, mobility, available checks and captures, king attackers and shelter, stored move
  type, rule-50 counter, score stability over the neighbouring plies, whether the score agrees with
  material and with the game result, layer-stack fallback, and source file.

For every class the report gives: learned / struggles / fails rates, `fail_lift` (the class's fail rate
divided by the overall fail rate), mean signed error (bias), and `loss_share` (the share of the total
`|err|^2.5` that the class accounts for). It ranks classes by fail rate, by struggle rate, by learned
rate, by loss share and by bias, and lists two-feature combinations that are harder than either feature on
its own.

## Outputs

- `report.md`: the report.
- `positions_classified.tsv`: every position with all features, the error and the label, for pandas
  follow-ups. Example:
  `df[df.f_endgame == "KBNvK"].groupby("f_source").err.describe()`
- `worst_positions.tsv`: the worst positions overall and for the hardest classes, with FENs.

## Per-source score calibration

`calibrate_sources.py` checks whether the binpacks put their scores on different scales. It fits the
loss's win/draw/loss curve (`out_offset`, `out_scaling`) to each source's own game results. It also fits a
factor `k` that puts each source on the pooled curve: multiplying the source's scores by `k` is the offline
rescale. It needs no net, only `positions.tsv`:

```bash
python position_analysis/calibrate_sources.py results/positions.tsv   # writes calibration.md / .tsv
```

Positions of one game share a result, so the intervals come from resampling whole binpack chunks, and `k` is
also fitted on two random halves of the chunks. A source is reported as `differs` only when the whole 95%
interval of `k` lies outside 1 ± `--tol` (default 0.05). `gap after` shows how much of the difference a
single factor can't fix, because the source's curve has a different shape. Defaults: quiet positions
only, 8+ pieces, |score| ≤ 2000.

## Results for nn-bddd9d57eb87 (1M positions, 7 binpacks in /data)

These come from `results_bddd9d57/report.md` (all positions) and `results_bddd9d57/quiet_only/report.md`
(training distribution, 790k positions).

Quiet positions only: 74% learned, 21% struggles, 5% fails. The correlation with the search score is 0.92
(for |score| < 2000), and the median error is 57 internal units.

**Learned well**
- Decided positions (|score| ≥ 1200): 92% learned.
- Clearly balanced positions (|score| ≤ 25): 92% learned.
- Most ≤7-piece endgames: 83% learned.
- Quiet openings: 76% learned, under 2% fail.
- Equal material: 78% learned.
- Big material imbalances where the score follows the material: 92 to 95% learned.

**Struggles** (mostly spread, not a systematic bias)
- Clear but not decisive advantages (score 300 to 1200): about 37 to 40% learned and about 50% struggles.
  The net's magnitude is imprecise exactly where the loss is steepest.
- Advantages without material (`big_eval_without_material`, i.e. positional or attacking compensation):
  42% learned.
- Quiet material imbalances: minor piece for pawns, two or more pawns up or down, rook vs two minors,
  exchange up or down.
- Middlegames with npm 31 to 50: 57% learned. This is the hardest phase.

**Fails**
- Positions where the side to move can win material with one capture: rook or more gives 47% fails, a
  queen gives 55% fails. Capture moves give 28% fails. In-check positions give 22% fails. All of them are
  biased: the net underestimates the side that can capture (bias −0.14 to −0.18) and is too
  optimistic for the side in check (+0.03). A static eval can't resolve these, and training with
  `--filtered` never shows them to the net. They are out of distribution, not a learning failure. Together,
  these two tactical types account for about 70% of the total loss on unfiltered data.
- Minor-piece endgames with tablebase scores: KBBvKN (63% fails), KNNvKP (43%), KQvKBB (31%), and KBNvK
  (27% fails plus 54% struggles). Most of these positions come from `tb5dtm.binpack`. That set scores
  positions as mate in N (DTM) even when the game result is a draw (long wins that run past the 50-move
  rule), so the labels contradict each other. The net settles around ±330 instead of committing either
  way. The combination `target_eval >= 10000 & score_vs_result=off_by_one` fails 87% of the time.
- Unstable targets: positions where the search score moves by more than 0.1 expected score in the
  neighbouring plies fail 3 to 4 times as often. Part of this is label noise, not net error.
- By dataset: `fishpack32` (13.5% fails) and `nodes5000pv2_UHO` (12.6%) are hardest. The TB set and the
  DFRC set (3 to 4%) are easiest.

Things that turned out *not* to matter: the rule-50 counter, and whether a position's layer stack is routed
to a fallback stack.

## Caveats

- These are heuristics. The SEE is crude (one capture, cheapest attacker), and "king attacks" counts
  attacked squares.
- A "fail" means disagreement with the search score at the depth the data was generated with. Use the
  score-stability features to separate net error from target noise.
- If you evaluate on the data the net was trained on, you are measuring fit (what it *could* learn). Use
  held-out binpacks to measure generalization.
