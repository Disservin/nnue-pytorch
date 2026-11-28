# fsqd-filter

Reads binpacks, applies the training data loader's skip filter (the real C++
`make_skip_predicate` from `data_loader/cpp`, compiled into the tool) with
random FEN skipping off, and writes the accepted positions of each input
`X.binpack` to `X.fsqd` in an output directory, using the `four-square`
library.

```bash
cargo build --release
./target/release/fsqd-filter <files-or-dirs>... -o <output-dir> [filter options]
```

- Directories are searched recursively for `*.binpack`; their subdirectory
  layout is mirrored under the output directory. A file given directly goes to
  the top of the output directory.
- `--threads N` files are converted at once (default 8), each into its own
  output, written as `X.fsqd.tmp` and renamed to `X.fsqd` when complete.
- Outputs that already exist are skipped, so an interrupted run continues with
  the same command. Leftover `.tmp` files are overwritten. `--force` redoes
  everything and retrains the model.
- `--list` prints the input → output mapping and exits. `--exclude TEXT`
  drops files whose path contains `TEXT`.
- Filter options have the same names and defaults as `train.py`.

## Model

All outputs share one model. The first run samples `--train-records`
accepted positions (default 4M) spread evenly over all inputs, trains the
model and saves it as `fsqd-model.bin` in the output directory. Later runs
load that file, so files converted at different times still share a
byte-identical model. Concatenated files then share one decoded model in the
reader.

## Joining the outputs

FSQD files concatenate into one valid multi-segment file:

```bash
find <output-dir> -name '*.fsqd' | sort | xargs cat > all.fsqd
```

## Behaviour compared with training

Each worker reads whole files in order, so the filter's stateful parts
(placeholder-zero detection, the adaptive piece-count filter) see positions in
file order, as in training. Their state carries over between the files a
worker reads. The training loader mixes chunks of all files in each thread, so
the piece-count filter adapts to a slightly different mix here. Outputs are
not shuffled.

Search-tree samples (ply 9999) have a placeholder result. They are written
with outcome `NoResult`, which the training loader maps back to the trainer's
"no result" outcome (2.0), so they keep training on the score alone.

## Training on the outputs

`train.py` reads `.fsqd` files like binpacks, and the two can be mixed in
one run. The loader applies only `--random-fen-skipping` to FSQD positions,
because this tool already applied every other filter. Train with the recipe's
`--random-fen-skipping 3` to keep the same effective skip rate. The other
filter options then only affect binpack inputs. FSQD stores no ply or move, so
the ply-based filters could not run on it anyway.

## The current recipe

The recipe's 59 distinct binpacks are everything under `nettest/data` except
the `linrock/` and `official-stockfish/` directories. (The recipe lists
`test80-2023-12-dec` twice; this tool reads each file once.)

```bash
./target/release/fsqd-filter /media/max/DEV3/stockfish-data/nettest/data \
  --exclude /linrock/ --exclude /official-stockfish/ \
  -o /media/max/DEV3/recipe-fsqd \
  --early-fen-skipping 18 --soft-early-fen-skipping 32 \
  --pc-y0 -0.20 --pc-y1 0.45 --pc-y2 1.0 --pc-y3 0.95 --pc-y4 0.75 \
  --ply-x1 0.00 --ply-y1 0.025 --ply-x2 22.0 --ply-y2 0.05 \
  --ply-x3 25.5 --ply-y3 0.20 --ply-x4 29.5 --ply-y4 0.80 \
  --tree-sample-accept 0.12 --decided-score 1000 --decided-accept 0.3
```
