# fsqd-shuffle

Joins FSQD files into one FSQD file in random record order, in a single pass
(each input is read once and the output is written once).

```bash
cargo build --release
./target/release/fsqd-shuffle <files-or-dirs>... -o all.fsqd [--buffer-gb 12] [--seed 1]
```

## How it shuffles

1. Every input is cut into runs of `--run-blocks` consecutive blocks
   (default 16 blocks, 4096 records, about 45 KB).
2. The runs of all inputs are read in one random order, so any part of any
   file is equally likely to come early or late in the output.
3. The records pass through a shuffle buffer of `--buffer-gb` (decoded
   records are 70 bytes; 12 GB holds 171M). Each record replaces a random
   slot, and the evicted record is written. At the end the rest of the buffer
   is shuffled and written.

Records from one run end up spread over about the buffer size. The order is
not an exactly uniform permutation, because positions read together stay
within a few hundred million of each other. For training batches it is
indistinguishable from one. A larger buffer or smaller runs bring it closer.

## Behaviour

- Directories are searched recursively for `*.fsqd`. `--exclude TEXT` drops
  files whose path contains `TEXT`, and `--list` prints the inputs and exits.
- The output is one segment with the first input's model. Inputs with another
  model are re-encoded with it.
- The output is written as `all.fsqd.tmp` and renamed after its record count
  is checked. An interrupted run starts over.
- Ctrl-C stops the pipeline, deletes the `.tmp` file and exits with status
  130. A second Ctrl-C exits at once and may leave the `.tmp` file behind.
- The output depends only on the inputs, `--seed`, `--run-blocks` and
  `--buffer-gb`, not on the thread counts.
- Threads: `--readers` decode runs (default 8), the main thread shuffles,
  `--encoders` encode blocks (default 16), one thread writes.

## Speed

On a Ryzen 9 5950X the pipeline reaches about 23M records/s when the disk
keeps up (measured with the output in RAM). Writing to a QLC SATA SSD that
is also being read is much slower: about 5.5M records/s were measured with
input and output both on a Crucial BX500.
