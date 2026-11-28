# Four-square file dictionary for independent training positions

This implementation proposal adds a Rust library module named `sfbinpack::four_square`. Each position describes its own board. Shuffling records before packing costs no move-history information. A block of up to 256 positions shares an entropy-coder state and can be read independently. The model is shared across blocks.

| Read this document | Implementation question |
|---|---|
| [Record codec](codec.md) | Which symbols describe a board, and how are the dictionaries and rANS tables built? |
| [File format](container.md) | How can two implementations exchange models, locate blocks, and validate a file? |
| [Library and workflows](library.md) | How should a loader, converter, or parallel relabeler use the Rust API? |

### Evidence and scope

The C++ prototype encoded 100,000 shuffled held-out positions from a filtered June 2023 Stockfish sample. A separate 100,000 positions trained the model. Four-square groups along files measured **10.757 bytes per record and 3.32 million complete records/s** on one Ryzen 9 5950X thread. The same group shape along ranks measured 11.054 bytes and 3.39 million/s in the earlier run. The size includes a four-byte state and eight-byte index allowance per block. It excludes the one-time model. The timing covers in-memory board and metadata reconstruction. The proposed Rust implementation, disk throughput, feature extraction, and multicore scaling still need measurement.

The record preserves all 64 squares, side to move, orthodox castling rights, the en passant square, the exact signed 16-bit score, and the side-to-move game result, or no result for search-tree samples. Projecting a binpack entry discards its move, ply and clocks. Apply filters that need those fields before projection. Preserve duplicates and input order by default. A relabeler that needs the fifty-move clock requires a separate source for that information.

### Implementation order

1. Implement the record type, deterministic model builder and scalar block codec. Verify symbol traces and cross-decoding before optimizing.
2. Add the container, bounded reads and indexed access. The header, model serialization and footer here are proposed for the Rust format. The prototype measured block payloads.
3. Add binpack projection, a reusable batch decoder, and ordered parallel writing. Benchmark separately at the block, storage and loader boundaries.

The existing binpack reader remains the conversion entry point. The crate already supports file and in-memory workflows. [Published sfbinpack crate](https://docs.rs/crate/sfbinpack/latest)

The nnue-pytorch training loader reads FSQD files directly through a C++
decoder, `data_loader/cpp/lib/fsqd.h`. `tools/fsqd-filter` writes them from
binpacks.
