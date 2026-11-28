//! FSQD: an indexed, entropy-coded container for independent chess training
//! positions ("four-square file dictionary").
//!
//! Every record describes its own board, so records can be shuffled before
//! packing. Up to 256 records share one rANS state in a block, every block is
//! independently decodable, and one immutable [`Model`] is shared by all
//! blocks of a segment. A file is one or more segments, so FSQD files can be
//! joined with `cat` and read as one [`Dataset`]. See `codec.md` and
//! `container.md` for the format.
//!
//! ```no_run
//! use std::sync::Arc;
//! use four_square::{Dataset, Model, Record, Writer};
//!
//! # fn main() -> four_square::Result<()> {
//! # let records: Vec<Record> = Vec::new();
//! let model = Arc::new(Model::train(|| Ok(records.iter().copied().map(Ok)))?);
//! let mut w = Writer::new(std::fs::File::create("out.fsqd")?, model)?;
//! for r in &records {
//!     w.push(r)?;
//! }
//! w.finish()?;
//!
//! let ds = Dataset::open(std::fs::File::open("out.fsqd")?)?;
//! let mut worker = ds.worker();
//! for id in 0..ds.blocks() {
//!     for r in ds.decode_block(id, &mut worker)? {
//!         let _ = (r.board.squares(), r.labels.score);
//!     }
//! }
//! # Ok(())
//! # }
//! ```

mod codec;
mod container;
mod error;
mod model;
mod rans;
mod record;

pub use container::{BlockIter, Dataset, IndexValidation, OpenOptions, ReadAt, Segment, Worker, Writer, BLOCK_RECORDS};
pub use error::{Error, Result};
pub use model::Model;
pub use record::{castling, piece, BoardState, Labels, Outcome, Record, TREE_SAMPLE_PLY};

#[cfg(test)]
mod tests;
