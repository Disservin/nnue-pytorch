use std::fmt;
use std::io;

/// Errors reported by the FSQD reader, writer and model builder.
#[derive(Debug)]
pub enum Error {
    /// An I/O operation on the source or sink failed.
    Io(io::Error),
    /// The serialized model or a model built from counts is invalid.
    InvalidModel(String),
    /// A record or a binpack entry cannot be represented.
    InvalidRecord(String),
    /// The header, footer or index is malformed.
    InvalidFile(String),
    /// Data ended early. `block` and `record` locate the failure when known.
    Truncated { block: Option<u64>, record: Option<u32> },
    /// A block payload decoded to something the format forbids.
    Corrupt { block: u64, record: Option<u32>, reason: &'static str },
    /// A block or record ID outside the file.
    OutOfRange { id: u64, len: u64 },
    /// An earlier write failed, so this writer accepts no further calls.
    WriterPoisoned,
}

pub type Result<T> = std::result::Result<T, Error>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "I/O error: {e}"),
            Error::InvalidModel(m) => write!(f, "invalid model: {m}"),
            Error::InvalidRecord(m) => write!(f, "invalid record: {m}"),
            Error::InvalidFile(m) => write!(f, "invalid file: {m}"),
            Error::Truncated { block, record } => {
                write!(f, "truncated data")?;
                if let Some(b) = block {
                    write!(f, " in block {b}")?;
                }
                if let Some(r) = record {
                    write!(f, " at record {r}")?;
                }
                Ok(())
            }
            Error::Corrupt { block, record, reason } => {
                write!(f, "corrupt block {block}")?;
                if let Some(r) = record {
                    write!(f, " at record {r}")?;
                }
                write!(f, ": {reason}")
            }
            Error::OutOfRange { id, len } => write!(f, "id {id} out of range (len {len})"),
            Error::WriterPoisoned => write!(f, "writer is poisoned by an earlier error"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Error::Io(e)
    }
}
