//! FSQD container, version 1.
//!
//! A file is one or more segments back to back:
//!
//! ```text
//! | header (16) | model | block 0 | ... | block B-1 | index (8 B) | footer (24) |
//! ```
//!
//! Offsets inside a segment are relative to its first byte, so concatenating
//! files (`cat a.fsqd b.fsqd`) yields a valid multi-segment file.

use std::collections::HashMap;
use std::io::{self, Write};
use std::ops::Range;
use std::sync::Arc;

use crate::codec::{self, BlockScratch};
use crate::error::{Error, Result};
use crate::model::{self, Model};
use crate::record::Record;

/// Records per block; only the last block of a file may be shorter.
pub const BLOCK_RECORDS: usize = 256;

const MAGIC: &[u8; 4] = b"FSQD";
const FOOTER_MAGIC: &[u8; 4] = b"FEND";
const VERSION: u16 = 1;
const HEADER_LEN: u64 = 16;
const FOOTER_LEN: u64 = 24;

/// Upper bound on a block payload: the state plus at most two refill bytes for
/// each of at most 3 + 16 * 6 + 5 symbols per record.
fn max_block_len(records: u64) -> u64 {
    4 + 208 * records
}

/// Positional reads from a file or memory.
#[allow(clippy::len_without_is_empty)]
pub trait ReadAt: Send + Sync {
    fn len(&self) -> io::Result<u64>;
    fn read_exact_at(&self, offset: u64, dst: &mut [u8]) -> io::Result<()>;
}

fn read_slice(src: &[u8], offset: u64, dst: &mut [u8]) -> io::Result<()> {
    let start = usize::try_from(offset).map_err(|_| io::ErrorKind::UnexpectedEof)?;
    let end = start.checked_add(dst.len()).ok_or(io::ErrorKind::UnexpectedEof)?;
    let s = src.get(start..end).ok_or(io::ErrorKind::UnexpectedEof)?;
    dst.copy_from_slice(s);
    Ok(())
}

impl ReadAt for Vec<u8> {
    fn len(&self) -> io::Result<u64> {
        Ok(self.as_slice().len() as u64)
    }
    fn read_exact_at(&self, offset: u64, dst: &mut [u8]) -> io::Result<()> {
        read_slice(self, offset, dst)
    }
}

impl ReadAt for &[u8] {
    fn len(&self) -> io::Result<u64> {
        Ok((**self).len() as u64)
    }
    fn read_exact_at(&self, offset: u64, dst: &mut [u8]) -> io::Result<()> {
        read_slice(self, offset, dst)
    }
}

impl ReadAt for std::fs::File {
    fn len(&self) -> io::Result<u64> {
        Ok(self.metadata()?.len())
    }
    #[cfg(unix)]
    fn read_exact_at(&self, offset: u64, dst: &mut [u8]) -> io::Result<()> {
        std::os::unix::fs::FileExt::read_exact_at(self, dst, offset)
    }
    #[cfg(windows)]
    fn read_exact_at(&self, mut offset: u64, mut dst: &mut [u8]) -> io::Result<()> {
        use std::os::windows::fs::FileExt;
        while !dst.is_empty() {
            match self.seek_read(dst, offset) {
                Ok(0) => return Err(io::ErrorKind::UnexpectedEof.into()),
                Ok(n) => {
                    dst = &mut dst[n..];
                    offset += n as u64;
                }
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }
}

impl<T: ReadAt + ?Sized> ReadAt for Arc<T> {
    fn len(&self) -> io::Result<u64> {
        (**self).len()
    }
    fn read_exact_at(&self, offset: u64, dst: &mut [u8]) -> io::Result<()> {
        (**self).read_exact_at(offset, dst)
    }
}

/// How much of the block index `Dataset::open` checks up front.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum IndexValidation {
    /// Check each block's offsets when it is read. Opening stays O(1).
    #[default]
    OnDemand,
    /// Read the whole index once at open and check that every block span is
    /// ordered, non-overlapping and within bounds.
    Full,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct OpenOptions {
    pub index_validation: IndexValidation,
}

/// Reusable per-thread buffers for decoding blocks.
#[derive(Default)]
pub struct Worker {
    pub(crate) payload: Vec<u8>,
    pub(crate) records: Vec<Record>,
}

/// One segment of an FSQD file: header, model, blocks, index and footer.
///
/// A file is one or more segments back to back, so standalone files can be
/// joined with `cat`. Block offsets in a segment's index and its footer's
/// `index_start` are relative to the segment's first byte.
pub struct Segment {
    start: u64,
    model: Arc<Model>,
    records: u64,
    blocks: u64,
    /// First record and block of this segment in file-wide numbering.
    record_base: u64,
    block_base: u64,
    /// Relative to `start`.
    first_block: u64,
    index_start: u64,
}

impl Segment {
    /// Byte offset of the segment in the file.
    pub fn start(&self) -> u64 {
        self.start
    }
    pub fn model(&self) -> &Arc<Model> {
        &self.model
    }
    pub fn records(&self) -> u64 {
        self.records
    }
    pub fn blocks(&self) -> u64 {
        self.blocks
    }
    /// File-wide ordinal of the segment's first record.
    pub fn first_record(&self) -> u64 {
        self.record_base
    }
    /// File-wide ID of the segment's first block.
    pub fn first_block_id(&self) -> u64 {
        self.block_base
    }

    fn block_records(&self, local: u64) -> u64 {
        if local + 1 < self.blocks {
            BLOCK_RECORDS as u64
        } else {
            self.records - local * BLOCK_RECORDS as u64
        }
    }

    fn check_span(&self, local: u64, start: u64, end: u64) -> Result<()> {
        let id = self.block_base + local;
        if start < self.first_block || end > self.index_start || start >= end {
            return Err(invalid(format!("block {id} has an invalid span {start}..{end}")));
        }
        let len = end - start;
        if len < 4 || len > max_block_len(self.block_records(local)) {
            return Err(invalid(format!("block {id} has an impossible length {len}")));
        }
        Ok(())
    }
}

/// An opened FSQD file: source and its segments. Blocks and records are
/// numbered across all segments in file order.
pub struct Dataset<S> {
    source: S,
    segments: Vec<Segment>,
    records: u64,
    blocks: u64,
}

fn invalid(msg: impl Into<String>) -> Error {
    Error::InvalidFile(msg.into())
}

fn read_u64(buf: &[u8]) -> u64 {
    u64::from_le_bytes(buf[..8].try_into().unwrap())
}

fn read_exact<S: ReadAt>(src: &S, offset: u64, dst: &mut [u8]) -> Result<()> {
    src.read_exact_at(offset, dst).map_err(|e| {
        if e.kind() == io::ErrorKind::UnexpectedEof {
            Error::Truncated { block: None, record: None }
        } else {
            Error::Io(e)
        }
    })
}

impl<S: ReadAt> Dataset<S> {
    pub fn open(source: S) -> Result<Self> {
        Self::open_with(source, OpenOptions::default())
    }

    /// Walk the segments backwards from the end of the file: each footer gives
    /// its segment's length, which locates the previous footer. Segments with
    /// byte-identical models share one decoded model.
    pub fn open_with(source: S, options: OpenOptions) -> Result<Self> {
        let len = source.len()?;
        if len < HEADER_LEN + FOOTER_LEN {
            return Err(Error::Truncated { block: None, record: None });
        }
        let mut models: HashMap<Vec<u8>, Arc<Model>> = HashMap::new();
        let mut segments = Vec::new();
        let mut end = len;
        while end > 0 {
            let seg = Self::read_segment(&source, end, &mut models, segments.len())?;
            end = seg.start;
            segments.push(seg);
        }
        segments.reverse();
        let (mut records, mut blocks) = (0u64, 0u64);
        for seg in &mut segments {
            seg.record_base = records;
            seg.block_base = blocks;
            records = records.checked_add(seg.records).ok_or_else(|| invalid("record count overflows"))?;
            blocks += seg.blocks;
        }
        let ds = Self { source, segments, records, blocks };
        for seg in &ds.segments {
            if seg.blocks > 0 {
                let (start, _) = ds.block_span(seg, 0)?;
                if start != seg.start + seg.first_block {
                    return Err(invalid(format!("segment at {}: first block does not follow the model", seg.start)));
                }
            }
            if options.index_validation == IndexValidation::Full {
                ds.validate_index(seg)?;
            }
        }
        Ok(ds)
    }

    /// Parse the segment that ends at byte `end`.
    fn read_segment(
        source: &S,
        end: u64,
        models: &mut HashMap<Vec<u8>, Arc<Model>>,
        from_end: usize,
    ) -> Result<Segment> {
        let at = |msg: &str| invalid(format!("segment {from_end} from the end (ending at byte {end}): {msg}"));
        if end < HEADER_LEN + FOOTER_LEN {
            return Err(at("too short for a header and footer"));
        }
        let mut footer = [0u8; FOOTER_LEN as usize];
        read_exact(source, end - FOOTER_LEN, &mut footer)?;
        if &footer[20..] != FOOTER_MAGIC {
            return Err(at("bad footer magic"));
        }
        if footer[16..20] != [0; 4] {
            return Err(at("reserved footer bits set"));
        }
        let records = read_u64(&footer[..8]);
        let index_start = read_u64(&footer[8..16]);
        let blocks = records.div_ceil(BLOCK_RECORDS as u64);
        let seg_len = blocks
            .checked_mul(8)
            .and_then(|b| b.checked_add(index_start))
            .and_then(|b| b.checked_add(FOOTER_LEN))
            .ok_or_else(|| at("index size overflows"))?;
        if seg_len > end {
            return Err(at("segment extends before the start of the file"));
        }
        let start = end - seg_len;

        let mut header = [0u8; HEADER_LEN as usize];
        read_exact(source, start, &mut header)?;
        if &header[..4] != MAGIC {
            return Err(at("bad magic"));
        }
        let version = u16::from_le_bytes([header[4], header[5]]);
        if version != VERSION {
            return Err(at(&format!("unsupported version {version}")));
        }
        if u16::from_le_bytes([header[6], header[7]]) != 0 {
            return Err(at("unknown flags"));
        }
        let model_len = read_u64(&header[8..]);
        if model_len > model::max_serialized_len() as u64 || HEADER_LEN + model_len > index_start {
            return Err(at(&format!("model length {model_len} out of bounds")));
        }
        let first_block = HEADER_LEN + model_len;
        if blocks == 0 && index_start != first_block {
            return Err(at("empty segment has data between model and index"));
        }
        let mut model_bytes = vec![0u8; model_len as usize];
        read_exact(source, start + HEADER_LEN, &mut model_bytes)?;
        let model = match models.get(&model_bytes) {
            Some(m) => m.clone(),
            None => {
                let m = Arc::new(Model::from_bytes(&model_bytes)?);
                models.insert(model_bytes, m.clone());
                m
            }
        };
        Ok(Segment { start, model, records, blocks, record_base: 0, block_base: 0, first_block, index_start })
    }

    fn validate_index(&self, seg: &Segment) -> Result<()> {
        const CHUNK: u64 = 8192;
        if seg.blocks == 0 {
            return Ok(());
        }
        let mut prev = seg.first_block;
        let mut buf = vec![0u8; (CHUNK * 8) as usize];
        let mut j = 0u64;
        while j < seg.blocks {
            let n = CHUNK.min(seg.blocks - j);
            let buf = &mut buf[..(n * 8) as usize];
            read_exact(&self.source, seg.start + seg.index_start + j * 8, buf)?;
            for (i, c) in buf.as_chunks::<8>().0.iter().enumerate() {
                let start = read_u64(c);
                let local = j + i as u64;
                if local > 0 {
                    seg.check_span(local - 1, prev, start)?;
                } else if start != seg.first_block {
                    return Err(invalid(format!("segment at {}: first block does not follow the model", seg.start)));
                }
                prev = start;
            }
            j += n;
        }
        seg.check_span(seg.blocks - 1, prev, seg.index_start)
    }

    /// Absolute byte range of local block `local` of `seg`.
    fn block_span(&self, seg: &Segment, local: u64) -> Result<(u64, u64)> {
        let mut buf = [0u8; 16];
        let last = local + 1 == seg.blocks;
        let n = if last { 8 } else { 16 };
        read_exact(&self.source, seg.start + seg.index_start + local * 8, &mut buf[..n])?;
        let start = read_u64(&buf);
        let end = if last { seg.index_start } else { read_u64(&buf[8..]) };
        seg.check_span(local, start, end)?;
        Ok((seg.start + start, seg.start + end))
    }

    /// The segment holding file-wide block `id` (which must be < blocks()).
    fn segment_of_block(&self, id: u64) -> &Segment {
        let i = self.segments.partition_point(|s| s.block_base + s.blocks <= id);
        &self.segments[i]
    }

    pub fn records(&self) -> u64 {
        self.records
    }

    pub fn blocks(&self) -> u64 {
        self.blocks
    }

    pub fn segments(&self) -> &[Segment] {
        &self.segments
    }

    pub fn worker(&self) -> Worker {
        Worker::default()
    }

    /// Number of records in file-wide block `id`.
    pub fn block_len(&self, id: u64) -> Result<u64> {
        if id >= self.blocks {
            return Err(Error::OutOfRange { id, len: self.blocks });
        }
        let seg = self.segment_of_block(id);
        Ok(seg.block_records(id - seg.block_base))
    }

    /// Decode file-wide block `id` into the worker's buffer. On error the
    /// worker holds no partial batch.
    pub fn decode_block<'a>(&self, id: u64, worker: &'a mut Worker) -> Result<&'a [Record]> {
        worker.records.clear();
        if id >= self.blocks {
            return Err(Error::OutOfRange { id, len: self.blocks });
        }
        let seg = self.segment_of_block(id);
        let local = id - seg.block_base;
        let (start, end) = self.block_span(seg, local)?;
        worker.payload.resize((end - start) as usize, 0);
        read_exact(&self.source, start, &mut worker.payload)?;
        let n = seg.block_records(local) as usize;
        codec::decode_block(&seg.model, &worker.payload, n, id, &mut worker.records)?;
        Ok(&worker.records)
    }

    /// Append records `range` (file-wide record ordinals) to `out`, decoding
    /// only the blocks that overlap it. On error nothing is appended.
    pub fn read_range(&self, range: Range<u64>, worker: &mut Worker, out: &mut Vec<Record>) -> Result<()> {
        if range.start >= range.end {
            return Ok(());
        }
        if range.end > self.records {
            return Err(Error::OutOfRange { id: range.end - 1, len: self.records });
        }
        let before = out.len();
        let b = BLOCK_RECORDS as u64;
        let first = self.segments.partition_point(|s| s.record_base + s.records <= range.start);
        for seg in &self.segments[first..] {
            if seg.record_base >= range.end {
                break;
            }
            // Range within this segment, in segment-local record ordinals.
            let lo = range.start.max(seg.record_base) - seg.record_base;
            let hi = range.end.min(seg.record_base + seg.records) - seg.record_base;
            if lo >= hi {
                continue;
            }
            for local in lo / b..=(hi - 1) / b {
                let recs = match self.decode_block(seg.block_base + local, worker) {
                    Ok(r) => r,
                    Err(e) => {
                        out.truncate(before);
                        return Err(e);
                    }
                };
                let base = local * b;
                let a = lo.max(base) - base;
                let z = hi.min(base + recs.len() as u64) - base;
                out.extend_from_slice(&recs[a as usize..z as usize]);
            }
        }
        Ok(())
    }

    /// Iterate over all blocks in order, decoding each into an owned batch.
    pub fn iter_blocks(&self) -> BlockIter<'_, S> {
        BlockIter { ds: self, next: 0, worker: Worker::default() }
    }
}

pub struct BlockIter<'a, S> {
    ds: &'a Dataset<S>,
    next: u64,
    worker: Worker,
}

impl<S: ReadAt> Iterator for BlockIter<'_, S> {
    type Item = Result<Vec<Record>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next >= self.ds.blocks {
            return None;
        }
        let id = self.next;
        self.next += 1;
        Some(self.ds.decode_block(id, &mut self.worker).map(|r| r.to_vec()))
    }
}

/// Encodes blocks away from the [`Writer`], so several threads can encode
/// while one thread appends the blocks in order with [`Writer::push_encoded`].
pub struct BlockEncoder {
    model: Arc<Model>,
    scratch: BlockScratch,
}

/// One encoded block: its payload and record count.
pub struct EncodedBlock {
    model: Arc<Model>,
    payload: Vec<u8>,
    records: usize,
}

impl EncodedBlock {
    pub fn records(&self) -> usize {
        self.records
    }

    /// Payload length in bytes.
    pub fn len(&self) -> usize {
        self.payload.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records == 0
    }
}

impl BlockEncoder {
    pub fn new(model: Arc<Model>) -> Self {
        Self { model, scratch: BlockScratch::default() }
    }

    /// Encode 1 to 256 records as one block.
    pub fn encode(&mut self, records: &[Record]) -> Result<EncodedBlock> {
        if records.is_empty() || records.len() > BLOCK_RECORDS {
            return Err(Error::InvalidRecord(format!("a block holds 1 to 256 records, not {}", records.len())));
        }
        let mut payload = Vec::new();
        codec::encode_block(&self.model, records, &mut self.scratch, &mut payload);
        Ok(EncodedBlock { model: self.model.clone(), payload, records: records.len() })
    }
}

/// Streaming FSQD writer. Call [`Writer::finish`] to write the index and
/// footer; a dropped writer leaves an incomplete file.
pub struct Writer<W: Write> {
    sink: Option<W>,
    model: Arc<Model>,
    pending: Vec<Record>,
    offsets: Vec<u64>,
    offset: u64,
    records: u64,
    /// A block shorter than 256 records was written: it must be the last.
    short_written: bool,
    scratch: BlockScratch,
    payload: Vec<u8>,
}

impl<W: Write> Writer<W> {
    /// Write the header and model. Offsets are counted from the writer's
    /// first byte, so writing to a file opened in append mode adds a new
    /// segment to it.
    pub fn new(mut sink: W, model: Arc<Model>) -> Result<Self> {
        let model_bytes = model.to_bytes();
        let mut header = Vec::with_capacity(HEADER_LEN as usize);
        header.extend_from_slice(MAGIC);
        header.extend_from_slice(&VERSION.to_le_bytes());
        header.extend_from_slice(&0u16.to_le_bytes());
        header.extend_from_slice(&(model_bytes.len() as u64).to_le_bytes());
        sink.write_all(&header)?;
        sink.write_all(&model_bytes)?;
        Ok(Self {
            sink: Some(sink),
            model,
            pending: Vec::with_capacity(BLOCK_RECORDS),
            offsets: Vec::new(),
            offset: HEADER_LEN + model_bytes.len() as u64,
            records: 0,
            short_written: false,
            scratch: BlockScratch::default(),
            payload: Vec::new(),
        })
    }

    pub fn push(&mut self, record: &Record) -> Result<()> {
        if self.sink.is_none() {
            return Err(Error::WriterPoisoned);
        }
        self.pending.push(*record);
        if self.pending.len() == BLOCK_RECORDS {
            self.flush_block()?;
        }
        Ok(())
    }

    /// Append a block from a [`BlockEncoder`] with this writer's model. Every
    /// block but the last must hold 256 records, and no records pushed with
    /// [`Writer::push`] may be pending.
    pub fn push_encoded(&mut self, block: &EncodedBlock) -> Result<()> {
        if self.sink.is_none() {
            return Err(Error::WriterPoisoned);
        }
        if !Arc::ptr_eq(&block.model, &self.model) && block.model.to_bytes() != self.model.to_bytes() {
            return Err(Error::InvalidModel("the block was encoded with a different model".into()));
        }
        if !self.pending.is_empty() {
            return Err(Error::InvalidRecord("push_encoded called with pushed records pending".into()));
        }
        self.write_block(&block.payload, block.records)
    }

    fn flush_block(&mut self) -> Result<()> {
        if self.pending.is_empty() {
            return Ok(());
        }
        let mut payload = std::mem::take(&mut self.payload);
        payload.clear();
        codec::encode_block(&self.model, &self.pending, &mut self.scratch, &mut payload);
        let res = self.write_block(&payload, self.pending.len());
        self.payload = payload;
        self.pending.clear();
        res
    }

    fn write_block(&mut self, payload: &[u8], records: usize) -> Result<()> {
        if self.short_written {
            return Err(Error::InvalidRecord("only the last block may hold fewer than 256 records".into()));
        }
        let sink = self.sink.as_mut().ok_or(Error::WriterPoisoned)?;
        if let Err(e) = sink.write_all(payload) {
            self.sink = None;
            return Err(e.into());
        }
        self.offsets.push(self.offset);
        self.offset += payload.len() as u64;
        self.records += records as u64;
        self.short_written = records < BLOCK_RECORDS;
        Ok(())
    }

    /// Flush the last block, then write the index and footer.
    pub fn finish(mut self) -> Result<W> {
        self.flush_block()?;
        let mut sink = self.sink.take().ok_or(Error::WriterPoisoned)?;
        let mut tail = Vec::with_capacity(self.offsets.len() * 8 + FOOTER_LEN as usize);
        for o in &self.offsets {
            tail.extend_from_slice(&o.to_le_bytes());
        }
        tail.extend_from_slice(&self.records.to_le_bytes());
        tail.extend_from_slice(&self.offset.to_le_bytes());
        tail.extend_from_slice(&0u32.to_le_bytes());
        tail.extend_from_slice(FOOTER_MAGIC);
        sink.write_all(&tail)?;
        sink.flush()?;
        Ok(sink)
    }

    pub fn records(&self) -> u64 {
        self.records + self.pending.len() as u64
    }
}
