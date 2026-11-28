//! Join FSQD files into one file in random record order, in a single pass.
//!
//! Every input is cut into runs of `--run-blocks` consecutive blocks. The
//! runs of all inputs are read in one random order, so any part of any file
//! is equally likely to come early or late. Their records pass through a
//! shuffle buffer of `--buffer-gb`: each record replaces a random slot and
//! the evicted record is written. Records of one run end up spread over
//! about the buffer size, so the order is close to, but not exactly, a
//! uniform permutation. The result depends only on the inputs, the seed, the
//! run size and the buffer size.
//!
//! Pipeline: reader threads decode runs, the main thread shuffles, encoder
//! threads encode blocks, and one writer thread appends them in order.

use std::collections::{BTreeMap, HashSet};
use std::fs::{self, File};
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use clap::Parser;
use four_square::{BlockEncoder, Dataset, EncodedBlock, Model, Record, Writer, BLOCK_RECORDS};

/// Records per encoder job: 64 full blocks.
const BATCH: usize = 64 * BLOCK_RECORDS;

/// Join FSQD files into one FSQD file in random record order.
#[derive(Parser, Debug)]
#[command(version)]
struct Cli {
    /// FSQD files or directories; directories are searched recursively for *.fsqd.
    #[arg(required = true)]
    inputs: Vec<PathBuf>,

    /// Output file. Written as <output>.tmp and renamed when complete.
    #[arg(short, long)]
    output: PathBuf,

    /// Skip discovered files whose path contains this text (repeatable).
    #[arg(long)]
    exclude: Vec<String>,

    /// List the inputs and exit.
    #[arg(long)]
    list: bool,

    /// Overwrite an existing output.
    #[arg(long)]
    force: bool,

    /// Shuffle buffer size in GB of decoded records (70 bytes each).
    #[arg(long, default_value_t = 12.0)]
    buffer_gb: f64,

    /// Consecutive blocks (256 records each) read as one unit.
    #[arg(long, default_value_t = 16)]
    run_blocks: u64,

    /// Seed for the run order and the shuffle buffer.
    #[arg(long, default_value_t = 1)]
    seed: u64,

    /// Decoder threads.
    #[arg(long, default_value_t = 8)]
    readers: usize,

    /// Encoder threads.
    #[arg(long, default_value_t = 8)]
    encoders: usize,
}

// ---------------------------------------------------------------- rng

/// xoshiro256++ seeded through SplitMix64.
struct Rng([u64; 4]);

impl Rng {
    fn new(seed: u64) -> Self {
        let mut x = seed;
        let mut next = || {
            x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        };
        Self([next(), next(), next(), next()])
    }

    #[inline(always)]
    fn next(&mut self) -> u64 {
        let s = &mut self.0;
        let r = s[0].wrapping_add(s[3]).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        r
    }

    /// Uniform in 0..n by multiply-shift. The bias is at most n / 2^64.
    #[inline(always)]
    fn below(&mut self, n: u64) -> u64 {
        ((self.next() as u128 * n as u128) >> 64) as u64
    }

    fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            v.swap(i, self.below(i as u64 + 1) as usize);
        }
    }
}

// ---------------------------------------------------------------- inputs

struct Input {
    path: PathBuf,
    ds: Dataset<File>,
}

fn discover(cli: &Cli) -> Result<Vec<PathBuf>, String> {
    let skip: Vec<PathBuf> =
        [cli.output.clone(), tmp_path(&cli.output)].iter().filter_map(|p| p.canonicalize().ok()).collect();
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for input in &cli.inputs {
        let meta = fs::metadata(input).map_err(|e| format!("{}: {e}", input.display()))?;
        let mut found = Vec::new();
        if meta.is_dir() {
            for entry in walkdir::WalkDir::new(input).follow_links(true) {
                let entry = entry.map_err(|e| e.to_string())?;
                if entry.file_type().is_file() && entry.path().extension().is_some_and(|x| x == "fsqd") {
                    found.push(entry.into_path());
                }
            }
            found.sort();
        } else {
            found.push(input.clone());
        }
        for path in found {
            let shown = path.to_string_lossy();
            if cli.exclude.iter().any(|x| shown.contains(x.as_str())) {
                continue;
            }
            let canon = path.canonicalize().map_err(|e| format!("{}: {e}", path.display()))?;
            if skip.contains(&canon) || !seen.insert(canon) {
                continue;
            }
            out.push(path);
        }
    }
    Ok(out)
}

fn tmp_path(output: &Path) -> PathBuf {
    let mut s = output.as_os_str().to_owned();
    s.push(".tmp");
    PathBuf::from(s)
}

fn gb(bytes: u64) -> f64 {
    bytes as f64 / 1e9
}

/// A unit of reading: `blocks` consecutive blocks of one input.
#[derive(Clone, Copy)]
struct Run {
    input: u32,
    first: u64,
    blocks: u32,
}

// ---------------------------------------------------------------- stages

/// Lets readers run at most `window` runs ahead of the consumer.
struct Window {
    consumed: Mutex<u64>,
    cv: Condvar,
    window: u64,
}

impl Window {
    fn wait_for(&self, run: u64) {
        let mut c = self.consumed.lock().unwrap();
        while run >= *c + self.window {
            c = self.cv.wait(c).unwrap();
        }
    }

    fn advance(&self, to: u64) {
        *self.consumed.lock().unwrap() = to;
        self.cv.notify_all();
    }
}

type ReadResult = (u64, Result<Vec<Record>, String>);

fn reader(inputs: &[Input], runs: &[Run], next: &AtomicUsize, window: &Window, tx: mpsc::Sender<ReadResult>) {
    let mut worker = four_square::Worker::default();
    loop {
        let i = next.fetch_add(1, Ordering::Relaxed);
        if i >= runs.len() {
            return;
        }
        window.wait_for(i as u64);
        let run = runs[i];
        let input = &inputs[run.input as usize];
        let mut recs = Vec::with_capacity(run.blocks as usize * BLOCK_RECORDS);
        let mut res = Ok(());
        for b in run.first..run.first + run.blocks as u64 {
            match input.ds.decode_block(b, &mut worker) {
                Ok(r) => recs.extend_from_slice(r),
                Err(e) => {
                    res = Err(format!("{}: block {b}: {e}", input.path.display()));
                    break;
                }
            }
        }
        if tx.send((i as u64, res.map(|_| recs))).is_err() {
            return;
        }
    }
}

type EncodeJob = (u64, Vec<Record>);
type EncodeResult = (u64, Result<Vec<EncodedBlock>, String>);

fn encoder(model: Arc<Model>, jobs: &Mutex<Receiver<EncodeJob>>, tx: SyncSender<EncodeResult>) {
    let mut enc = BlockEncoder::new(model);
    loop {
        let job = jobs.lock().unwrap().recv();
        let Ok((seq, recs)) = job else { return };
        let blocks: Result<Vec<_>, _> =
            recs.chunks(BLOCK_RECORDS).map(|c| enc.encode(c).map_err(|e| e.to_string())).collect();
        if tx.send((seq, blocks)).is_err() {
            return;
        }
    }
}

/// Append encoded batches in sequence order.
fn writer(
    mut w: Writer<BufWriter<File>>,
    rx: Receiver<EncodeResult>,
    bytes: &AtomicU64,
    written: &AtomicU64,
) -> Result<File, String> {
    let mut pending = BTreeMap::new();
    let mut next = 0u64;
    for (seq, blocks) in rx {
        pending.insert(seq, blocks?);
        while let Some(blocks) = pending.remove(&next) {
            for b in &blocks {
                w.push_encoded(b).map_err(|e| format!("writing the output: {e}"))?;
                bytes.fetch_add(b.len() as u64, Ordering::Relaxed);
                written.fetch_add(b.records() as u64, Ordering::Relaxed);
            }
            next += 1;
        }
    }
    if !pending.is_empty() {
        return Err("encoder batches missing".into());
    }
    let sink = w.finish().map_err(|e| format!("finishing the output: {e}"))?;
    let file = sink.into_inner().map_err(|e| format!("flushing the output: {}", e.error()))?;
    file.sync_all().map_err(|e| format!("syncing the output: {e}"))?;
    Ok(file)
}

/// The shuffle buffer, feeding full batches to the encoders.
struct Shuffler {
    buf: Vec<Record>,
    cap: usize,
    rng: Rng,
    batch: Vec<Record>,
    seq: u64,
    jobs: SyncSender<EncodeJob>,
    slots: Vec<usize>,
}

impl Shuffler {
    fn push_all(&mut self, recs: &[Record]) -> Result<(), String> {
        let mut recs = recs;
        if self.buf.len() < self.cap {
            let n = recs.len().min(self.cap - self.buf.len());
            self.buf.extend_from_slice(&recs[..n]);
            recs = &recs[n..];
        }
        // Draw the slots first and prefetch ahead: the buffer is far larger
        // than the caches, so each replacement is a memory miss.
        self.slots.clear();
        self.slots.extend(recs.iter().map(|_| self.rng.below(self.cap as u64) as usize));
        const AHEAD: usize = 16;
        for (k, rec) in recs.iter().enumerate() {
            if let Some(&j) = self.slots.get(k + AHEAD) {
                prefetch(&self.buf[j]);
            }
            let j = self.slots[k];
            let out = std::mem::replace(&mut self.buf[j], *rec);
            self.emit(out)?;
        }
        Ok(())
    }

    fn emit(&mut self, r: Record) -> Result<(), String> {
        self.batch.push(r);
        if self.batch.len() == BATCH {
            self.flush()?;
        }
        Ok(())
    }

    fn flush(&mut self) -> Result<(), String> {
        if self.batch.is_empty() {
            return Ok(());
        }
        let batch = std::mem::replace(&mut self.batch, Vec::with_capacity(BATCH));
        self.jobs.send((self.seq, batch)).map_err(|_| "the encoder or writer stopped".to_string())?;
        self.seq += 1;
        Ok(())
    }

    /// Shuffle what is left in the buffer and emit all of it.
    fn drain(mut self) -> Result<(), String> {
        let mut buf = std::mem::take(&mut self.buf);
        self.rng.shuffle(&mut buf);
        for r in buf {
            self.emit(r)?;
        }
        self.flush()
    }
}

#[inline(always)]
fn prefetch<T>(p: &T) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_prefetch(p as *const T as *const i8, std::arch::x86_64::_MM_HINT_T0);
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = p;
}

// ---------------------------------------------------------------- main

fn main() {
    let cli = Cli::parse();
    if let Err(e) = run(&cli) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run(cli: &Cli) -> Result<(), String> {
    if cli.run_blocks == 0 || cli.run_blocks > u32::MAX as u64 {
        return Err("--run-blocks must be at least 1".into());
    }
    let paths = discover(cli)?;
    if paths.is_empty() {
        return Err("no .fsqd inputs found".into());
    }

    let mut inputs = Vec::with_capacity(paths.len());
    let mut total_records = 0u64;
    let mut total_bytes = 0u64;
    for path in paths {
        let file = File::open(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let size = file.metadata().map_err(|e| format!("{}: {e}", path.display()))?.len();
        let ds = Dataset::open(file).map_err(|e| format!("{}: {e}", path.display()))?;
        if cli.list {
            println!(
                "{}  {} records, {} segments, {:.2} GB",
                path.display(),
                ds.records(),
                ds.segments().len(),
                gb(size)
            );
        }
        total_records += ds.records();
        total_bytes += size;
        inputs.push(Input { path, ds });
    }
    eprintln!("{} inputs, {} records, {:.1} GB", inputs.len(), total_records, gb(total_bytes));
    if cli.list || total_records == 0 {
        return Ok(());
    }
    if cli.output.exists() && !cli.force {
        return Err(format!("{} exists; pass --force to overwrite it", cli.output.display()));
    }

    // The output model: the first segment's. Records of inputs with another
    // model are re-encoded with it, which works but may compress worse.
    let model = inputs.iter().flat_map(|i| i.ds.segments()).next().unwrap().model().clone();
    let model_bytes = model.to_bytes();
    for input in &inputs {
        if input.ds.segments().iter().any(|s| !Arc::ptr_eq(s.model(), &model) && s.model().to_bytes() != model_bytes) {
            eprintln!("note: {} uses another model; its records are re-encoded", input.path.display());
        }
    }

    let mut runs = Vec::new();
    for (i, input) in inputs.iter().enumerate() {
        let blocks = input.ds.blocks();
        let mut b = 0;
        while b < blocks {
            let n = cli.run_blocks.min(blocks - b);
            runs.push(Run { input: i as u32, first: b, blocks: n as u32 });
            b += n;
        }
    }
    Rng::new(cli.seed).shuffle(&mut runs);

    let rec_size = std::mem::size_of::<Record>();
    let cap = ((cli.buffer_gb * 1e9) as usize / rec_size).clamp(1, total_records as usize);
    eprintln!(
        "{} runs of up to {} blocks, shuffle buffer {} records ({:.1} GB), seed {}",
        runs.len(),
        cli.run_blocks,
        cap,
        gb((cap * rec_size) as u64),
        cli.seed
    );

    let tmp = tmp_path(&cli.output);
    let file = File::create(&tmp).map_err(|e| format!("{}: {e}", tmp.display()))?;
    let w = Writer::new(BufWriter::with_capacity(8 << 20, file), model.clone())
        .map_err(|e| format!("{}: {e}", tmp.display()))?;

    let start = Instant::now();
    let readers = cli.readers.max(1);
    let encoders = cli.encoders.max(1);
    let window = Window { consumed: Mutex::new(0), cv: Condvar::new(), window: 8 * readers as u64 };
    let next_run = AtomicUsize::new(0);
    let bytes = AtomicU64::new(0);
    let written = AtomicU64::new(0);

    let result = std::thread::scope(|s| -> Result<(), String> {
        let (read_tx, read_rx) = mpsc::channel::<ReadResult>();
        for _ in 0..readers {
            let tx = read_tx.clone();
            let (inputs, runs, next_run, window) = (&inputs, &runs, &next_run, &window);
            s.spawn(move || reader(inputs, runs, next_run, window, tx));
        }
        drop(read_tx);

        let (job_tx, job_rx) = mpsc::sync_channel::<EncodeJob>(2 * encoders);
        let job_rx = Arc::new(Mutex::new(job_rx));
        let (enc_tx, enc_rx) = mpsc::sync_channel::<EncodeResult>(4 * encoders);
        for _ in 0..encoders {
            let (model, jobs, tx) = (model.clone(), job_rx.clone(), enc_tx.clone());
            s.spawn(move || encoder(model, &jobs, tx));
        }
        drop(enc_tx);
        let (bytes_ref, written_ref) = (&bytes, &written);
        let writer = s.spawn(move || writer(w, enc_rx, bytes_ref, written_ref));

        let mut shuffler = Shuffler {
            buf: Vec::with_capacity(cap),
            cap,
            rng: Rng::new(cli.seed ^ 0x5348_5546_464c_4521),
            batch: Vec::with_capacity(BATCH),
            seq: 0,
            jobs: job_tx,
            slots: Vec::new(),
        };
        let consume = || -> Result<(), String> {
            let mut pending = BTreeMap::new();
            let mut next = 0u64;
            let mut read = 0u64;
            let mut last = Instant::now();
            for (i, recs) in read_rx {
                pending.insert(i, recs?);
                while let Some(recs) = pending.remove(&next) {
                    shuffler.push_all(&recs)?;
                    read += recs.len() as u64;
                    next += 1;
                    window.advance(next);
                }
                if last.elapsed() >= Duration::from_secs(30) {
                    last = Instant::now();
                    let secs = start.elapsed().as_secs_f64();
                    let w = written.load(Ordering::Relaxed);
                    eprintln!(
                        "  {:.0}s: read {:.1}M of {:.1}M records ({:.1}%, {:.1}M/s), wrote {:.1}M ({:.1} GB)",
                        secs,
                        read as f64 / 1e6,
                        total_records as f64 / 1e6,
                        100.0 * read as f64 / total_records as f64,
                        read as f64 / 1e6 / secs,
                        w as f64 / 1e6,
                        gb(bytes.load(Ordering::Relaxed)),
                    );
                }
            }
            if next != runs.len() as u64 {
                return Err("a reader stopped early".into());
            }
            eprintln!("  {:.0}s: all records read, draining the shuffle buffer", start.elapsed().as_secs_f64());
            shuffler.drain()
        };
        let res = consume();
        // Release readers waiting for the window, so they see the closed
        // channel and stop.
        window.advance(u64::MAX / 2);
        // On error the channels close and the other threads stop; the
        // writer's error, if any, is the more useful one.
        let w = writer.join().unwrap();
        res?;
        w.map(|_| ())
    });
    if let Err(e) = result {
        let _ = fs::remove_file(&tmp);
        return Err(e);
    }

    let check = Dataset::open(File::open(&tmp).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    if check.records() != total_records {
        return Err(format!("{} has {} records, expected {total_records}", tmp.display(), check.records()));
    }
    fs::rename(&tmp, &cli.output).map_err(|e| format!("renaming to {}: {e}", cli.output.display()))?;
    let size = fs::metadata(&cli.output).map(|m| m.len()).unwrap_or(0);
    let secs = start.elapsed().as_secs_f64();
    eprintln!(
        "done in {:.0}s: {} records ({:.1}M/s) into {}, {:.1} GB, {:.2} B/record",
        secs,
        total_records,
        total_records as f64 / 1e6 / secs,
        cli.output.display(),
        gb(size),
        size as f64 / total_records as f64
    );
    Ok(())
}
