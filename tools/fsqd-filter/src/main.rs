//! Convert binpacks into FSQD files, keeping only the positions that the
//! training data loader's skip filter accepts (random FEN skipping is off).
//!
//! Every input `X.binpack` becomes `X.fsqd` in the output directory (input
//! subdirectories are mirrored). All outputs share one model, trained once
//! from a sample of all inputs and saved as `fsqd-model.bin` in the output
//! directory, so the files can be concatenated with `cat` and a rerun reuses
//! the same model. Finished outputs are skipped on a rerun.

use std::collections::HashSet;
use std::ffi::{c_char, c_int, c_void, CString};
use std::fs;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use clap::Parser;
use four_square::{BoardState, Labels, Model, Outcome, Record, Writer};

const MODEL_FILE: &str = "fsqd-model.bin";

/// Set by the first Ctrl-C: stop, remove unfinished outputs and exit.
static STOP: AtomicBool = AtomicBool::new(false);

fn stopped() -> bool {
    STOP.load(Ordering::Relaxed)
}

/// First Ctrl-C asks for a clean stop; a second one exits at once.
fn install_ctrl_c() {
    let res = ctrlc::set_handler(|| {
        if STOP.swap(true, Ordering::SeqCst) {
            eprintln!("\nsecond Ctrl-C: exiting now; unfinished .tmp files may remain");
            std::process::exit(130);
        }
        eprintln!("\nCtrl-C: stopping and removing unfinished outputs (Ctrl-C again to exit at once)");
    });
    if let Err(e) = res {
        eprintln!("warning: cannot install the Ctrl-C handler: {e}");
    }
}

// ---------------------------------------------------------------- C ABI

/// Mirror of `DataloaderSkipConfig` in data_loader/cpp/training_data_loader_structs.h.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct SkipConfig {
    filtered: bool,
    random_fen_skipping: c_int,
    wld_filtered: bool,
    early_fen_skipping: c_int,
    soft_early_fen_skipping: c_int,
    simple_eval_skipping: c_int,
    param_index: c_int,
    pc_y0: f64,
    pc_y1: f64,
    pc_y2: f64,
    pc_y3: f64,
    pc_y4: f64,
    ply_x1: f64,
    ply_y1: f64,
    ply_x2: f64,
    ply_y2: f64,
    ply_x3: f64,
    ply_y3: f64,
    ply_x4: f64,
    ply_y4: f64,
    tree_sample_accept: f64,
    decided_score: c_int,
    decided_accept: f64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ScanRecord {
    score: i16,
    result: i16,
    squares: [u8; 64],
    black_to_move: u8,
    castling: u8,
    ep: u8,
    tree_sample: u8,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
struct ScanStats {
    read: u64,
    kept: u64,
    tree_samples: u64,
}

type ScanSink = extern "C" fn(*mut c_void, *const ScanRecord, usize, u64) -> c_int;

extern "C" {
    fn fsqd_scan(
        path: *const c_char,
        config: SkipConfig,
        max_kept: u64,
        sink: ScanSink,
        ctx: *mut c_void,
        stats: *mut ScanStats,
        err: *mut c_char,
        err_len: usize,
    ) -> c_int;
}

/// Scan one binpack. `on_batch(records, entries read so far)` returns false
/// to stop. Returns the stats and whether the file was read to the end (or
/// `max_kept` was reached) rather than stopped.
fn scan(
    path: &Path,
    config: SkipConfig,
    max_kept: u64,
    mut on_batch: impl FnMut(&[ScanRecord], u64) -> bool,
) -> Result<(ScanStats, bool), String> {
    extern "C" fn trampoline<F: FnMut(&[ScanRecord], u64) -> bool>(
        ctx: *mut c_void,
        recs: *const ScanRecord,
        n: usize,
        read: u64,
    ) -> c_int {
        let f = unsafe { &mut *(ctx as *mut F) };
        let recs = unsafe { std::slice::from_raw_parts(recs, n) };
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(recs, read))) {
            Ok(true) => 0,
            _ => 1,
        }
    }
    fn tramp_for<F: FnMut(&[ScanRecord], u64) -> bool>(_: &F) -> ScanSink {
        trampoline::<F>
    }

    let cpath = CString::new(path.as_os_str().as_encoded_bytes()).map_err(|e| e.to_string())?;
    let sink = tramp_for(&on_batch);
    let mut stats = ScanStats::default();
    let mut err = [0 as c_char; 512];
    let rc = unsafe {
        fsqd_scan(
            cpath.as_ptr(),
            config,
            max_kept,
            sink,
            &mut on_batch as *mut _ as *mut c_void,
            &mut stats,
            err.as_mut_ptr(),
            err.len(),
        )
    };
    match rc {
        0 => Ok((stats, true)),
        1 => Ok((stats, false)),
        _ => Err(unsafe { std::ffi::CStr::from_ptr(err.as_ptr()) }.to_string_lossy().into_owned()),
    }
}

fn to_record(r: &ScanRecord) -> Option<Record> {
    let ep = (r.ep < 64).then_some(r.ep);
    let board = BoardState::new(r.squares, r.black_to_move != 0, r.castling, ep).ok()?;
    // Search-tree samples carry a placeholder result; they train on the score alone.
    let outcome = if r.tree_sample != 0 { Outcome::NoResult } else { Outcome::from_result(r.result).ok()? };
    Some(Record { board, labels: Labels { score: r.score, outcome } })
}

// ---------------------------------------------------------------- CLI

/// Filter binpacks with the training data loader's skip filter and write the
/// accepted positions of each input X.binpack to X.fsqd in the output
/// directory, converting several files in parallel.
///
/// Filter options use the same names and defaults as train.py
/// (data_loader/config.py DataloaderSkipConfig). Random FEN skipping is always
/// off. Search-tree samples (ply 9999) are written with outcome NoResult.
///
/// All outputs share one model, saved as fsqd-model.bin in the output
/// directory and reused by later runs. Outputs that already exist are skipped,
/// so an interrupted run can be restarted with the same command. The outputs
/// can be joined with `cat` into one multi-segment FSQD file.
#[derive(Parser, Debug)]
#[command(version, verbatim_doc_comment)]
struct Cli {
    /// Binpack files or directories; directories are searched recursively for *.binpack.
    #[arg(required = true)]
    inputs: Vec<PathBuf>,

    /// Output directory; created if missing. Subdirectories of input directories are mirrored.
    #[arg(short, long = "output-dir", visible_alias = "output")]
    output_dir: PathBuf,

    /// Skip discovered files whose path contains this text (repeatable).
    #[arg(long)]
    exclude: Vec<String>,

    /// List the inputs and their outputs and exit without reading them.
    #[arg(long)]
    list: bool,

    /// Re-convert files whose output exists and retrain the model.
    #[arg(long)]
    force: bool,

    /// Files converted in parallel.
    #[arg(long, default_value_t = 8)]
    threads: usize,

    /// Accepted positions sampled (spread evenly over the inputs) to train the model.
    #[arg(long, default_value_t = 4_000_000)]
    train_records: u64,

    // ---- skip filter, as in train.py ----
    /// Skip captures and positions in check.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    filtered: bool,
    /// Disable --filtered.
    #[arg(long, conflicts_with = "filtered")]
    no_filtered: bool,
    /// Skip positions by how well score and result agree.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    wld_filtered: bool,
    /// Disable --wld-filtered.
    #[arg(long, conflicts_with = "wld_filtered")]
    no_wld_filtered: bool,
    /// Skip positions up to this ply. -1 = disabled.
    #[arg(long, default_value_t = -1, allow_negative_numbers = true)]
    early_fen_skipping: i32,
    /// Soft ply filter below this ply. <= 0 = disabled.
    #[arg(long, alias = "soft-early-fen_skipping", default_value_t = 20, allow_negative_numbers = true)]
    soft_early_fen_skipping: i32,
    /// Skip positions with |simple eval| below this. -1 = disabled.
    #[arg(long, default_value_t = -1, allow_negative_numbers = true)]
    simple_eval_skipping: i32,
    #[arg(long, default_value_t = 0)]
    param_index: i32,
    #[arg(long, default_value_t = 0.0, allow_negative_numbers = true)]
    pc_y0: f64,
    #[arg(long, default_value_t = 0.4, allow_negative_numbers = true)]
    pc_y1: f64,
    #[arg(long, default_value_t = 1.0, allow_negative_numbers = true)]
    pc_y2: f64,
    #[arg(long, default_value_t = 1.0, allow_negative_numbers = true)]
    pc_y3: f64,
    #[arg(long, default_value_t = 0.75, allow_negative_numbers = true)]
    pc_y4: f64,
    #[arg(long, default_value_t = 0.0, allow_negative_numbers = true)]
    ply_x1: f64,
    #[arg(long, default_value_t = 0.1, allow_negative_numbers = true)]
    ply_y1: f64,
    #[arg(long, default_value_t = 6.0, allow_negative_numbers = true)]
    ply_x2: f64,
    #[arg(long, default_value_t = 0.15, allow_negative_numbers = true)]
    ply_y2: f64,
    #[arg(long, default_value_t = 10.0, allow_negative_numbers = true)]
    ply_x3: f64,
    #[arg(long, default_value_t = 0.25, allow_negative_numbers = true)]
    ply_y3: f64,
    #[arg(long, default_value_t = 18.0, allow_negative_numbers = true)]
    ply_x4: f64,
    #[arg(long, default_value_t = 0.75, allow_negative_numbers = true)]
    ply_y4: f64,
    /// Acceptance probability for search-tree samples (ply 9999).
    #[arg(long, default_value_t = 1.0)]
    tree_sample_accept: f64,
    /// Positions with |score| >= this are kept with probability --decided-accept. 0 = disabled.
    #[arg(long, default_value_t = 1000)]
    decided_score: i32,
    #[arg(long, default_value_t = 0.3)]
    decided_accept: f64,
}

impl Cli {
    fn skip_config(&self) -> SkipConfig {
        SkipConfig {
            filtered: self.filtered && !self.no_filtered,
            random_fen_skipping: 0,
            wld_filtered: self.wld_filtered && !self.no_wld_filtered,
            early_fen_skipping: self.early_fen_skipping,
            soft_early_fen_skipping: self.soft_early_fen_skipping,
            simple_eval_skipping: self.simple_eval_skipping,
            param_index: self.param_index,
            pc_y0: self.pc_y0,
            pc_y1: self.pc_y1,
            pc_y2: self.pc_y2,
            pc_y3: self.pc_y3,
            pc_y4: self.pc_y4,
            ply_x1: self.ply_x1,
            ply_y1: self.ply_y1,
            ply_x2: self.ply_x2,
            ply_y2: self.ply_y2,
            ply_x3: self.ply_x3,
            ply_y3: self.ply_y3,
            ply_x4: self.ply_x4,
            ply_y4: self.ply_y4,
            tree_sample_accept: self.tree_sample_accept,
            decided_score: self.decided_score,
            decided_accept: self.decided_accept,
        }
    }
}

// ---------------------------------------------------------------- inputs

struct Job {
    input: PathBuf,
    size: u64,
    output: PathBuf,
}

impl Job {
    fn tmp(&self) -> PathBuf {
        self.output.with_extension("fsqd.tmp")
    }
}

/// Find the inputs and map each `dir/sub/X.binpack` to `out/sub/X.fsqd`
/// (a file given directly maps to `out/X.fsqd`).
fn discover(cli: &Cli) -> Result<Vec<Job>, String> {
    let mut seen_inputs = HashSet::new();
    let mut seen_outputs: Vec<(PathBuf, PathBuf)> = Vec::new();
    let mut jobs = Vec::new();
    for input in &cli.inputs {
        let meta = fs::metadata(input).map_err(|e| format!("{}: {e}", input.display()))?;
        let mut found = Vec::new();
        if meta.is_dir() {
            for entry in walkdir::WalkDir::new(input).follow_links(true) {
                let entry = entry.map_err(|e| e.to_string())?;
                if entry.file_type().is_file() && entry.path().extension().is_some_and(|x| x == "binpack") {
                    let rel = entry.path().strip_prefix(input).unwrap().to_path_buf();
                    found.push((entry.into_path(), rel));
                }
            }
            found.sort();
        } else {
            let name = input.file_name().ok_or_else(|| format!("{}: not a file", input.display()))?;
            found.push((input.clone(), PathBuf::from(name)));
        }
        for (path, rel) in found {
            let shown = path.to_string_lossy();
            if cli.exclude.iter().any(|x| shown.contains(x.as_str())) {
                continue;
            }
            let canon = path.canonicalize().map_err(|e| format!("{}: {e}", path.display()))?;
            if !seen_inputs.insert(canon) {
                continue;
            }
            let size = fs::metadata(&path).map_err(|e| format!("{}: {e}", path.display()))?.len();
            fs::File::open(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            let output = cli.output_dir.join(rel).with_extension("fsqd");
            if let Some((other, _)) = seen_outputs.iter().find(|(_, o)| *o == output) {
                return Err(format!(
                    "{} and {} would both be written to {}",
                    other.display(),
                    path.display(),
                    output.display()
                ));
            }
            seen_outputs.push((path.clone(), output.clone()));
            jobs.push(Job { input: path, size, output });
        }
    }
    Ok(jobs)
}

fn gb(bytes: u64) -> f64 {
    bytes as f64 / 1e9
}

/// Run `job(index)` for `n` items on `threads` workers, taking items in order.
fn for_each(n: usize, threads: usize, job: impl Fn(usize) + Sync) {
    let next = AtomicUsize::new(0);
    std::thread::scope(|s| {
        for _ in 0..threads.clamp(1, n.max(1)) {
            s.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= n {
                    break;
                }
                job(i);
            });
        }
    });
}

// ---------------------------------------------------------------- model

/// Load the saved model, or train one from accepted positions sampled evenly
/// over all inputs and save it.
fn load_or_train_model(cli: &Cli, jobs: &[Job], config: SkipConfig) -> Result<Arc<Model>, String> {
    let path = cli.output_dir.join(MODEL_FILE);
    if path.exists() && !cli.force {
        let bytes = fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let model = Model::from_bytes(&bytes).map_err(|e| format!("{}: {e}", path.display()))?;
        eprintln!("using the saved model {} (pass --force to retrain)", path.display());
        return Ok(Arc::new(model));
    }

    let per_file = cli.train_records.div_ceil(jobs.len() as u64).max(1);
    eprintln!("training the model on up to {per_file} accepted positions per file");
    let start = Instant::now();
    let samples: Vec<Mutex<Vec<Record>>> = jobs.iter().map(|_| Mutex::new(Vec::new())).collect();
    let errors = Mutex::new(Vec::new());
    for_each(jobs.len(), cli.threads, |i| {
        let mut local = Vec::new();
        if stopped() {
            return;
        }
        let res = scan(&jobs[i].input, config, per_file, |batch, _| {
            local.extend(batch.iter().filter_map(to_record));
            !stopped()
        });
        match res {
            Ok(_) => *samples[i].lock().unwrap() = local,
            Err(e) => errors.lock().unwrap().push(format!("{}: {e}", jobs[i].input.display())),
        }
    });
    if stopped() {
        return Err("interrupted while training the model; nothing was written".into());
    }
    let errors = errors.into_inner().unwrap();
    if !errors.is_empty() {
        return Err(errors.join("\n"));
    }
    // Concatenate in input order so the model does not depend on thread timing.
    let train: Vec<Record> = samples.into_iter().flat_map(|m| m.into_inner().unwrap()).collect();
    let model = Model::train(|| Ok(train.iter().copied().map(Ok))).map_err(|e| e.to_string())?;
    let tmp = path.with_extension("bin.tmp");
    fs::write(&tmp, model.to_bytes()).map_err(|e| format!("{}: {e}", tmp.display()))?;
    fs::rename(&tmp, &path).map_err(|e| format!("{}: {e}", path.display()))?;
    eprintln!(
        "trained on {} positions in {:.0}s, saved {}",
        train.len(),
        start.elapsed().as_secs_f64(),
        path.display()
    );
    Ok(Arc::new(model))
}

// ---------------------------------------------------------------- conversion

#[derive(Default)]
struct Progress {
    read: AtomicU64,
    written: AtomicU64,
    bytes_done: AtomicU64,
    files_done: AtomicUsize,
}

#[derive(Default)]
struct Totals {
    read: u64,
    written: u64,
    tree: u64,
    invalid: u64,
    out_bytes: u64,
}

struct FileResult {
    stats: ScanStats,
    written: u64,
    invalid: u64,
    out_bytes: u64,
}

/// Convert one binpack into its FSQD file via `<output>.fsqd.tmp`, which is
/// removed on failure and renamed on success.
fn convert_one(job: &Job, config: SkipConfig, model: &Arc<Model>, progress: &Progress) -> Result<FileResult, String> {
    let tmp = job.tmp();
    let fail = |e: String| {
        let _ = fs::remove_file(&tmp);
        Err(e)
    };
    if let Some(dir) = tmp.parent() {
        fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    let file = fs::File::create(&tmp).map_err(|e| format!("{}: {e}", tmp.display()))?;
    let mut writer = match Writer::new(BufWriter::with_capacity(8 << 20, file), model.clone()) {
        Ok(w) => w,
        Err(e) => return fail(e.to_string()),
    };

    let mut write_err = None;
    let (mut invalid, mut last_read) = (0u64, 0u64);
    let res = scan(&job.input, config, 0, |batch, read| {
        progress.read.fetch_add(read - last_read, Ordering::Relaxed);
        last_read = read;
        for r in batch {
            match to_record(r) {
                Some(rec) => {
                    if let Err(e) = writer.push(&rec) {
                        write_err = Some(format!("writing {}: {e}", tmp.display()));
                        return false;
                    }
                    progress.written.fetch_add(1, Ordering::Relaxed);
                }
                None => invalid += 1,
            }
        }
        !stopped()
    });
    let stats = match (res, write_err) {
        (_, Some(e)) | (Err(e), None) => return fail(e),
        (Ok((_, false)), None) => return fail("interrupted".into()),
        (Ok((stats, true)), None) => stats,
    };
    progress.read.fetch_add(stats.read - last_read, Ordering::Relaxed);
    let written = writer.records();
    let file = match writer.finish().map_err(|e| e.to_string()).and_then(|w| w.into_inner().map_err(|e| e.to_string()))
    {
        Ok(f) => f,
        Err(e) => return fail(format!("finishing {}: {e}", tmp.display())),
    };
    if let Err(e) = file.sync_all() {
        return fail(format!("syncing {}: {e}", tmp.display()));
    }
    drop(file);
    if let Err(e) = fs::rename(&tmp, &job.output) {
        return fail(format!("rename to {}: {e}", job.output.display()));
    }
    let out_bytes = fs::metadata(&job.output).map(|m| m.len()).unwrap_or(0);
    Ok(FileResult { stats, written, invalid, out_bytes })
}

fn main() {
    let cli = Cli::parse();
    install_ctrl_c();
    match run(&cli) {
        Ok(true) => {}
        Ok(false) if stopped() => std::process::exit(130),
        Ok(false) => std::process::exit(1),
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(if stopped() { 130 } else { 1 });
        }
    }
}

fn report_progress(progress: &Progress, files: usize, total_bytes: u64, start: Instant) {
    let secs = start.elapsed().as_secs_f64();
    let read = progress.read.load(Ordering::Relaxed);
    let written = progress.written.load(Ordering::Relaxed);
    eprintln!(
        "  {:.0}s: {}/{} files done ({:.1}/{:.1} GB), read {:.1}M positions ({:.1}M/s), wrote {:.1}M ({:.2}%)",
        secs,
        progress.files_done.load(Ordering::Relaxed),
        files,
        gb(progress.bytes_done.load(Ordering::Relaxed)),
        gb(total_bytes),
        read as f64 / 1e6,
        read as f64 / 1e6 / secs,
        written as f64 / 1e6,
        100.0 * written as f64 / read.max(1) as f64,
    );
}

/// Returns Ok(false) if some files failed.
fn run(cli: &Cli) -> Result<bool, String> {
    if cli.output_dir.exists() && !cli.output_dir.is_dir() {
        return Err(format!("{} exists and is not a directory", cli.output_dir.display()));
    }
    let jobs = discover(cli)?;
    if jobs.is_empty() {
        return Err("no .binpack files found".into());
    }
    for j in &jobs {
        let mark = if j.output.exists() && !cli.force { "  (exists, skipped)" } else { "" };
        eprintln!("{:>8.2} GB  {}\n         -> {}{mark}", gb(j.size), j.input.display(), j.output.display());
    }
    let total: u64 = jobs.iter().map(|j| j.size).sum();
    eprintln!("{} files, {:.1} GB", jobs.len(), gb(total));
    if cli.list {
        return Ok(true);
    }
    let todo: Vec<&Job> = jobs.iter().filter(|j| cli.force || !j.output.exists()).collect();
    if todo.is_empty() {
        eprintln!("all outputs exist; nothing to do (pass --force to redo them)");
        return Ok(true);
    }
    fs::create_dir_all(&cli.output_dir).map_err(|e| format!("{}: {e}", cli.output_dir.display()))?;

    let config = cli.skip_config();
    eprintln!("skip filter: {config:?}");
    let model = load_or_train_model(cli, &jobs, config)?;

    let todo_bytes: u64 = todo.iter().map(|j| j.size).sum();
    eprintln!(
        "converting {} files ({:.1} GB) on {} threads into {}",
        todo.len(),
        gb(todo_bytes),
        cli.threads,
        cli.output_dir.display()
    );
    let progress = Progress::default();
    let failures = Mutex::new(Vec::new());
    let totals = Mutex::new(Totals::default());
    let done = AtomicBool::new(false);
    let start = Instant::now();

    std::thread::scope(|s| {
        s.spawn(|| {
            let mut last = Instant::now();
            while !done.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(250));
                if last.elapsed() >= Duration::from_secs(30) {
                    last = Instant::now();
                    report_progress(&progress, todo.len(), todo_bytes, start);
                }
            }
        });

        for_each(todo.len(), cli.threads, |i| {
            let job = todo[i];
            if stopped() {
                return;
            }
            let t = Instant::now();
            let res = convert_one(job, config, &model, &progress);
            if res.is_err() && stopped() {
                eprintln!("stopped {} (unfinished output removed)", job.input.display());
                return;
            }
            let n = progress.files_done.fetch_add(1, Ordering::Relaxed) + 1;
            progress.bytes_done.fetch_add(job.size, Ordering::Relaxed);
            match res {
                Ok(r) => {
                    {
                        let mut tot = totals.lock().unwrap();
                        tot.read += r.stats.read;
                        tot.written += r.written;
                        tot.tree += r.stats.tree_samples;
                        tot.invalid += r.invalid;
                        tot.out_bytes += r.out_bytes;
                    }
                    eprintln!(
                        "[{n}/{}] {} -> {}: read {}, kept {} ({:.2}%){}{}, {:.2} GB, {:.2} B/pos, {:.0}s",
                        todo.len(),
                        job.input.display(),
                        job.output.display(),
                        r.stats.read,
                        r.written,
                        100.0 * r.written as f64 / r.stats.read.max(1) as f64,
                        if r.stats.tree_samples > 0 {
                            format!(", {} search-tree samples", r.stats.tree_samples)
                        } else {
                            String::new()
                        },
                        if r.invalid > 0 { format!(", {} unrepresentable", r.invalid) } else { String::new() },
                        gb(r.out_bytes),
                        r.out_bytes as f64 / r.written.max(1) as f64,
                        t.elapsed().as_secs_f64(),
                    );
                }
                Err(e) => {
                    eprintln!("[{n}/{}] FAILED {}: {e}", todo.len(), job.input.display());
                    failures.lock().unwrap().push(job.input.display().to_string());
                }
            }
        });
        done.store(true, Ordering::Relaxed);
    });

    let t = totals.into_inner().unwrap();
    let failures = failures.into_inner().unwrap();
    if stopped() {
        eprintln!(
            "interrupted after {:.0}s: finished files are complete, unfinished ones were removed; \
             rerun the same command to continue",
            start.elapsed().as_secs_f64()
        );
        return Ok(false);
    }
    eprintln!(
        "done in {:.0}s: read {} positions, wrote {} ({:.2}%) into {} files, {:.2} GB, {:.2} B/pos",
        start.elapsed().as_secs_f64(),
        t.read,
        t.written,
        100.0 * t.written as f64 / t.read.max(1) as f64,
        todo.len() - failures.len(),
        gb(t.out_bytes),
        t.out_bytes as f64 / t.written.max(1) as f64,
    );
    if t.tree > 0 {
        eprintln!("{} of the written positions are search-tree samples (outcome NoResult)", t.tree);
    }
    if t.invalid > 0 {
        eprintln!("skipped {} positions that FSQD cannot represent", t.invalid);
    }
    if !failures.is_empty() {
        eprintln!("{} files failed (rerun the same command to retry them):", failures.len());
        for f in &failures {
            eprintln!("  {f}");
        }
        return Ok(false);
    }
    eprintln!(
        "join them into one file with: find {} -name '*.fsqd' | sort | xargs cat > all.fsqd",
        cli.output_dir.display()
    );
    Ok(true)
}
