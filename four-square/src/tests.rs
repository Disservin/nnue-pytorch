use std::collections::HashMap;
use std::io::{self, Write};
use std::sync::Arc;

use crate::codec::{self, Sink};
use crate::model::*;
use crate::record::{castling, piece};
use crate::*;

// ---------------------------------------------------------------- fixtures

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// A random board: two kings, a random number of pieces biased towards pawns,
/// random side to move, castling, en passant and labels.
fn random_record(rng: &mut Rng) -> Record {
    let mut sq = [piece::EMPTY; 64];
    let wk = rng.below(64) as usize;
    let mut bk = rng.below(64) as usize;
    while bk == wk {
        bk = rng.below(64) as usize;
    }
    sq[wk] = piece::WHITE_KING;
    sq[bk] = piece::BLACK_KING;
    let count = rng.below(31);
    for _ in 0..count {
        let s = rng.below(64) as usize;
        if sq[s] != piece::EMPTY {
            continue;
        }
        let p = if rng.below(2) == 0 { rng.below(2) } else { rng.below(10) };
        sq[s] = p as u8;
    }
    let ep = if rng.below(4) == 0 { Some(rng.below(64) as u8) } else { None };
    let board = BoardState::new(sq, rng.below(2) == 1, rng.below(16) as u8, ep).unwrap();
    let score = match rng.below(8) {
        0 => rng.next() as i16,
        _ => (rng.below(1200) as i16) - 600,
    };
    let outcome = Outcome::from_symbol(rng.below(4) as u32);
    Record { board, labels: Labels { score, outcome } }
}

fn records(seed: u64, n: usize) -> Vec<Record> {
    let mut rng = Rng(seed);
    (0..n).map(|_| random_record(&mut rng)).collect()
}

fn board(pieces: &[(usize, u8)], black: bool, castle: u8, ep: Option<u8>) -> BoardState {
    let mut sq = [piece::EMPTY; 64];
    for &(s, p) in pieces {
        sq[s] = p;
    }
    BoardState::new(sq, black, castle, ep).unwrap()
}

/// Records that exercise the edge cases listed in codec.md.
fn special_records() -> Vec<Record> {
    let mut v = Vec::new();
    let lab = |score, r| Labels { score, outcome: Outcome::from_result(r).unwrap() };
    // Extreme scores, every result, both sides to move.
    for (i, &score) in [-32768i16, -1, 0, 32767].iter().enumerate() {
        let b = board(&[(4, piece::WHITE_KING), (60, piece::BLACK_KING)], i % 2 == 1, 0, None);
        v.push(Record { board: b, labels: lab(score, i as i16 % 3 - 1) });
    }
    // Every castling flag alone and all together.
    for c in [
        castling::WHITE_KING_SIDE,
        castling::WHITE_QUEEN_SIDE,
        castling::BLACK_KING_SIDE,
        castling::BLACK_QUEEN_SIDE,
        15,
    ] {
        let b = board(
            &[(4, piece::WHITE_KING), (60, piece::BLACK_KING), (0, piece::WHITE_ROOK), (63, piece::BLACK_ROOK)],
            false,
            c,
            None,
        );
        v.push(Record { board: b, labels: lab(10, 0) });
    }
    // EP present on both sides.
    for (black, ep) in [(false, 43u8), (true, 20)] {
        let b = board(
            &[(4, piece::WHITE_KING), (60, piece::BLACK_KING), (35, piece::WHITE_PAWN), (36, piece::BLACK_PAWN)],
            black,
            0,
            Some(ep),
        );
        v.push(Record { board: b, labels: lab(-5, 1) });
    }
    // Promoted material: four white queens, three black knights.
    let b = board(
        &[(6, piece::WHITE_KING), (62, piece::BLACK_KING), (0, 8), (9, 8), (18, 8), (27, 8), (40, 3), (49, 3), (58, 3)],
        true,
        0,
        None,
    );
    v.push(Record { board: b, labels: lab(2000, -1) });
    // Group 0 (a1..a4) with pieces at offsets 0 and 3 only: mask 9 on escape.
    let b = board(
        &[(0, piece::WHITE_QUEEN), (24, piece::BLACK_QUEEN), (7, piece::WHITE_KING), (63, piece::BLACK_KING)],
        false,
        0,
        None,
    );
    v.push(Record { board: b, labels: lab(0, 0) });
    // A search-tree sample: score only, no game result.
    let b = board(&[(6, piece::WHITE_KING), (62, piece::BLACK_KING), (27, piece::WHITE_ROOK)], false, 0, None);
    v.push(Record { board: b, labels: Labels { score: 553, outcome: Outcome::NoResult } });
    v
}

fn train(recs: &[Record]) -> Arc<Model> {
    Arc::new(Model::train(|| Ok(recs.iter().copied().map(Ok))).unwrap())
}

fn write(model: &Arc<Model>, recs: &[Record]) -> Vec<u8> {
    let mut w = Writer::new(Vec::new(), model.clone()).unwrap();
    for r in recs {
        w.push(r).unwrap();
    }
    w.finish().unwrap()
}

fn read_all(bytes: Vec<u8>) -> Result<Vec<Record>> {
    let ds = Dataset::open_with(bytes, OpenOptions { index_validation: IndexValidation::Full })?;
    let mut out = Vec::new();
    for b in ds.iter_blocks() {
        out.extend(b?);
    }
    Ok(out)
}

// ---------------------------------------------------------------- codec

#[derive(Default)]
struct Trace {
    hits: [u64; GROUPS],
    escapes: [u64; GROUPS],
    masks: HashMap<u32, u64>,
    syms: Vec<(usize, u32)>,
}

impl Sink for Trace {
    fn emit(&mut self, table: usize, sym: u32) {
        self.syms.push((table, sym));
        if (T_TOKEN..T_MASK).contains(&table) {
            let g = (table - T_TOKEN) % GROUPS;
            if sym == ESCAPE {
                self.escapes[g] += 1;
            } else {
                self.hits[g] += 1;
            }
        }
        if (T_MASK..T_LIT).contains(&table) {
            *self.masks.entry(sym).or_default() += 1;
        }
    }
}

#[test]
fn round_trip_with_full_coverage() {
    let model = train(&records(1, 4000));
    let mut test = records(2, 1500);
    test.extend(special_records());
    test.extend(records(3, 37)); // final partial block
    let mut trace = Trace::default();
    for r in &test {
        codec::walk(&model.dict_index, r, &mut trace);
    }
    for g in 0..GROUPS {
        assert!(
            trace.hits[g] > 0 && trace.escapes[g] > 0,
            "group {g}: {} hits, {} escapes",
            trace.hits[g],
            trace.escapes[g]
        );
    }
    assert!(trace.masks.contains_key(&9), "mask with bits 0 and 3");

    let bytes = write(&model, &test);
    assert_eq!(read_all(bytes).unwrap(), test);
}

#[test]
fn spec_examples() {
    // Group 0 = rook, empty, black pawn, empty -> word 0x0C010C06.
    let b = board(
        &[(0, piece::WHITE_ROOK), (16, piece::BLACK_PAWN), (7, piece::WHITE_KING), (63, piece::BLACK_KING)],
        false,
        0,
        None,
    );
    let sq = codec::board_without_kings(&b);
    assert_eq!(codec::group_word(&sq, 0), 0x0C010C06);
    assert_eq!(codec::group_square(1, 0), 32); // a5
    assert_eq!(codec::group_square(15, 3), 63); // h8

    // With an empty dictionary that word is a miss: 127, mask 5, then 6 and 1.
    let model = train(&[]);
    let rec = Record { board: b, labels: Labels { score: 0, outcome: Outcome::Draw } };
    let mut trace = Trace::default();
    codec::walk(&model.dict_index, &rec, &mut trace);
    let phase = trace.syms[2].1 as usize;
    let g0 = trace.syms.iter().position(|&(t, _)| t == T_TOKEN + phase * GROUPS).unwrap();
    assert_eq!(&trace.syms[g0..g0 + 4], &[(T_TOKEN, ESCAPE), (T_MASK, 5), (T_LIT, 6), (T_LIT + 2, 1)]);
}

#[test]
fn small_files() {
    let model = train(&records(5, 300));
    assert_eq!(read_all(write(&model, &[])).unwrap(), vec![]);
    let one = records(6, 1);
    assert_eq!(read_all(write(&model, &one)).unwrap(), one);
    let exact = records(7, BLOCK_RECORDS * 2);
    assert_eq!(read_all(write(&model, &exact)).unwrap(), exact);
    // A model trained on nothing still codes everything.
    let empty = train(&[]);
    let recs = records(8, 600);
    assert_eq!(read_all(write(&empty, &recs)).unwrap(), recs);
}

#[test]
fn read_range_trims_blocks() {
    let model = train(&records(9, 500));
    let recs = records(10, 1000);
    let ds = Dataset::open(write(&model, &recs)).unwrap();
    assert_eq!((ds.records(), ds.blocks()), (1000, 4));
    let mut w = ds.worker();
    for (a, b) in [(0u64, 1u64), (255, 257), (10, 990), (768, 1000), (500, 500)] {
        let mut out = Vec::new();
        ds.read_range(a..b, &mut w, &mut out).unwrap();
        assert_eq!(out, recs[a as usize..b as usize]);
    }
    let mut out = Vec::new();
    assert!(matches!(ds.read_range(999..1001, &mut w, &mut out), Err(Error::OutOfRange { .. })));
    assert!(out.is_empty());
    assert!(matches!(ds.decode_block(4, &mut w), Err(Error::OutOfRange { id: 4, len: 4 })));
}

#[test]
fn training_is_deterministic() {
    // HashMap iteration order differs between instances; output must not.
    let recs = records(11, 3000);
    let a = write(&train(&recs), &recs);
    let b = write(&train(&recs), &recs);
    assert_eq!(a, b);
}

#[test]
fn golden_file() {
    let recs = records(12, 700);
    let bytes = write(&train(&recs), &recs);
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden/synthetic-700.fsqd");
    if std::env::var_os("FSQD_BLESS").is_some() {
        std::fs::create_dir_all(std::path::Path::new(path).parent().unwrap()).unwrap();
        std::fs::write(path, &bytes).unwrap();
    }
    let golden = std::fs::read(path).expect("golden file missing; run with FSQD_BLESS=1 to create it");
    assert_eq!(bytes, golden, "encoder output changed; re-bless only for an intended format change");
    assert_eq!(read_all(golden).unwrap(), recs);
}

// ---------------------------------------------------------------- model

#[test]
fn model_serialization() {
    let model = train(&records(13, 1000));
    let bytes = model.to_bytes();
    assert!(bytes.len() <= max_serialized_len());
    assert_eq!(Model::from_bytes(&bytes).unwrap().to_bytes(), bytes);

    let dict_bytes: usize = (0..GROUPS).map(|g| 1 + 4 * model.dict_len(g) as usize).sum();
    // A zero frequency in the first table.
    let mut bad = bytes.clone();
    let f0 = u16::from_le_bytes([bad[dict_bytes], bad[dict_bytes + 1]]);
    bad[dict_bytes..dict_bytes + 2].copy_from_slice(&0u16.to_le_bytes());
    bad[dict_bytes + 2..dict_bytes + 4]
        .copy_from_slice(&(u16::from_le_bytes([bytes[dict_bytes + 2], bytes[dict_bytes + 3]]) + f0).to_le_bytes());
    assert!(matches!(Model::from_bytes(&bad), Err(Error::InvalidModel(_))));
    // Sum not 1024.
    let mut bad = bytes.clone();
    bad[dict_bytes] = bad[dict_bytes].wrapping_add(1);
    assert!(matches!(Model::from_bytes(&bad), Err(Error::InvalidModel(_))));
    // Truncated and over-long.
    assert!(Model::from_bytes(&bytes[..bytes.len() - 1]).is_err());
    let mut long = bytes.clone();
    long.push(0);
    assert!(Model::from_bytes(&long).is_err());
    // Dictionary length above 127.
    let mut bad = bytes.clone();
    bad[0] = 128;
    assert!(matches!(Model::from_bytes(&bad), Err(Error::InvalidModel(_))));
    // Duplicate word and invalid piece byte in dictionary 0.
    assert!(model.dict_len(0) >= 2);
    let mut bad = bytes.clone();
    let w0: [u8; 4] = bad[1..5].try_into().unwrap();
    bad[5..9].copy_from_slice(&w0);
    assert!(matches!(Model::from_bytes(&bad), Err(Error::InvalidModel(_))));
    let mut bad = bytes.clone();
    bad[1] = 10; // a king inside a dictionary word
    assert!(matches!(Model::from_bytes(&bad), Err(Error::InvalidModel(_))));
}

// ---------------------------------------------------------------- container

struct Layout {
    model_end: usize,
    offsets: Vec<u64>,
    index_start: usize,
}

fn layout(bytes: &[u8]) -> Layout {
    let model_len = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
    let n = bytes.len();
    let records = u64::from_le_bytes(bytes[n - 24..n - 16].try_into().unwrap());
    let index_start = u64::from_le_bytes(bytes[n - 16..n - 8].try_into().unwrap()) as usize;
    let blocks = records.div_ceil(256) as usize;
    let offsets = (0..blocks)
        .map(|j| u64::from_le_bytes(bytes[index_start + 8 * j..index_start + 8 * j + 8].try_into().unwrap()))
        .collect();
    Layout { model_end: 16 + model_len, offsets, index_start }
}

fn set_u64(bytes: &mut [u8], at: usize, v: u64) {
    bytes[at..at + 8].copy_from_slice(&v.to_le_bytes());
}

#[test]
fn rejects_bad_headers_and_footers() {
    let model = train(&records(14, 300));
    let good = write(&model, &records(15, 600));
    let n = good.len();
    assert!(read_all(good.clone()).is_ok());
    let mutate = |at: usize, v: u8| {
        let mut b = good.clone();
        b[at] = v;
        b
    };
    assert!(matches!(read_all(mutate(0, b'X')), Err(Error::InvalidFile(_))));
    assert!(matches!(read_all(mutate(4, 2)), Err(Error::InvalidFile(_))));
    assert!(matches!(read_all(mutate(6, 1)), Err(Error::InvalidFile(_))));
    assert!(matches!(read_all(mutate(n - 8, 1)), Err(Error::InvalidFile(_)))); // reserved
    assert!(matches!(read_all(mutate(n - 1, b'X')), Err(Error::InvalidFile(_))));
    assert!(matches!(read_all(mutate(n - 24, 0)), Err(Error::InvalidFile(_)))); // record count
    assert!(read_all(good[..n - 1].to_vec()).is_err());
    assert!(read_all(good[..30].to_vec()).is_err());
    // Model length pointing past the file.
    let mut b = good.clone();
    set_u64(&mut b, 8, n as u64);
    assert!(matches!(read_all(b), Err(Error::InvalidFile(_))));
}

#[test]
fn rejects_bad_index_and_payloads() {
    let model = train(&records(16, 300));
    let recs = records(17, 600); // 3 blocks
    let good = write(&model, &recs);
    let l = layout(&good);

    // Out-of-order offsets: full validation fails at open, on-demand at read.
    // Block 0's span is always checked at open, so break block 1's end.
    let mut b = good.clone();
    set_u64(&mut b, l.index_start + 16, l.offsets[0]);
    assert!(Dataset::open_with(b.clone(), OpenOptions { index_validation: IndexValidation::Full }).is_err());
    let ds = Dataset::open(b).unwrap();
    let mut w = ds.worker();
    assert!(ds.decode_block(0, &mut w).is_ok());
    assert!(matches!(ds.decode_block(1, &mut w), Err(Error::InvalidFile(_))));
    assert!(w.records.is_empty());

    // First block not directly after the model.
    let mut b = good.clone();
    set_u64(&mut b, l.index_start, l.model_end as u64 + 1);
    assert!(Dataset::open(b).is_err());

    // Clip one byte from the last payload and shift index/footer accordingly.
    let mut b = good[..l.index_start - 1].to_vec();
    b.extend_from_slice(&good[l.index_start..]);
    let n = b.len();
    set_u64(&mut b, n - 16, (l.index_start - 1) as u64);
    let ds = Dataset::open(b).unwrap();
    let mut w = ds.worker();
    assert!(ds.decode_block(1, &mut w).is_ok());
    let err = ds.decode_block(2, &mut w).unwrap_err();
    assert!(matches!(err, Error::Truncated { block: Some(2), .. } | Error::Corrupt { block: 2, .. }), "{err}");

    // Corrupt a payload byte: decoding must fail or, if it happens to stay in
    // range, must not report success with the original records silently changed
    // into an unchecked state. Every failure leaves the worker empty.
    let mut failures = 0;
    for k in 0..40 {
        let mut b = good.clone();
        let at = l.offsets[1] as usize + 4 + k * 7;
        b[at] ^= 0x5a;
        let ds = Dataset::open(b).unwrap();
        let mut w = ds.worker();
        match ds.decode_block(1, &mut w) {
            Err(_) => {
                failures += 1;
                assert!(w.records.is_empty());
            }
            Ok(r) => assert_ne!(r, &recs[256..512]),
        }
    }
    assert!(failures > 30, "only {failures} of 40 corruptions detected");
}

struct FailingSink {
    budget: usize,
}

impl Write for FailingSink {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if buf.len() > self.budget {
            return Err(io::Error::other("disk full"));
        }
        self.budget -= buf.len();
        Ok(buf.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[test]
fn failed_write_poisons_writer() {
    let model = train(&records(18, 300));
    let budget = 16 + model.to_bytes().len() + 10;
    let mut w = Writer::new(FailingSink { budget }, model).unwrap();
    let recs = records(19, BLOCK_RECORDS + 1);
    let mut first_err = None;
    for r in &recs {
        if let Err(e) = w.push(r) {
            first_err = Some(e);
            break;
        }
    }
    assert!(matches!(first_err, Some(Error::Io(_))));
    assert!(matches!(w.push(&recs[0]), Err(Error::WriterPoisoned)));
    assert!(matches!(w.finish(), Err(Error::WriterPoisoned)));
}

#[test]
fn file_source() {
    let model = train(&records(20, 300));
    let recs = records(21, 700);
    let path = std::env::temp_dir().join(format!("fsqd-test-{}.fsqd", std::process::id()));
    let f = std::fs::File::create(&path).unwrap();
    Writer::new(io::BufWriter::new(f), model).unwrap().finish_with(&recs);
    let ds = Dataset::open(std::fs::File::open(&path).unwrap()).unwrap();
    let mut out = Vec::new();
    ds.read_range(0..ds.records(), &mut ds.worker(), &mut out).unwrap();
    std::fs::remove_file(&path).unwrap();
    assert_eq!(out, recs);
}

trait FinishWith {
    fn finish_with(self, recs: &[Record]);
}

impl<W: Write> FinishWith for Writer<W> {
    fn finish_with(mut self, recs: &[Record]) {
        for r in recs {
            self.push(r).unwrap();
        }
        self.finish().unwrap();
    }
}

// ---------------------------------------------------------------- projection

fn check_projection(e: &sfbinpack::TrainingDataEntry) {
    use sfbinpack::chess::castling_rights::CastlingRights;
    use sfbinpack::chess::color::Color;
    use sfbinpack::chess::coords::Square;
    let r = Record::project(e).unwrap();
    for i in 0..64 {
        assert_eq!(r.board.piece_at(i), e.pos.piece_at(Square::new(i as u32)).id());
    }
    assert_eq!(r.board.black_to_move(), e.pos.side_to_move() == Color::Black);
    let cr = e.pos.castling_rights();
    assert_eq!(r.board.castling() & castling::WHITE_KING_SIDE != 0, cr.contains(CastlingRights::WHITE_KING_SIDE));
    assert_eq!(r.board.castling() & castling::WHITE_QUEEN_SIDE != 0, cr.contains(CastlingRights::WHITE_QUEEN_SIDE));
    assert_eq!(r.board.castling() & castling::BLACK_KING_SIDE != 0, cr.contains(CastlingRights::BLACK_KING_SIDE));
    assert_eq!(r.board.castling() & castling::BLACK_QUEEN_SIDE != 0, cr.contains(CastlingRights::BLACK_QUEEN_SIDE));
    let ep = e.pos.ep_square();
    assert_eq!(r.board.ep_square(), (ep != Square::NONE).then(|| ep.index() as u8));
    assert_eq!(r.labels.score, e.score);
    if e.ply == TREE_SAMPLE_PLY {
        assert_eq!(r.labels.outcome, Outcome::NoResult);
    } else {
        assert_eq!(r.labels.outcome.as_result(), Some(e.result));
    }
}

#[test]
fn projection_from_fens() {
    use sfbinpack::chess::position::Position;
    for (fen, score, ply, result) in [
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 25, 10, 0),
        ("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3", -40, 10, 1),
        ("r3k2r/8/8/8/8/8/8/R3K2R b Kq - 4 20", 32767, 10, -1),
        ("8/8/8/4k3/8/8/2QQQ3/4K3 b - - 0 70", -32768, 10, -1),
        // Search-tree sample as stored in sf-searchtree binpacks: placeholder result 0.
        ("2kr3r/1pqnnp2/p3p2p/b2pP1pP/P1pP4/2P5/1PQNNPP1/R1B1R1K1 w - g6 0 5000", 553, TREE_SAMPLE_PLY, 0),
    ] {
        let pos = Position::from_fen(fen).unwrap();
        let e = sfbinpack::TrainingDataEntry { pos, mv: Default::default(), score, ply, result };
        check_projection(&e);
    }
    let tree = sfbinpack::TrainingDataEntry {
        pos: Position::from_fen("8/8/8/4k3/8/8/8/4K3 w - - 0 1").unwrap(),
        mv: Default::default(),
        score: -111,
        ply: TREE_SAMPLE_PLY,
        result: 0,
    };
    assert_eq!(Record::project(&tree).unwrap().labels.outcome, Outcome::NoResult);
    let bad = sfbinpack::TrainingDataEntry {
        pos: Position::from_fen("8/8/8/4k3/8/8/8/4K3 w - - 0 1").unwrap(),
        mv: Default::default(),
        score: 0,
        ply: 0,
        result: 2,
    };
    assert!(matches!(Record::project(&bad), Err(Error::InvalidRecord(_))));
}

/// Projects every entry of the repository's PGO sample binpack, compares each
/// field with the reader and round-trips the projected records.
#[test]
fn projection_matches_reader_on_real_data() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../.pgo/small.binpack");
    let Ok(file) = std::fs::File::open(path) else {
        eprintln!("skipping: {path} not found");
        return;
    };
    let mut reader = sfbinpack::CompressedTrainingDataEntryReader::new(file).unwrap();
    let mut recs = Vec::new();
    while reader.has_next() && recs.len() < 200_000 {
        let e = reader.next();
        check_projection(&e);
        recs.push(Record::project(&e).unwrap());
    }
    assert!(!recs.is_empty());
    let (train_set, test_set) = recs.split_at(recs.len() / 2);
    let model = train(train_set);
    assert_eq!(read_all(write(&model, test_set)).unwrap(), test_set);
}

// ---------------------------------------------------------------- segments

#[test]
fn concatenated_files_read_as_one() {
    let m1 = train(&records(30, 500));
    let m2 = train(&records(31, 500));
    let a = records(32, 300); // short last block in the middle of the file
    let b = records(33, 0); // empty segment
    let c = records(34, 600);
    let mut file = write(&m1, &a);
    file.extend(write(&m2, &b));
    file.extend(write(&m1, &c));

    let ds = Dataset::open_with(file.clone(), OpenOptions { index_validation: IndexValidation::Full }).unwrap();
    assert_eq!(ds.segments().len(), 3);
    assert_eq!((ds.records(), ds.blocks()), (900, 5));
    let lens: Vec<u64> = (0..ds.blocks()).map(|i| ds.block_len(i).unwrap()).collect();
    assert_eq!(lens, [256, 44, 256, 256, 88]);
    // Byte-identical models are decoded once and shared.
    let s = ds.segments();
    assert!(Arc::ptr_eq(s[0].model(), s[2].model()));
    assert!(!Arc::ptr_eq(s[0].model(), s[1].model()));
    assert_eq!((s[1].start(), s[2].first_record(), s[2].first_block_id()), (write(&m1, &a).len() as u64, 300, 2));

    let all: Vec<Record> = a.iter().chain(&c).copied().collect();
    assert_eq!(read_all(file.clone()).unwrap(), all);
    let mut w = ds.worker();
    for (lo, hi) in [(0u64, 900u64), (250, 310), (299, 301), (300, 556), (890, 900), (5, 5)] {
        let mut out = Vec::new();
        ds.read_range(lo..hi, &mut w, &mut out).unwrap();
        assert_eq!(out, all[lo as usize..hi as usize], "range {lo}..{hi}");
    }
    assert!(matches!(ds.decode_block(5, &mut w), Err(Error::OutOfRange { id: 5, len: 5 })));
}

#[test]
fn append_mode_adds_a_segment() {
    let model = train(&records(35, 300));
    let a = records(36, 400);
    let b = records(37, 123);
    let path = std::env::temp_dir().join(format!("fsqd-append-{}.fsqd", std::process::id()));
    Writer::new(fs_create(&path), model.clone()).unwrap().finish_with(&a);
    let f = std::fs::OpenOptions::new().append(true).open(&path).unwrap();
    Writer::new(io::BufWriter::new(f), model).unwrap().finish_with(&b);
    let bytes = std::fs::read(&path).unwrap();
    std::fs::remove_file(&path).unwrap();
    let all: Vec<Record> = a.iter().chain(&b).copied().collect();
    assert_eq!(read_all(bytes).unwrap(), all);
}

fn fs_create(path: &std::path::Path) -> io::BufWriter<std::fs::File> {
    io::BufWriter::new(std::fs::File::create(path).unwrap())
}

#[test]
fn broken_middle_segment_is_rejected() {
    let model = train(&records(38, 300));
    let first = write(&model, &records(39, 300));
    let second = write(&model, &records(40, 300));
    let third = write(&model, &records(41, 300));
    let join = |a: &[u8], b: &[u8], c: &[u8]| [a, b, c].concat();

    // Damaged footer magic of the middle segment.
    let mut mid = second.clone();
    let n = mid.len();
    mid[n - 1] = b'X';
    assert!(matches!(Dataset::open(join(&first, &mid, &third)), Err(Error::InvalidFile(_))));

    // Middle segment's record count changed so its block count changes: the
    // computed start no longer lands on a header.
    let set_count = |v: u64| {
        let mut mid = second.clone();
        mid[n - 24..n - 16].copy_from_slice(&v.to_le_bytes());
        join(&first, &mid, &third)
    };
    assert!(matches!(Dataset::open(set_count(300 + 256)), Err(Error::InvalidFile(_))));
    // Same block count: the segment still lines up, and decoding its short
    // last block fails instead.
    let ds = Dataset::open(set_count(301)).unwrap();
    let mut w = ds.worker();
    assert!(ds.decode_block(2, &mut w).is_ok());
    assert!(matches!(ds.decode_block(3, &mut w), Err(Error::Corrupt { block: 3, .. } | Error::Truncated { .. })));

    // Junk between segments.
    let mut junk = first.clone();
    junk.extend_from_slice(&[0u8; 10]);
    assert!(Dataset::open(join(&junk, &second, &third)).is_err());

    // A truncated first segment.
    assert!(Dataset::open(join(&first[..first.len() - 3], &second, &third)).is_err());

    assert_eq!(Dataset::open(join(&first, &second, &third)).unwrap().records(), 900);
}

#[test]
fn encoded_blocks_match_pushed_records() {
    let recs = records(31, 1000);
    let model = train(&recs);
    let mut enc = BlockEncoder::new(model.clone());
    let mut w = Writer::new(Vec::new(), model.clone()).unwrap();
    for chunk in recs.chunks(BLOCK_RECORDS) {
        let block = enc.encode(chunk).unwrap();
        assert_eq!(block.records(), chunk.len());
        w.push_encoded(&block).unwrap();
    }
    assert_eq!(w.records(), 1000);
    assert_eq!(w.finish().unwrap(), write(&model, &recs));

    // A model with the same bytes is accepted.
    let copy = Arc::new(Model::from_bytes(&model.to_bytes()).unwrap());
    let mut w = Writer::new(Vec::new(), copy).unwrap();
    w.push_encoded(&enc.encode(&recs[..256]).unwrap()).unwrap();
}

#[test]
fn encoded_block_misuse_is_rejected() {
    let recs = records(32, 600);
    let model = train(&recs);
    let mut enc = BlockEncoder::new(model.clone());
    assert!(enc.encode(&[]).is_err());
    assert!(enc.encode(&recs[..257]).is_err());

    // Only the last block may be short.
    let mut w = Writer::new(Vec::new(), model.clone()).unwrap();
    w.push_encoded(&enc.encode(&recs[..100]).unwrap()).unwrap();
    assert!(w.push_encoded(&enc.encode(&recs[100..356]).unwrap()).is_err());

    // Pending pushed records.
    let mut w = Writer::new(Vec::new(), model.clone()).unwrap();
    w.push(&recs[0]).unwrap();
    assert!(w.push_encoded(&enc.encode(&recs[1..257]).unwrap()).is_err());

    // Another model.
    let other = train(&records(33, 300));
    let mut w = Writer::new(Vec::new(), other).unwrap();
    assert!(matches!(w.push_encoded(&enc.encode(&recs[..256]).unwrap()), Err(Error::InvalidModel(_))));
}
