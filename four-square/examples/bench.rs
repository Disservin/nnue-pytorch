//! Measure FSQD size and single-thread decode speed on a binpack sample.
//!
//! Usage: bench <file.binpack> [train records] [test records]
//!
//! Reads the first `train + test` entries (no filtering), shuffles them,
//! trains on the first part and encodes the rest into memory.

use std::sync::Arc;
use std::time::Instant;

use four_square::{Dataset, Model, Record, Writer};

fn main() -> four_square::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).expect("usage: bench <file.binpack> [train] [test]");
    let n_train: usize = args.get(2).map_or(100_000, |s| s.parse().unwrap());
    let n_test: usize = args.get(3).map_or(100_000, |s| s.parse().unwrap());

    let mut reader = sfbinpack::CompressedTrainingDataEntryReader::new(std::fs::File::open(path)?)
        .map_err(|e| four_square::Error::InvalidFile(e.to_string()))?;
    let mut recs = Vec::with_capacity(n_train + n_test);
    while reader.has_next() && recs.len() < n_train + n_test {
        recs.push(Record::project(&reader.next())?);
    }
    // Deterministic Fisher-Yates shuffle so records come from many games.
    let mut s = 0x9E3779B97F4A7C15u64;
    for i in (1..recs.len()).rev() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        recs.swap(i, (s % (i as u64 + 1)) as usize);
    }
    let (train, test) = recs.split_at(n_train.min(recs.len()));

    let t = Instant::now();
    let model = Arc::new(Model::train(|| Ok(train.iter().copied().map(Ok)))?);
    let train_s = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let mut w = Writer::new(Vec::new(), model.clone())?;
    for r in test {
        w.push(r)?;
    }
    let bytes = w.finish()?;
    let encode_s = t.elapsed().as_secs_f64();

    let model_len = model.to_bytes().len();
    let n = test.len() as f64;
    let ds = Dataset::open(bytes.as_slice())?;
    let mut worker = ds.worker();

    // Verify once, then time warm in-memory decoding.
    let mut decoded = Vec::with_capacity(test.len());
    ds.read_range(0..ds.records(), &mut worker, &mut decoded)?;
    assert!(decoded == test, "round trip mismatch");

    let mut rates = Vec::new();
    for _ in 0..7 {
        let t = Instant::now();
        let mut sink = 0u64;
        for id in 0..ds.blocks() {
            let block = ds.decode_block(id, &mut worker)?;
            sink = sink.wrapping_add(block.len() as u64 + block[0].labels.score as u64);
        }
        std::hint::black_box(sink);
        rates.push(n / t.elapsed().as_secs_f64() / 1e6);
    }
    rates.sort_by(|a, b| a.partial_cmp(b).unwrap());

    println!("train {} records in {:.2}s, encode {} in {:.2}s", train.len(), train_s, test.len(), encode_s);
    println!("model: {} bytes (one-time)", model_len);
    println!("whole file:        {:.3} bytes/record", bytes.len() as f64 / n);
    println!("payload + index:   {:.3} bytes/record", (bytes.len() - 16 - model_len - 24) as f64 / n);
    println!("decode (1 thread): {:.2} M records/s median of 7", rates[3]);
    Ok(())
}
