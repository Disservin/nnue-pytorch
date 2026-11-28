use std::collections::HashMap;

use crate::codec::{self, Sink};
use crate::error::{Error, Result};
use crate::rans::{self, DecodeTable};
use crate::record::{piece, Record};

pub(crate) const GROUPS: usize = 16;
pub(crate) const PHASES: usize = 8;
pub(crate) const MAX_DICT: usize = 127;
pub(crate) const ESCAPE: u32 = 127;

// Table families in model serialization order. Within a family, selectors
// are enumerated lexicographically with the leftmost selector slowest.
pub(crate) const T_WK: usize = 0; //                    global, 64
pub(crate) const T_BK: usize = T_WK + 1; //             white king rank, 64
pub(crate) const T_PHASE: usize = T_BK + 8; //          global, 8
pub(crate) const T_TOKEN: usize = T_PHASE + 1; //       phase x group, 128
pub(crate) const T_MASK: usize = T_TOKEN + PHASES * GROUPS; // phase x group, 16
pub(crate) const T_LIT: usize = T_MASK + PHASES * GROUPS; //   phase x file-major square, 12
pub(crate) const T_STATE: usize = T_LIT + PHASES * 64; //  global, 32
pub(crate) const T_EP: usize = T_STATE + 1; //          global, 65
pub(crate) const T_RESULT: usize = T_EP + 1; //         material context C, 4
pub(crate) const T_SHIGH: usize = T_RESULT + 11; //     C x result, 256
pub(crate) const T_SLOW: usize = T_SHIGH + 11 * RESULTS; // score high byte, 256
/// Result symbols: loss, draw, win, no result.
pub(crate) const RESULTS: usize = 4;
pub(crate) const N_TABLES: usize = T_SLOW + 256;

pub(crate) fn alphabet(table: usize) -> usize {
    match table {
        t if t < T_PHASE => 64,
        t if t < T_TOKEN => 8,
        t if t < T_MASK => 128,
        t if t < T_LIT => 16,
        t if t < T_STATE => 12,
        t if t < T_EP => 32,
        t if t < T_RESULT => 65,
        t if t < T_SHIGH => RESULTS,
        _ => 256,
    }
}

/// Signed material weight of each piece ID (white positive).
pub(crate) const MATERIAL: [i32; 13] = [1, -1, 3, -3, 3, -3, 5, -5, 9, -9, 0, 0, 0];

/// An immutable compiled model: 16 group dictionaries and every probability
/// table, plus the decode lookup tables built from them. Share it across
/// threads behind an `Arc`.
pub struct Model {
    dicts: [Vec<u32>; GROUPS],
    pub(crate) dict_index: [HashMap<u32, u32>; GROUPS],
    /// Per dictionary word: (signed material, non-empty square count).
    pub(crate) dict_info: [Vec<(i32, u32)>; GROUPS],
    /// Offset of each table in `freqs` / `starts`; `N_TABLES + 1` entries.
    sym_offset: Vec<usize>,
    freqs: Vec<u16>,
    starts: Vec<u16>,
    pub(crate) decode: Vec<Box<DecodeTable>>,
}

fn sym_offsets() -> Vec<usize> {
    let mut off = Vec::with_capacity(N_TABLES + 1);
    let mut acc = 0;
    for t in 0..N_TABLES {
        off.push(acc);
        acc += alphabet(t);
    }
    off.push(acc);
    off
}

/// Upper bound of a serialized model, from the grammar.
pub(crate) fn max_serialized_len() -> usize {
    GROUPS * (1 + 4 * MAX_DICT) + 2 * sym_offsets()[N_TABLES]
}

struct CountSink<'a> {
    offsets: &'a [usize],
    counts: &'a mut [u64],
    overflow: bool,
}

impl Sink for CountSink<'_> {
    #[inline]
    fn emit(&mut self, table: usize, sym: u32) {
        let c = &mut self.counts[self.offsets[table] + sym as usize];
        match c.checked_add(1) {
            Some(v) => *c = v,
            None => self.overflow = true,
        }
    }
}

impl Model {
    /// Build a model deterministically in two passes over the training
    /// records. `open` is called once per pass and must yield the same records
    /// both times. Pass one counts four-square words after removing kings and
    /// freezes the dictionaries; pass two counts every table's symbols.
    pub fn train<I>(mut open: impl FnMut() -> Result<I>) -> Result<Self>
    where
        I: Iterator<Item = Result<Record>>,
    {
        let mut words: [HashMap<u32, u64>; GROUPS] = Default::default();
        for rec in open()? {
            let rec = rec?;
            let board = codec::board_without_kings(&rec.board);
            for (g, map) in words.iter_mut().enumerate() {
                let c = map.entry(codec::group_word(&board, g)).or_insert(0);
                *c = c.checked_add(1).ok_or_else(|| Error::InvalidModel("word count overflow".into()))?;
            }
        }
        let dicts: [Vec<u32>; GROUPS] = std::array::from_fn(|g| {
            let mut v: Vec<(u32, u64)> = words[g].iter().map(|(&w, &c)| (w, c)).collect();
            v.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(b.0.cmp(&a.0)));
            v.into_iter().take(MAX_DICT).map(|(w, _)| w).collect()
        });

        // Pass two needs the dictionary lookup only; tables are still empty.
        let offsets = sym_offsets();
        let dict_index = build_index(&dicts);
        let mut counts = vec![0u64; offsets[N_TABLES]];
        let mut sink = CountSink { offsets: &offsets, counts: &mut counts, overflow: false };
        for rec in open()? {
            codec::walk(&dict_index, &rec?, &mut sink);
        }
        if sink.overflow {
            return Err(Error::InvalidModel("symbol count overflow".into()));
        }
        let mut freqs = Vec::with_capacity(counts.len());
        for t in 0..N_TABLES {
            freqs.extend(rans::normalize(&counts[offsets[t]..offsets[t + 1]]));
        }
        Self::from_parts(dicts, freqs)
    }

    fn from_parts(dicts: [Vec<u32>; GROUPS], freqs: Vec<u16>) -> Result<Self> {
        let offsets = sym_offsets();
        if freqs.len() != offsets[N_TABLES] {
            return Err(Error::InvalidModel("wrong number of frequencies".into()));
        }
        for (g, d) in dicts.iter().enumerate() {
            if d.len() > MAX_DICT {
                return Err(Error::InvalidModel(format!("dictionary {g} has {} words", d.len())));
            }
            for &w in d {
                if w.to_le_bytes().iter().any(|&p| p > 9 && p != piece::EMPTY) {
                    return Err(Error::InvalidModel(format!("dictionary {g} word {w:#010x} has a bad piece")));
                }
            }
        }
        let dict_index = build_index(&dicts);
        for (g, idx) in dict_index.iter().enumerate() {
            if idx.len() != dicts[g].len() {
                return Err(Error::InvalidModel(format!("dictionary {g} has duplicate words")));
            }
        }
        let mut starts = Vec::with_capacity(freqs.len());
        let mut decode = Vec::with_capacity(N_TABLES);
        for t in 0..N_TABLES {
            let f = &freqs[offsets[t]..offsets[t + 1]];
            if f.contains(&0) || f.iter().map(|&x| x as u32).sum::<u32>() != rans::PROB_SCALE {
                return Err(Error::InvalidModel(format!("table {t} is not positive with sum 1024")));
            }
            let mut acc = 0u16;
            for &x in f {
                starts.push(acc);
                acc += x;
            }
            decode.push(rans::build_decode_table(f));
        }
        let dict_info = std::array::from_fn(|g| {
            dicts[g]
                .iter()
                .map(|&w| {
                    let b = w.to_le_bytes();
                    let mat = b.iter().map(|&p| MATERIAL[p as usize]).sum();
                    let cnt = b.iter().filter(|&&p| p != piece::EMPTY).count() as u32;
                    (mat, cnt)
                })
                .collect()
        });
        Ok(Self { dicts, dict_index, dict_info, sym_offset: offsets, freqs, starts, decode })
    }

    #[inline(always)]
    pub(crate) fn symbol(&self, table: usize, sym: u32) -> (u16, u16) {
        let i = self.sym_offset[table] + sym as usize;
        (self.starts[i], self.freqs[i])
    }

    pub(crate) fn dict_len(&self, group: usize) -> u32 {
        self.dicts[group].len() as u32
    }

    pub(crate) fn dict_word(&self, group: usize, id: u32) -> u32 {
        self.dicts[group][id as usize]
    }

    /// Dictionaries, then every table's u16 frequencies, all little-endian.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        for d in &self.dicts {
            out.push(d.len() as u8);
            for &w in d {
                out.extend_from_slice(&w.to_le_bytes());
            }
        }
        for &f in &self.freqs {
            out.extend_from_slice(&f.to_le_bytes());
        }
        out
    }

    /// Parse and validate a serialized model. The input must be consumed exactly.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let short = || Error::InvalidModel("model data ends early".into());
        let mut pos = 0usize;
        let mut dicts: [Vec<u32>; GROUPS] = Default::default();
        for (g, d) in dicts.iter_mut().enumerate() {
            let len = *bytes.get(pos).ok_or_else(short)? as usize;
            pos += 1;
            if len > MAX_DICT {
                return Err(Error::InvalidModel(format!("dictionary {g} length {len} exceeds 127")));
            }
            for _ in 0..len {
                let w = bytes.get(pos..pos + 4).ok_or_else(short)?;
                d.push(u32::from_le_bytes(w.try_into().unwrap()));
                pos += 4;
            }
        }
        let n = sym_offsets()[N_TABLES];
        let raw = bytes.get(pos..pos + 2 * n).ok_or_else(short)?;
        if pos + 2 * n != bytes.len() {
            return Err(Error::InvalidModel("trailing bytes after model".into()));
        }
        let freqs = raw.as_chunks::<2>().0.iter().map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        Self::from_parts(dicts, freqs)
    }
}

fn build_index(dicts: &[Vec<u32>; GROUPS]) -> [HashMap<u32, u32>; GROUPS] {
    std::array::from_fn(|g| dicts[g].iter().enumerate().map(|(i, &w)| (w, i as u32)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout() {
        assert_eq!(T_TOKEN, 10);
        assert_eq!(T_LIT, 266);
        assert_eq!(T_SLOW, 835);
        assert_eq!(N_TABLES, 1091);
        assert_eq!(alphabet(T_EP), 65);
        assert_eq!(alphabet(T_RESULT + 10), 4);
        assert_eq!(alphabet(T_SHIGH), 256);
        assert_eq!(alphabet(T_LIT + 511), 12);
    }
}
