//! Record symbol walk (shared by training and encoding) and block decoding.

use std::collections::HashMap;

use crate::error::{Error, Result};
use crate::model::*;
use crate::rans::{self, Decoder};
use crate::record::{piece, BoardState, Labels, Outcome, Record};

pub(crate) trait Sink {
    fn emit(&mut self, table: usize, sym: u32);
}

/// Board squares of group `g`, lowest rank first. Group `2f` covers ranks
/// 1-4 of file `f`, group `2f + 1` ranks 5-8.
#[inline(always)]
pub(crate) fn group_square(g: usize, k: usize) -> usize {
    8 * ((g & 1) * 4 + k) + (g >> 1)
}

/// The four piece IDs of group `g`, first square in the low byte.
#[inline(always)]
pub(crate) fn group_word(board: &[u8; 64], g: usize) -> u32 {
    (0..4).fold(0u32, |w, k| w | (board[group_square(g, k)] as u32) << (8 * k))
}

pub(crate) fn board_without_kings(b: &BoardState) -> [u8; 64] {
    let mut sq = *b.squares();
    sq[b.white_king() as usize] = piece::EMPTY;
    sq[b.black_king() as usize] = piece::EMPTY;
    sq
}

#[inline(always)]
fn material_context(material: i32, black_to_move: bool) -> usize {
    let m = if black_to_move { -material } else { material };
    (m.clamp(-5, 5) + 5) as usize
}

/// Emit the record's symbols in decoder order.
pub(crate) fn walk<S: Sink>(dicts: &[HashMap<u32, u32>; GROUPS], rec: &Record, sink: &mut S) {
    let b = &rec.board;
    let wk = b.white_king() as usize;
    sink.emit(T_WK, wk as u32);
    sink.emit(T_BK + wk / 8, b.black_king() as u32);
    let board = board_without_kings(b);
    let nonking = board.iter().filter(|&&p| p != piece::EMPTY).count();
    let phase = (nonking / 8).min(7);
    sink.emit(T_PHASE, phase as u32);
    let mut material = 0;
    for (g, dict) in dicts.iter().enumerate() {
        let word = group_word(&board, g);
        let table = phase * GROUPS + g;
        if let Some(&id) = dict.get(&word) {
            sink.emit(T_TOKEN + table, id);
        } else {
            sink.emit(T_TOKEN + table, ESCAPE);
            let bytes = word.to_le_bytes();
            let mask = (0..4).fold(0, |m, k| m | ((bytes[k] != piece::EMPTY) as u32) << k);
            sink.emit(T_MASK + table, mask);
            for (k, &p) in bytes.iter().enumerate() {
                if p != piece::EMPTY {
                    sink.emit(T_LIT + phase * 64 + 4 * g + k, p as u32);
                }
            }
        }
        material += word.to_le_bytes().iter().map(|&p| MATERIAL[p as usize]).sum::<i32>();
    }
    sink.emit(T_STATE, b.black_to_move() as u32 | (b.castling() as u32) << 1);
    sink.emit(T_EP, b.ep_raw() as u32);
    let c = material_context(material, b.black_to_move());
    let r = rec.labels.outcome as usize;
    sink.emit(T_RESULT + c, r as u32);
    let s = rec.labels.score as u16;
    sink.emit(T_SHIGH + c * RESULTS + r, (s >> 8) as u32);
    sink.emit(T_SLOW + (s >> 8) as usize, (s & 0xff) as u32);
}

struct EncodeSink<'a> {
    model: &'a Model,
    out: &'a mut Vec<(u16, u16)>,
}

impl Sink for EncodeSink<'_> {
    #[inline(always)]
    fn emit(&mut self, table: usize, sym: u32) {
        self.out.push(self.model.symbol(table, sym));
    }
}

/// Reusable encoder buffers.
#[derive(Default)]
pub(crate) struct BlockScratch {
    symbols: Vec<(u16, u16)>,
    stack: Vec<u8>,
}

/// Append the payload of one block (state + refill bytes) to `out`.
pub(crate) fn encode_block(model: &Model, records: &[Record], scratch: &mut BlockScratch, out: &mut Vec<u8>) {
    scratch.symbols.clear();
    let mut sink = EncodeSink { model, out: &mut scratch.symbols };
    for r in records {
        walk(&model.dict_index, r, &mut sink);
    }
    rans::encode(&scratch.symbols, &mut scratch.stack, out);
}

/// Decode `n` records from one block payload into `out` (cleared first).
/// On error `out` is left empty.
pub(crate) fn decode_block(model: &Model, payload: &[u8], n: usize, block: u64, out: &mut Vec<Record>) -> Result<()> {
    out.clear();
    let res = decode_into(model, payload, n, block, out);
    if res.is_err() {
        out.clear();
    }
    res
}

fn decode_into(model: &Model, payload: &[u8], n: usize, block: u64, out: &mut Vec<Record>) -> Result<()> {
    let mut d = match Decoder::new(payload) {
        None => return Err(Error::Truncated { block: Some(block), record: None }),
        Some(Err(())) => return Err(Error::Corrupt { block, record: None, reason: "initial state out of range" }),
        Some(Ok(d)) => d,
    };
    let dec = &model.decode;
    out.reserve(n);
    for i in 0..n {
        let rec = i as u32;
        let corrupt = |reason| Error::Corrupt { block, record: Some(rec), reason };
        let truncated = || Error::Truncated { block: Some(block), record: Some(rec) };
        macro_rules! get {
            ($t:expr) => {
                d.get(&dec[$t]).ok_or_else(truncated)?
            };
        }

        let wk = get!(T_WK) as usize;
        let bk = get!(T_BK + wk / 8) as usize;
        if wk == bk {
            return Err(corrupt("king collision"));
        }
        let phase = get!(T_PHASE) as usize;
        let mut sq = [piece::EMPTY; 64];
        let mut material = 0i32;
        let mut nonking = 0u32;
        for g in 0..GROUPS {
            let table = phase * GROUPS + g;
            let tok = get!(T_TOKEN + table);
            if tok == ESCAPE {
                let mask = get!(T_MASK + table);
                for k in 0..4 {
                    if mask >> k & 1 != 0 {
                        let p = get!(T_LIT + phase * 64 + 4 * g + k);
                        if p >= 10 {
                            return Err(corrupt("king literal"));
                        }
                        sq[group_square(g, k)] = p as u8;
                        material += MATERIAL[p as usize];
                        nonking += 1;
                    }
                }
            } else {
                if tok >= model.dict_len(g) {
                    return Err(corrupt("missing dictionary id"));
                }
                let w = model.dict_word(g, tok).to_le_bytes();
                for k in 0..4 {
                    sq[group_square(g, k)] = w[k];
                }
                let (m, c) = model.dict_info[g][tok as usize];
                material += m;
                nonking += c;
            }
        }
        if sq[wk] != piece::EMPTY || sq[bk] != piece::EMPTY {
            return Err(corrupt("occupied king destination"));
        }
        if (nonking as usize / 8).min(7) != phase {
            return Err(corrupt("phase does not match the board"));
        }
        sq[wk] = piece::WHITE_KING;
        sq[bk] = piece::BLACK_KING;

        let state = get!(T_STATE);
        let black = state & 1 != 0;
        let ep = get!(T_EP);
        let c = material_context(material, black);
        let r = get!(T_RESULT + c);
        let hi = get!(T_SHIGH + c * RESULTS + r as usize);
        let lo = get!(T_SLOW + hi as usize);
        out.push(Record {
            board: BoardState::from_parts(sq, black, (state >> 1) as u8, ep as u8),
            labels: Labels { score: ((hi << 8) | lo) as u16 as i16, outcome: Outcome::from_symbol(r) },
        });
    }
    if !d.finished_cleanly() {
        return Err(Error::Corrupt { block, record: None, reason: "final state or trailing bytes" });
    }
    Ok(())
}
