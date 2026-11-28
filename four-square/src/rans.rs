//! Byte-wise rANS with 10-bit probabilities and a 32-bit state in [L, 256 L).

pub(crate) const PROB_BITS: u32 = 10;
pub(crate) const PROB_SCALE: u32 = 1 << PROB_BITS;
pub(crate) const RANS_L: u32 = 1 << 23;

/// Frequencies summing to 1024, all positive, from raw counts:
/// `f[i] = 1 + floor((P + c[i]) * (1024 - n) / T) - floor(P * (1024 - n) / T)`
/// where `P` is the count before symbol `i` and `T` the total. An all-zero
/// table is treated as one count per symbol.
pub(crate) fn normalize(counts: &[u64]) -> Vec<u16> {
    let n = counts.len();
    assert!(n > 0 && n < PROB_SCALE as usize);
    let budget = (PROB_SCALE as usize - n) as u128;
    let total: u128 = counts.iter().map(|&c| c as u128).sum();
    let (total, ones) = if total == 0 { (n as u128, true) } else { (total, false) };
    let mut prefix = 0u128;
    let mut freqs = Vec::with_capacity(n);
    for &c in counts {
        let c = if ones { 1 } else { c as u128 };
        let f = 1 + (prefix + c) * budget / total - prefix * budget / total;
        freqs.push(f as u16);
        prefix += c;
    }
    debug_assert_eq!(freqs.iter().map(|&f| f as u32).sum::<u32>(), PROB_SCALE);
    freqs
}

/// Decode-table slot: symbol (8 bits), frequency (11 bits), offset (10 bits).
#[inline(always)]
pub(crate) fn pack_slot(sym: u32, freq: u32, offset: u32) -> u32 {
    sym | (freq << 8) | (offset << 19)
}

pub(crate) type DecodeTable = [u32; PROB_SCALE as usize];

pub(crate) fn build_decode_table(freqs: &[u16]) -> Box<DecodeTable> {
    let mut t = Box::new([0u32; PROB_SCALE as usize]);
    let mut start = 0usize;
    for (sym, &f) in freqs.iter().enumerate() {
        for k in 0..f as usize {
            t[start + k] = pack_slot(sym as u32, f as u32, k as u32);
        }
        start += f as usize;
    }
    t
}

/// Encode `(start, freq)` pairs given in decoder order. Appends the final
/// state (u32 LE) followed by the refill bytes in forward-readable order.
pub(crate) fn encode(symbols: &[(u16, u16)], stack: &mut Vec<u8>, out: &mut Vec<u8>) {
    stack.clear();
    let mut x = RANS_L;
    for &(s, f) in symbols.iter().rev() {
        let f = f as u32;
        let x_max = f << (23 - PROB_BITS + 8);
        while x >= x_max {
            stack.push(x as u8);
            x >>= 8;
        }
        x = (x / f) * PROB_SCALE + x % f + s as u32;
    }
    out.extend_from_slice(&x.to_le_bytes());
    out.extend(stack.iter().rev());
}

/// Decoder over one block payload.
pub(crate) struct Decoder<'a> {
    x: u32,
    data: &'a [u8],
    pos: usize,
}

impl<'a> Decoder<'a> {
    /// `None` if the payload is shorter than the state; `Some(Err(()))` if the
    /// initial state lies outside [L, 256 L).
    pub(crate) fn new(data: &'a [u8]) -> Option<std::result::Result<Self, ()>> {
        let head: [u8; 4] = data.get(..4)?.try_into().ok()?;
        let x = u32::from_le_bytes(head);
        if (x as u64) < RANS_L as u64 || (x as u64) >= 256 * RANS_L as u64 {
            return Some(Err(()));
        }
        Some(Ok(Self { x, data, pos: 4 }))
    }

    /// Decode one symbol; `None` on truncation.
    #[inline(always)]
    pub(crate) fn get(&mut self, table: &DecodeTable) -> Option<u32> {
        let e = table[(self.x & (PROB_SCALE - 1)) as usize];
        let sym = e & 0xff;
        let freq = (e >> 8) & 0x7ff;
        let offset = e >> 19;
        self.x = freq * (self.x >> PROB_BITS) + offset;
        while self.x < RANS_L {
            let b = *self.data.get(self.pos)?;
            self.pos += 1;
            self.x = (self.x << 8) | b as u32;
        }
        Some(sym)
    }

    /// True if the final state is L and every payload byte was consumed.
    pub(crate) fn finished_cleanly(&self) -> bool {
        self.x == RANS_L && self.pos == self.data.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_arithmetic_example() {
        // f = 512, s = 512 from x = L gives 16777728 with no refill bytes.
        let mut out = Vec::new();
        encode(&[(512, 512)], &mut Vec::new(), &mut out);
        assert_eq!(out, 16777728u32.to_le_bytes());
        let mut freqs = vec![0u16; 2];
        freqs[0] = 512;
        freqs[1] = 512;
        let table = build_decode_table(&freqs);
        let mut d = Decoder::new(&out).unwrap().unwrap();
        assert_eq!(d.get(&table), Some(1));
        assert!(d.finished_cleanly());
    }

    #[test]
    fn normalize_properties() {
        for counts in [vec![0u64; 5], vec![1, 0, 0, 1_000_000], vec![7; 256], vec![u64::MAX / 2, 3, 0]] {
            let f = normalize(&counts);
            assert_eq!(f.len(), counts.len());
            assert!(f.iter().all(|&x| x >= 1));
            assert_eq!(f.iter().map(|&x| x as u32).sum::<u32>(), PROB_SCALE);
        }
        // All-zero counts behave as one count per symbol.
        assert_eq!(normalize(&[0, 0, 0]), normalize(&[1, 1, 1]));
    }

    #[test]
    fn round_trip_many_symbols() {
        let freqs = normalize(&[100, 1, 50, 7, 0, 3000]);
        let table = build_decode_table(&freqs);
        let starts: Vec<u16> = freqs
            .iter()
            .scan(0u16, |acc, &f| {
                let s = *acc;
                *acc += f;
                Some(s)
            })
            .collect();
        let syms: Vec<usize> = (0..10_000).map(|i| (i * 7 + i / 3) % 6).collect();
        let pairs: Vec<(u16, u16)> = syms.iter().map(|&s| (starts[s], freqs[s])).collect();
        let mut out = Vec::new();
        encode(&pairs, &mut Vec::new(), &mut out);
        let mut d = Decoder::new(&out).unwrap().unwrap();
        for &s in &syms {
            assert_eq!(d.get(&table), Some(s as u32));
        }
        assert!(d.finished_cleanly());
    }
}
