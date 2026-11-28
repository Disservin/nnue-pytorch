use sfbinpack::chess::castling_rights::CastlingRights;
use sfbinpack::chess::color::Color;
use sfbinpack::chess::coords::Square;
use sfbinpack::TrainingDataEntry;

use crate::error::{Error, Result};

/// Piece IDs, in the order `P p N n B b R r Q q K k empty`.
pub mod piece {
    pub const WHITE_PAWN: u8 = 0;
    pub const BLACK_PAWN: u8 = 1;
    pub const WHITE_KNIGHT: u8 = 2;
    pub const BLACK_KNIGHT: u8 = 3;
    pub const WHITE_BISHOP: u8 = 4;
    pub const BLACK_BISHOP: u8 = 5;
    pub const WHITE_ROOK: u8 = 6;
    pub const BLACK_ROOK: u8 = 7;
    pub const WHITE_QUEEN: u8 = 8;
    pub const BLACK_QUEEN: u8 = 9;
    pub const WHITE_KING: u8 = 10;
    pub const BLACK_KING: u8 = 11;
    pub const EMPTY: u8 = 12;
}

/// Castling flags of [`BoardState::castling`].
pub mod castling {
    pub const WHITE_KING_SIDE: u8 = 1;
    pub const WHITE_QUEEN_SIDE: u8 = 2;
    pub const BLACK_KING_SIDE: u8 = 4;
    pub const BLACK_QUEEN_SIDE: u8 = 8;
}

/// A board with side to move, castling rights and en passant square.
///
/// Squares are rank-major with a1 = 0 (`8 * rank + file`). Every value is
/// validated on construction: piece IDs are 0..=12, there is exactly one king
/// of each colour, castling uses the low four bits and the en passant square
/// is 0..64.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BoardState {
    squares: [u8; 64],
    black_to_move: bool,
    castling: u8,
    ep: u8, // 64 = none
}

impl BoardState {
    pub fn new(squares: [u8; 64], black_to_move: bool, castling: u8, ep_square: Option<u8>) -> Result<Self> {
        let (mut wk, mut bk) = (0, 0);
        for &p in &squares {
            match p {
                piece::WHITE_KING => wk += 1,
                piece::BLACK_KING => bk += 1,
                0..=12 => {}
                _ => return Err(Error::InvalidRecord(format!("piece id {p} out of range"))),
            }
        }
        if wk != 1 || bk != 1 {
            return Err(Error::InvalidRecord(format!("need one king per colour, found {wk} white and {bk} black")));
        }
        if castling > 15 {
            return Err(Error::InvalidRecord(format!("castling flags {castling:#x} out of range")));
        }
        let ep = match ep_square {
            None => 64,
            Some(sq) if sq < 64 => sq,
            Some(sq) => return Err(Error::InvalidRecord(format!("ep square {sq} out of range"))),
        };
        Ok(Self { squares, black_to_move, castling, ep })
    }

    /// Caller guarantees the invariants checked by [`BoardState::new`].
    pub(crate) fn from_parts(squares: [u8; 64], black_to_move: bool, castling: u8, ep: u8) -> Self {
        Self { squares, black_to_move, castling, ep }
    }

    /// Piece IDs of all 64 squares, rank-major, a1 = 0.
    pub fn squares(&self) -> &[u8; 64] {
        &self.squares
    }

    pub fn piece_at(&self, square: usize) -> u8 {
        self.squares[square]
    }

    pub fn black_to_move(&self) -> bool {
        self.black_to_move
    }

    /// Castling flags, see [`castling`].
    pub fn castling(&self) -> u8 {
        self.castling
    }

    pub fn ep_square(&self) -> Option<u8> {
        (self.ep < 64).then_some(self.ep)
    }

    pub(crate) fn ep_raw(&self) -> u8 {
        self.ep
    }

    pub fn white_king(&self) -> u8 {
        self.king(piece::WHITE_KING)
    }

    pub fn black_king(&self) -> u8 {
        self.king(piece::BLACK_KING)
    }

    fn king(&self, id: u8) -> u8 {
        self.squares.iter().position(|&p| p == id).expect("validated board has both kings") as u8
    }
}

/// Ply that marks a search-tree sample in a binpack. Such positions come from
/// search trees, not finished games, so their result field is a placeholder.
pub const TREE_SAMPLE_PLY: u16 = 9999;

/// Game result from the side to move's point of view, or `NoResult` for
/// positions without one (search-tree samples), which train on the score alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Outcome {
    Loss = 0,
    Draw = 1,
    Win = 2,
    NoResult = 3,
}

impl Outcome {
    /// From a binpack result: -1, 0 or 1 for the side to move.
    pub fn from_result(result: i16) -> Result<Self> {
        match result {
            -1 => Ok(Outcome::Loss),
            0 => Ok(Outcome::Draw),
            1 => Ok(Outcome::Win),
            r => Err(Error::InvalidRecord(format!("result {r} is not -1, 0 or 1"))),
        }
    }

    /// The binpack result (-1, 0 or 1), or `None` for `NoResult`.
    pub fn as_result(self) -> Option<i16> {
        (self != Outcome::NoResult).then_some(self as i16 - 1)
    }

    pub(crate) fn from_symbol(s: u32) -> Self {
        match s {
            0 => Outcome::Loss,
            1 => Outcome::Draw,
            2 => Outcome::Win,
            _ => Outcome::NoResult,
        }
    }
}

/// Score and outcome, both relative to the side to move.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Labels {
    pub score: i16,
    pub outcome: Outcome,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Record {
    pub board: BoardState,
    pub labels: Labels,
}

impl Record {
    /// Project a binpack entry. The move, ply and clocks are dropped, so apply
    /// any filter that needs them before projecting. Search-tree samples
    /// (ply [`TREE_SAMPLE_PLY`]) get [`Outcome::NoResult`] whatever their
    /// stored result.
    pub fn project(entry: &TrainingDataEntry) -> Result<Self> {
        let pos = &entry.pos;
        let mut squares = [piece::EMPTY; 64];
        for (i, sq) in squares.iter_mut().enumerate() {
            *sq = pos.piece_at(Square::new(i as u32)).id();
        }
        let cr = pos.castling_rights();
        let mut castle = 0u8;
        for (flag, right) in [
            (castling::WHITE_KING_SIDE, CastlingRights::WHITE_KING_SIDE),
            (castling::WHITE_QUEEN_SIDE, CastlingRights::WHITE_QUEEN_SIDE),
            (castling::BLACK_KING_SIDE, CastlingRights::BLACK_KING_SIDE),
            (castling::BLACK_QUEEN_SIDE, CastlingRights::BLACK_QUEEN_SIDE),
        ] {
            if cr.contains(right) {
                castle |= flag;
            }
        }
        let ep = pos.ep_square();
        let ep = (ep != Square::NONE).then(|| ep.index() as u8);
        let board = BoardState::new(squares, pos.side_to_move() == Color::Black, castle, ep)?;
        let outcome =
            if entry.ply == TREE_SAMPLE_PLY { Outcome::NoResult } else { Outcome::from_result(entry.result)? };
        let labels = Labels { score: entry.score, outcome };
        Ok(Record { board, labels })
    }
}
