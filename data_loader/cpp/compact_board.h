#pragma once

#include "lib/chess.h"

#include <array>
#include <bit>
#include <cstdint>
#include <stdexcept>

// Feature-only view built directly from the 32-byte record. It contains no
// chess::Board/Position, move state, piece-count table, castling or EP state.
// Absolute-square ordering is retained so native extractor output is identical.
struct CompactFeatureBoard {
    std::array<std::uint64_t, 12> piece_bits{};
    std::array<std::uint64_t, 2> color_bits{};
    std::array<std::uint8_t, 64> mailbox;
    std::array<int, 2> king_square{};
    std::uint64_t occupancy = 0;
    chess::Color stm;

    explicit CompactFeatureBoard(const std::uint8_t* record) :
        stm(record[30] ? chess::Color::Black : chess::Color::White) {
        using namespace chess;
        if (record[30] > 1 || record[31] > 1 || record[26] > 2)
            throw std::invalid_argument("invalid compact metadata");
        std::uint64_t packed_occupancy = 0;
        for (int i = 0; i < 8; ++i)
            packed_occupancy |= std::uint64_t(record[i]) << (8 * i);
        if (std::popcount(packed_occupancy) > 32)
            throw std::invalid_argument("compact position has more than 32 pieces");
        mailbox.fill(12); // native Piece::none() has ID 12
        int count = 0, kings[2] = {0, 0};
        while (packed_occupancy) {
            const int square = std::countr_zero(packed_occupancy);
            packed_occupancy &= packed_occupancy - 1;
            const int code = (record[8 + count / 2] >> (4 * (count % 2))) & 15;
            const int type = code & 7;
            const int opponent = (code & 8) != 0;
            if (type > 5) throw std::invalid_argument("invalid compact piece type");
            const int absolute = square ^ (record[30] ? 56 : 0);
            if (type == 0 && (absolute < 8 || absolute >= 56))
                throw std::invalid_argument("pawn on first or eighth rank");
            const int color = opponent ^ int(record[30]);
            const int id = type * 2 + color;
            const std::uint64_t bit = std::uint64_t{1} << absolute;
            piece_bits[id] |= bit;
            color_bits[color] |= bit;
            occupancy |= bit;
            mailbox[absolute] = static_cast<std::uint8_t>(id);
            if (type == 5) {
                ++kings[opponent];
                king_square[color] = absolute;
                if ((!opponent && record[27] != square)
                    || (opponent && record[28] != (square ^ 56)))
                    throw std::invalid_argument("compact king square does not match board");
            }
            ++count;
        }
        if (kings[0] != 1 || kings[1] != 1)
            throw std::invalid_argument("compact position must have one king per side");
    }

    chess::Bitboard piecesBB() const { return chess::Bitboard::fromBits(occupancy); }
    chess::Bitboard piecesBB(chess::Color c) const { return chess::Bitboard::fromBits(color_bits[int(c)]); }
    chess::Bitboard piecesBB(chess::Piece p) const { return chess::Bitboard::fromBits(piece_bits[int(p)]); }
    chess::Piece pieceAt(chess::Square s) const { return chess::Piece::fromId(mailbox[int(s)]); }
    chess::Square kingSquare(chess::Color c) const { return chess::Square(king_square[int(c)]); }
    chess::Color sideToMove() const { return stm; }
    chess::Bitboard attacks(chess::Square s) const {
        const auto p = pieceAt(s);
        if (p == chess::Piece::none()) return chess::Bitboard::none();
        if (p.type() == chess::PieceType::Pawn)
            return chess::bb::pawnAttacks(chess::Bitboard::square(s), p.color());
        return chess::bb::attacks(p.type(), s, piecesBB());
    }
};
