// End-to-end filtered binpack -> Bulletformat test. Link against the loader.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include "training_data_loader_abi.h"
#include "training_data_loader_internal.h"
#include "lib/binpack.h"

#include <array>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <string>

using namespace chess;
using namespace binpack;

static void check_record(const std::uint8_t* r, const Position& original) {
    std::uint64_t occ = 0;
    for (int i = 0; i < 8; ++i) occ |= std::uint64_t(r[i]) << (8 * i);
    int n = 0;
    for (int sq = 0; sq < 64; ++sq) {
        auto piece = original.pieceAt(Square(sq ^ (r[30] ? 56 : 0)));
        const bool exists = piece != Piece::none();
        assert(bool(occ & (std::uint64_t{1} << sq)) == exists);
        if (!exists) continue;
        const int code = (r[8 + n / 2] >> (4 * (n % 2))) & 15;
        const int expected = ordinal(piece.type()) | ((piece.color() != original.sideToMove()) ? 8 : 0);
        assert(code == expected);
        if (piece.type() == PieceType::King) {
            if (piece.color() == original.sideToMove()) assert(r[27] == sq);
            else assert(r[28] == (sq ^ 56));
        }
        ++n;
    }
    assert(r[29] == original.rule50Counter());
    assert(r[30] == (original.sideToMove() == Color::Black));
}

int main(int argc, char** argv) {
    assert(argc == 2);
    const std::string path = argv[1];
    std::remove(path.c_str());
    std::map<int, Position> expected;
    {
        CompressedTrainingDataEntryWriter writer(path);
        auto pos = Position::fromFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        for (int i = 0; i < 12; ++i) {
            TrainingDataEntry e{};
            e.pos = pos;
            bool found = false;
            movegen::forEachLegalMove(pos, [&](Move move) { if (!found) { e.move = move; found = true; } });
            assert(found);
            e.score = static_cast<std::int16_t>(100 + i);
            e.ply = static_cast<std::uint16_t>(i);
            e.result = static_cast<std::int16_t>((i % 3) - 1);
            expected.emplace(e.score, pos);
            writer.addTrainingDataEntry(e);
            pos = pos.afterMove(e.move);
        }
        TrainingDataEntry tree{};
        tree.pos = pos;
        bool found = false;
        movegen::forEachLegalMove(pos, [&](Move move) { if (!found) { tree.move = move; found = true; } });
        assert(found);
        tree.score = 222;
        tree.ply = 9999;
        tree.result = 0;
        expected.emplace(tree.score, pos);
        writer.addTrainingDataEntry(tree);
    }
    const char* files[] = {path.c_str()};
    DataloaderSkipConfig filter{};
    filter.early_fen_skipping = -1;
    filter.soft_early_fen_skipping = 0;
    filter.tree_sample_accept = 1.0;
    DataloaderIOConfig io{};
    io.shuffle_buffer_entries = 32;
    auto* stream = create_compact_batch_stream(4, 1, files, 5, false, filter, io);
    assert(stream);
    int total = 0, white = 0, black = 0;
    while (auto* batch = fetch_next_compact_batch(stream)) {
        const auto count = compact_batch_size(batch);
        assert(count <= 5);
        const auto* records = compact_batch_records(batch);
        std::vector<std::string> fens;
        std::vector<int> scores, plies, results;
        for (std::size_t i = 0; i < count; ++i) {
            const auto* record = records + 32 * i;
            const int score = record[24] | (int(record[25]) << 8);
            if (record[31]) assert(score == 222 && record[26] == 1);
            else {
                assert(score >= 100 && score < 112);
                assert(record[26] == (score - 100) % 3);
            }
            const auto& pos = expected.at(score);
            fens.push_back(pos.fen());
            scores.push_back(score);
            plies.push_back(record[31] ? 9999 : score - 100);
            results.push_back(int(record[26]) - 1);
            assert(record[31] <= 1);
            check_record(record, pos);
            if (score == 100) {
                // Standard initial position: known Bulletformat bytes, independent
                // of the round-trip decoder above.
                const std::array<std::uint8_t, 32> golden = {
                  0xff, 0xff, 0, 0, 0, 0, 0xff, 0xff,
                  0x13, 0x42, 0x25, 0x31, 0, 0, 0, 0,
                  0x88, 0x88, 0x88, 0x88, 0x9b, 0xca, 0xad, 0xb9,
                  100, 0, 0, 4, 4, 0, 0, 0};
                assert(std::memcmp(records + 32 * i, golden.data(), 32) == 0);
            }
            (record[30] ? black : white)++;
            ++total;
        }
        std::vector<const char*> fen_ptrs;
        for (const auto& fen : fens) fen_ptrs.push_back(fen.c_str());
        constexpr auto keys = "piececount:queen:pawnhalf:manyoffi";
        for (const char* test_keys : {keys, "queen:piececount", "manyoffi:pawnhalf"})
        for (const char* name : {"Full_Threats", "PP_3Wide", "K32H3",
                                 "Full_Threats+PP_3Wide+K32H3", "K32H3+Full_Threats+PP_3Wide"}) {
            auto* reference = get_sparse_batch_from_fens(name, test_keys, int(count), fen_ptrs.data(),
                                                        scores.data(), plies.data(), results.data(),
                                                        /*prepack=*/false);
            char error[256] = {};
            auto* decoded = get_sparse_batch_from_compact_backend(name, test_keys, records, count * 32, 8, true,
                                                          error, sizeof(error));
            if (!decoded) std::fprintf(stderr, "compact extraction: %s\n", error);
            assert(reference && decoded && reference->size == decoded->size);
            assert(reference->num_inputs == decoded->num_inputs);
            assert(reference->max_active_features == decoded->max_active_features);
            assert(reference->num_active_white_features == decoded->num_active_white_features);
            assert(reference->num_active_black_features == decoded->num_active_black_features);
            const std::size_t int_bytes = (3 * std::size_t(count) + 2
                                           + reference->num_active_white_features
                                           + reference->num_active_black_features) * sizeof(int);
            assert(std::memcmp(reference->packed, decoded->packed, int_bytes) == 0);
            assert(std::memcmp(reference->is_white, decoded->is_white, count * sizeof(float)) == 0);
            assert(std::memcmp(reference->outcome, decoded->outcome, count * sizeof(float)) == 0);
            assert(std::memcmp(reference->score, decoded->score, count * sizeof(float)) == 0);
            destroy_sparse_batch(reference);
            destroy_sparse_batch(decoded);
        }
        std::vector<std::uint8_t> invalid(records, records + count * 32);
        invalid[30] = 2;
        char error[256] = {};
        assert(!get_sparse_batch_from_compact_backend("K32H3", keys, invalid.data(), invalid.size(), 8, true, error, sizeof(error)));
        assert(std::strlen(error) > 0);
        destroy_compact_batch(batch);
    }
    assert(std::strlen(compact_batch_stream_error(stream)) == 0);
    std::printf("decoded=%d white=%d black=%d\n", total, white, black);
    assert(total == 13 && white && black);
    destroy_compact_batch_stream(stream);
    // Ordinary positions are rejected by the hard-ply filter, while the
    // score-only tree position bypasses it when tree acceptance is enabled.
    filter.early_fen_skipping = 11;
    stream = create_compact_batch_stream(4, 1, files, 5, false, filter, io);
    assert(stream);
    auto* tree_batch = fetch_next_compact_batch(stream);
    assert(tree_batch && compact_batch_size(tree_batch) == 1);
    assert(compact_batch_records(tree_batch)[31] == 1);
    destroy_compact_batch(tree_batch);
    assert(fetch_next_compact_batch(stream) == nullptr);
    assert(std::strlen(compact_batch_stream_error(stream)) == 0);
    destroy_compact_batch_stream(stream);
    filter.tree_sample_accept = 0.0;
    stream = create_compact_batch_stream(4, 1, files, 5, false, filter, io);
    assert(stream);
    assert(fetch_next_compact_batch(stream) == nullptr);
    assert(std::strlen(compact_batch_stream_error(stream)) == 0);
    destroy_compact_batch_stream(stream);
    // Tree-only filtering must not activate regular position filters.
    filter.early_fen_skipping = -1;
    stream = create_compact_batch_stream(4, 1, files, 5, false, filter, io);
    assert(stream);
    int ordinary = 0;
    while (auto* batch = fetch_next_compact_batch(stream)) {
        for (std::size_t i = 0; i < compact_batch_size(batch); ++i)
            assert(compact_batch_records(batch)[i * 32 + 31] == 0);
        ordinary += static_cast<int>(compact_batch_size(batch));
        destroy_compact_batch(batch);
    }
    assert(ordinary == 12);
    assert(std::strlen(compact_batch_stream_error(stream)) == 0);
    destroy_compact_batch_stream(stream);
    std::remove(path.c_str());
    // The native Position already stores this counter in a byte; test its maximum.
    {
        CompressedTrainingDataEntryWriter writer(path);
        TrainingDataEntry e{};
        e.pos = Position::fromFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        e.pos.setRule50Counter(255);
        bool found = false;
        movegen::forEachLegalMove(e.pos, [&](Move move) { if (!found) { e.move = move; found = true; } });
        assert(found);
        e.score = 123;
        e.result = 0;
        writer.addTrainingDataEntry(e);
    }
    filter.tree_sample_accept = 1.0;
    stream = create_compact_batch_stream(4, 1, files, 5, false, filter, io);
    assert(stream);
    auto* counter_batch = fetch_next_compact_batch(stream);
    assert(counter_batch && compact_batch_size(counter_batch) == 1);
    assert(compact_batch_records(counter_batch)[29] == 255);
    destroy_compact_batch(counter_batch);
    assert(fetch_next_compact_batch(stream) == nullptr);
    assert(std::strlen(compact_batch_stream_error(stream)) == 0);
    destroy_compact_batch_stream(stream);
    std::remove(path.c_str());
    // Broader parity: positions from random legal games, including captures,
    // both perspectives, and varied king/queen/pawn configurations.
    std::map<int, TrainingDataEntry> random_entries;
    {
        CompressedTrainingDataEntryWriter writer(path);
        std::mt19937 rng(48271);
        auto pos = Position::startPosition();
        int ply = 0;
        for (int i = 0; i < 8192; ++i) {
            std::vector<Move> moves;
            movegen::forEachLegalMove(pos, [&](Move move) { moves.push_back(move); });
            if (moves.empty() || ply == 160) {
                pos = Position::startPosition();
                ply = 0;
                moves.clear();
                movegen::forEachLegalMove(pos, [&](Move move) { moves.push_back(move); });
            }
            TrainingDataEntry e{};
            e.pos = pos;
            e.move = moves[rng() % moves.size()];
            e.score = static_cast<std::int16_t>(1000 + i);
            e.ply = static_cast<std::uint16_t>(ply);
            e.result = 0;
            random_entries.emplace(e.score, e);
            writer.addTrainingDataEntry(e);
            pos = pos.afterMove(e.move);
            ++ply;
        }
    }
    stream = create_compact_batch_stream(4, 1, files, 47, false, filter, io);
    assert(stream);
    int verified = 0;
    constexpr auto random_features = "Full_Threats+PP_3Wide+K32H3";
    constexpr auto random_keys = "piececount:queen:pawnhalf:manyoffi";
    while (auto* batch = fetch_next_compact_batch(stream)) {
        const auto count = compact_batch_size(batch);
        const auto* records = compact_batch_records(batch);
        std::vector<std::string> fens;
        std::vector<int> scores, plies, results;
        for (std::size_t i = 0; i < count; ++i) {
            const auto* r = records + 32 * i;
            const int score = r[24] | (int(r[25]) << 8);
            const auto& e = random_entries.at(score);
            fens.push_back(e.pos.fen());
            scores.push_back(e.score);
            plies.push_back(e.ply);
            results.push_back(e.result);
        }
        std::vector<const char*> fen_ptrs;
        for (const auto& fen : fens) fen_ptrs.push_back(fen.c_str());
        auto* reference = get_sparse_batch_from_fens(random_features, random_keys, int(count),
          fen_ptrs.data(), scores.data(), plies.data(), results.data(), /*prepack=*/false);
        char error[256] = {};
        auto* decoded = get_sparse_batch_from_compact_backend(random_features, random_keys, records,
                                                       count * 32, 8, true, error, sizeof(error));
        auto* single = get_sparse_batch_from_compact_parallel(random_features, random_keys, records,
                                                       count * 32, 1, error, sizeof(error));
        if (!decoded) std::fprintf(stderr, "random compact extraction: %s\n", error);
        assert(reference && decoded && single);
        assert(reference->num_active_white_features == decoded->num_active_white_features);
        assert(reference->num_active_black_features == decoded->num_active_black_features);
        assert(single->num_active_white_features == decoded->num_active_white_features);
        assert(single->num_active_black_features == decoded->num_active_black_features);
        const auto bytes = (3 * std::size_t(count) + 2 + reference->num_active_white_features
                            + reference->num_active_black_features) * sizeof(int);
        assert(std::memcmp(reference->packed, decoded->packed, bytes) == 0);
        assert(std::memcmp(single->packed, decoded->packed, bytes) == 0);
        destroy_sparse_batch(reference);
        destroy_sparse_batch(decoded);
        destroy_sparse_batch(single);
        destroy_compact_batch(batch);
        verified += static_cast<int>(count);
    }
    assert(verified == 8192 && std::strlen(compact_batch_stream_error(stream)) == 0);
    destroy_compact_batch_stream(stream);
    std::remove(path.c_str());
    std::printf("verified both perspectives on %d random positions\n", verified);
    std::puts("compact batch format, metadata, partial batch and filtering passed");
}
