// C ABI around the data loader's binpack reader and skip predicate.
//
// fsqd_scan reads one binpack sequentially and hands every position the
// loader's make_skip_predicate accepts to a callback, in batches. Reading in
// file order keeps the predicate's stateful heuristics (placeholder-zero
// detection, adaptive piece-count filter) working as in training.

#include "training_data_loader_internal.h"

#include <cstdint>
#include <cstring>
#include <exception>
#include <vector>

using namespace binpack;
using namespace chess;

std::function<bool(const TrainingDataEntry&)> make_skip_predicate(DataloaderSkipConfig config);

namespace {
constexpr int    TREE_SAMPLE_PLY = 9999;
constexpr size_t BATCH           = 4096;
}

extern "C" {

struct ScanRecord {
    int16_t score;
    int16_t result;
    uint8_t squares[64];  // piece id (type * 2 + color), 12 = empty; a1 = 0
    uint8_t black_to_move;
    uint8_t castling;     // bit 0 K, 1 Q, 2 k, 3 q
    uint8_t ep;           // square, 64 = none
    uint8_t tree_sample;  // ply 9999: the result is a placeholder
};

struct ScanStats {
    uint64_t read;
    uint64_t kept;
    uint64_t tree_samples;  // accepted search-tree samples (counted in kept)
};

// Called with each batch of accepted positions and the number of entries read
// so far. Return nonzero to stop the scan.
typedef int (*ScanSink)(void* ctx, const ScanRecord* records, size_t n, uint64_t read);

// Returns 0 when the file was fully read (or max_kept reached), 1 when the
// sink stopped the scan, -1 on error with a message in err.
int fsqd_scan(const char*          path,
              DataloaderSkipConfig config,
              uint64_t             max_kept,
              ScanSink             sink,
              void*                ctx,
              ScanStats*           stats,
              char*                err,
              size_t               err_len) {
    *stats = ScanStats{};
    try
    {
        auto skip = make_skip_predicate(config);
        CompressedTrainingDataEntryReader reader(path, std::ios_base::in | std::ios_base::binary);
        std::vector<ScanRecord>           buf;
        buf.reserve(BATCH);
        while (reader.hasNext())
        {
            const TrainingDataEntry e = reader.next();
            ++stats->read;
            if (skip && skip(e))
                continue;
            ScanRecord r{};
            r.tree_sample = e.ply == TREE_SAMPLE_PLY;
            stats->tree_samples += r.tree_sample;
            r.score  = e.score;
            r.result = e.result;
            for (int i = 0; i < 64; ++i)
            {
                const Piece p = e.pos.pieceAt(Square(i));
                r.squares[i]  = static_cast<uint8_t>(static_cast<int>(p));
            }
            r.black_to_move = e.pos.sideToMove() == Color::Black;
            r.castling      = static_cast<uint8_t>(e.pos.castlingRights());
            r.ep = e.pos.epSquare() == Square::none() ? 64 : static_cast<uint8_t>(ordinal(e.pos.epSquare()));
            buf.push_back(r);
            ++stats->kept;
            if (buf.size() == BATCH)
            {
                if (sink(ctx, buf.data(), buf.size(), stats->read))
                    return 1;
                buf.clear();
            }
            if (max_kept && stats->kept >= max_kept)
                break;
        }
        if (!buf.empty() && sink(ctx, buf.data(), buf.size(), stats->read))
            return 1;
        return 0;
    } catch (const std::exception& ex)
    {
        std::snprintf(err, err_len, "%s", ex.what());
    } catch (...)
    {
        std::snprintf(err, err_len, "unknown C++ exception");
    }
    return -1;
}

}  // extern "C"
