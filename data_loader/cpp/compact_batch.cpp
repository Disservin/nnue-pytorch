#include "training_data_loader_internal.h"
#include "training_data_loader_abi.h"

#include <exception>
#include <array>
#include <bit>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <mutex>
#include <thread>

using namespace binpack;
using namespace chess;

// Bulletformat 1.8 ChessBoard, serialized explicitly in little-endian order.
// Its board is side-to-move-relative; extra bytes preserve training metadata.

struct CompactBatch {
    std::vector<std::uint8_t> records;
};

static bool encode_bullet(const TrainingDataEntry& entry, std::uint8_t* out) {
    std::fill(out, out + 32, 0);
    const auto stm = entry.pos.sideToMove();
    const bool black = stm == Color::Black;
    int count = 0, our_kings = 0, their_kings = 0;
    std::uint64_t occupied = 0;
    for (int square = 0; square < 64; ++square) {
        const auto piece = entry.pos.pieceAt(Square(square ^ (black ? 56 : 0)));
        if (piece == Piece::none()) continue;
        if (count == 32 || ordinal(piece.type()) > ordinal(PieceType::King)) return false;
        occupied |= std::uint64_t{1} << square;
        const bool opponent = piece.color() != stm;
        const auto code = static_cast<std::uint8_t>(ordinal(piece.type()) | (opponent ? 8 : 0));
        out[8 + count / 2] |= static_cast<std::uint8_t>(code << (4 * (count % 2)));
        if (piece.type() == PieceType::King) {
            if (opponent) { out[28] = static_cast<std::uint8_t>(square ^ 56); ++their_kings; }
            else { out[27] = static_cast<std::uint8_t>(square); ++our_kings; }
        }
        ++count;
    }
    if (our_kings != 1 || their_kings != 1 || entry.result < -1 || entry.result > 1) return false;
    for (int i = 0; i < 8; ++i) out[i] = static_cast<std::uint8_t>(occupied >> (8 * i));
    const auto score = static_cast<std::uint16_t>(entry.score);
    out[24] = static_cast<std::uint8_t>(score);
    out[25] = static_cast<std::uint8_t>(score >> 8);
    out[26] = static_cast<std::uint8_t>(entry.result + 1);
    out[29] = entry.pos.rule50Counter();
    out[30] = static_cast<std::uint8_t>(black);
    out[31] = static_cast<std::uint8_t>(entry.ply == 9999);
    return true;
}

struct CompactBatchStream : Stream<CompactBatch> {
    CompactBatchStream(int workers, const std::vector<std::string>& files, int batch_size,
                       bool cyclic, DataloaderSkipConfig config, DataloaderIOConfig io) :
        Stream<CompactBatch>(workers, files, cyclic, make_skip_predicate(config), nullptr, 0, 1, io, 1),
        batch_size(batch_size) {}

    CompactBatch* next() override {
        if (!error.empty()) return nullptr;
        try {
            std::vector<TrainingDataEntry> entries;
            entries.reserve(batch_size);
            m_stream->fill_threadsafe(entries, batch_size);
            if (entries.empty()) return nullptr;
            auto batch = std::make_unique<CompactBatch>();
            batch->records.resize(entries.size() * 32);
            for (std::size_t i = 0; i < entries.size(); ++i) {
                const auto& e = entries[i];
                if (!encode_bullet(e, batch->records.data() + i * 32)) {
                    error = "Cannot encode position in bulletformat: invalid king count, piece count, or result";
                    return nullptr;
                }
            }
            return batch.release();
        } catch (const std::exception& ex) { error = ex.what(); return nullptr; }
        catch (...) { error = "Unknown compact batch error"; return nullptr; }
    }
    int batch_size;
    std::string error;
};

NNUE_API CompactBatchStream* NNUE_CDECL create_compact_batch_stream(
  int workers, int nfiles, const char* const* files, int batch_size, bool cyclic,
  DataloaderSkipConfig config, DataloaderIOConfig io) {
    if (workers < 1 || nfiles < 1 || !files || batch_size < 1) return nullptr;
    try {
        std::vector<std::string> paths(files, files + nfiles);
        for (const auto& p : paths)
            if (!training_data::has_extension(p, "binpack")) return nullptr;
        return new CompactBatchStream(workers, paths, batch_size, cyclic, config, io);
    } catch (...) { return nullptr; }
}
NNUE_API void NNUE_CDECL destroy_compact_batch_stream(CompactBatchStream* s) { delete s; }
NNUE_API CompactBatch* NNUE_CDECL fetch_next_compact_batch(CompactBatchStream* s) { return s ? s->next() : nullptr; }
NNUE_API const char* NNUE_CDECL compact_batch_stream_error(CompactBatchStream* s) { return s ? s->error.c_str() : "Invalid stream"; }
NNUE_API void NNUE_CDECL destroy_compact_batch(CompactBatch* batch) { delete batch; }
NNUE_API std::size_t NNUE_CDECL compact_batch_size(const CompactBatch* b) { return b ? b->records.size() / 32 : 0; }
NNUE_API const std::uint8_t* NNUE_CDECL compact_batch_records(const CompactBatch* b) { return b ? b->records.data() : nullptr; }

static TrainingDataEntry decode_bullet(const std::uint8_t* record) {
    if (record[30] > 1 || record[31] > 1 || record[26] > 2)
        throw std::invalid_argument("invalid compact metadata");
    std::uint64_t occupied = 0;
    for (int i = 0; i < 8; ++i) occupied |= std::uint64_t(record[i]) << (8 * i);
    if (std::popcount(occupied) > 32)
        throw std::invalid_argument("compact position has more than 32 pieces");
    TrainingDataEntry entry{};
    const bool black = record[30] != 0;
    const Color stm = black ? Color::Black : Color::White;
    entry.pos.setSideToMove(stm);
    entry.pos.setCastlingRights(CastlingRights::None);
    entry.pos.setRule50Counter(record[29]);
    int count = 0, kings[2] = {0, 0};
    while (occupied) {
        const int square = std::countr_zero(occupied);
        occupied &= occupied - 1;
        const int code = (record[8 + count / 2] >> (4 * (count % 2))) & 15;
        const int type = code & 7;
        const bool opponent = code & 8;
        if (type > 5) throw std::invalid_argument("invalid compact piece type");
        const Color color = opponent ? !stm : stm;
        const int absolute_square = square ^ (black ? 56 : 0);
        // The feature encodings assume legal pawn ranks.
        if (type == 0 && (absolute_square < 8 || absolute_square >= 56))
            throw std::invalid_argument("pawn on first or eighth rank");
        entry.pos.place(Piece(static_cast<PieceType>(type), color), Square(absolute_square));
        if (type == 5) {
            ++kings[opponent];
            if ((!opponent && record[27] != square)
                || (opponent && record[28] != (square ^ 56)))
                throw std::invalid_argument("compact king square does not match board");
        }
        ++count;
    }
    if (kings[0] != 1 || kings[1] != 1)
        throw std::invalid_argument("compact position must have one king per side");
    const int bits = int(record[24]) | (int(record[25]) << 8);
    entry.score = static_cast<std::int16_t>(bits >= 32768 ? bits - 65536 : bits);
    entry.result = static_cast<std::int16_t>(int(record[26]) - 1);
    entry.ply = record[31] ? 9999 : 0;
    return entry;
}

SparseBatch::SparseBatch(const IFeatureExtractor& feature_set, const LayerStackKeys& ls_keys,
                         const std::uint8_t* records, std::size_t count, int workers, bool direct) {
    num_inputs = feature_set.inputs();
    size = static_cast<int>(count);
    max_active_features = feature_set.max_active_features();
    if (count * std::size_t(max_active_features) > std::size_t(std::numeric_limits<int>::max()))
        throw std::invalid_argument("compact batch exceeds supported feature capacity");
    const std::size_t rows = count * std::size_t(max_active_features);
    // Packed layout (see the binpack constructor), built from padded rows parked behind the
    // offsets and compacted once the counts are known.
    std::unique_ptr<float[]> floats(new float[count * 3]);
    std::unique_ptr<int[]> ints(new int[3 * count + 2 + 2 * rows]);
    is_white = floats.get();
    outcome = is_white + count;
    score = outcome + count;
    packed = ints.get();
    int* const white_offsets = packed;
    int* const black_offsets = white_offsets + count + 1;
    int* const white = black_offsets + count + 1;
    int* const black = white + rows;
    int* const stacks_parked = black + rows;
    std::vector<int> feature_counts(count * 2);

    std::exception_ptr failure;
    std::mutex failure_mutex;
    std::atomic_bool failed{false};
    const int actual_workers = static_cast<int>(std::min(count, std::size_t(workers)));
    {
        std::vector<std::jthread> threads;
        threads.reserve(actual_workers);
        for (int worker = 0; worker < actual_workers; ++worker) {
            const auto start = count * std::size_t(worker) / actual_workers;
            const auto end = count * std::size_t(worker + 1) / actual_workers;
            threads.emplace_back([&, start, end] {
                try {
                    // At most 32 pieces and the supported feature sets imply
                    // fewer than 2048 candidate IDs, even for malformed positions.
                    std::array<int, 2048> scratch;
                    for (std::size_t i = start; i < end && !failed.load(std::memory_order_relaxed); ++i) {
                        const auto* record = records + 32 * i;
                        auto write = [&](const auto& board, auto extract) {
                            for (int perspective = 0; perspective < 2; ++perspective) {
                                auto color = perspective == 0 ? Color::White : Color::Black;
                                const int n = extract(color);
                                if (n < 0 || n > max_active_features)
                                    throw std::invalid_argument("compact position exceeds feature capacity");
                                int* row = (perspective == 0 ? white : black) + i * max_active_features;
                                std::copy_n(scratch.data(), n, row);
                                feature_counts[std::size_t(perspective) * count + i] = n;
                            }
                            stacks_parked[i] = ls_keys.index(board);
                        };
                        if (direct) {
                            const CompactFeatureBoard board(record);
                            write(board, [&](Color color) {
                                return feature_set.fill_features_compact(board, scratch.data(), color).first;
                            });
                        } else {
                            const auto entry = decode_bullet(record);
                            write(entry.pos, [&](Color color) {
                                return feature_set.fill_features_sparse(entry, scratch.data(), color).first;
                            });
                        }
                        is_white[i] = record[30] ? 0.0f : 1.0f;
                        outcome[i] = record[31] ? 2.0f : record[26] / 2.0f;
                        const int bits = int(record[24]) | (int(record[25]) << 8);
                        score[i] = static_cast<float>(bits >= 32768 ? bits - 65536 : bits);
                    }
                } catch (...) {
                    failed.store(true, std::memory_order_relaxed);
                    std::lock_guard lock(failure_mutex);
                    if (!failure) failure = std::current_exception();
                }
            });
        }
    }
    if (failure) std::rethrow_exception(failure);
    // Compact the rows in order: every destination is at or before its source.
    num_active_white_features = 0;
    white_offsets[0] = 0;
    for (std::size_t i = 0; i < count; ++i) {
        const int n = feature_counts[i];
        std::memmove(white + num_active_white_features, white + i * max_active_features,
                     std::size_t(n) * sizeof(int));
        num_active_white_features += n;
        white_offsets[i + 1] = num_active_white_features;
    }
    int* const black_ids = white + num_active_white_features;
    num_active_black_features = 0;
    black_offsets[0] = 0;
    for (std::size_t i = 0; i < count; ++i) {
        const int n = feature_counts[count + i];
        std::memmove(black_ids + num_active_black_features, black + i * max_active_features,
                     std::size_t(n) * sizeof(int));
        num_active_black_features += n;
        black_offsets[i + 1] = num_active_black_features;
    }
    std::memmove(black_ids + num_active_black_features, stacks_parked, count * sizeof(int));
    m_float_block = floats.release();
    m_int_block = ints.release();
}

NNUE_API SparseBatch* NNUE_CDECL get_sparse_batch_from_compact_backend(
  const char* feature_set, const char* ls_keys, const std::uint8_t* records,
  std::size_t record_bytes, int workers, bool direct, char* error, std::size_t error_capacity) {
    auto fail = [&](const char* message) -> SparseBatch* {
        if (error && error_capacity) {
            std::strncpy(error, message, error_capacity - 1);
            error[error_capacity - 1] = '\0';
        }
        return nullptr;
    };
    if (error && error_capacity) error[0] = '\0';
    if (!feature_set || !ls_keys || workers < 1 || record_bytes % 32 || (record_bytes && !records)
        || record_bytes / 32 > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        return fail("invalid compact batch arguments");
    try {
        // Bound scratch usage even for malformed positions: for these three
        // feature kinds, <=32 pieces imply <2048 feature candidates in total.
        std::string names(feature_set);
        std::unordered_set<std::string> seen;
        for (std::size_t start = 0; start < names.size();) {
            auto end = names.find('+', start);
            auto part = names.substr(start, end == std::string::npos ? end : end - start);
            if ((part != "Full_Threats" && part != "PP_3Wide" && part != "K32H3")
                || !seen.insert(part).second)
                return fail("unsupported or repeated compact feature component");
            if (end == std::string::npos) break;
            start = end + 1;
            if (start == names.size()) return fail("invalid feature list");
        }
        auto feature = get_feature(names);
        auto keys = get_layer_stack_keys(ls_keys);
        if (!feature || !keys) return fail("unknown features or layer-stack keys");
        return new SparseBatch(*feature, *keys, records, record_bytes / 32, workers, direct);
    } catch (const std::exception& ex) { return fail(ex.what()); }
    catch (...) { return fail("unknown compact extraction error"); }
}

NNUE_API SparseBatch* NNUE_CDECL get_sparse_batch_from_compact_parallel(
  const char* feature_set, const char* ls_keys, const std::uint8_t* records,
  std::size_t record_bytes, int workers, char* error, std::size_t error_capacity) {
    return get_sparse_batch_from_compact_backend(feature_set, ls_keys, records, record_bytes,
                                                 workers, false, error, error_capacity);
}
