#include <iostream>
#include <memory>
#include <string>
#include <algorithm>
#include <iterator>
#include <future>
#include <mutex>
#include <thread>
#include <deque>
#include <random>

#include "../lib/nnue_training_data_formats.h"
#include "../lib/nnue_training_data_stream.h"
#include "../lib/rng.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/cast.h>

using namespace binpack;
using namespace chess;

static Square orient(Color color, Square sq)
{
    if (color == Color::White)
    {
        return sq;
    }
    else
    {
        // IMPORTANT: for now we use rotate180 instead of rank flip
        //            for compatibility with the stockfish master branch.
        //            Note that this is inconsistent with nodchip/master.
        return sq.flippedVertically().flippedHorizontally();
    }
}

static Square orient_flip(Color color, Square sq)
{
    if (color == Color::White)
    {
        return sq;
    }
    else
    {
        return sq.flippedVertically();
    }
}

struct HalfKP {
    static constexpr int NUM_SQ = 64;
    static constexpr int NUM_PT = 10;
    static constexpr int NUM_PLANES = (NUM_SQ * NUM_PT + 1);
    static constexpr int INPUTS = NUM_PLANES * NUM_SQ;

    static constexpr int MAX_ACTIVE_FEATURES = 32;

    static int feature_index(Color color, Square ksq, Square sq, Piece p)
    {
        auto p_idx = static_cast<int>(p.type()) * 2 + (p.color() != color);
        return 1 + static_cast<int>(orient(color, sq)) + p_idx * NUM_SQ + static_cast<int>(ksq) * NUM_PLANES;
    }

    static std::pair<int, int> fill_features_sparse(const TrainingDataEntry& e, int* features, float* values, Color color)
    {
        auto& pos = e.pos;
        auto pieces = pos.piecesBB() & ~(pos.piecesBB(Piece(PieceType::King, Color::White)) | pos.piecesBB(Piece(PieceType::King, Color::Black)));
        auto ksq = pos.kingSquare(color);

        // We order the features so that the resulting sparse
        // tensor is coalesced.
        int j = 0;
        for(Square sq : pieces)
        {
            auto p = pos.pieceAt(sq);
            values[j] = 1.0f;
            features[j] = feature_index(color, orient(color, ksq), sq, p);
            ++j;
        }

        return { j, INPUTS };
    }
};

struct HalfKPFactorized {
    // Factorized features
    static constexpr int K_INPUTS = HalfKP::NUM_SQ;
    static constexpr int PIECE_INPUTS = HalfKP::NUM_SQ * HalfKP::NUM_PT;
    static constexpr int INPUTS = HalfKP::INPUTS + K_INPUTS + PIECE_INPUTS;

    static constexpr int MAX_K_FEATURES = 1;
    static constexpr int MAX_PIECE_FEATURES = 32;
    static constexpr int MAX_ACTIVE_FEATURES = HalfKP::MAX_ACTIVE_FEATURES + MAX_K_FEATURES + MAX_PIECE_FEATURES;

    static std::pair<int, int> fill_features_sparse(const TrainingDataEntry& e, int* features, float* values, Color color)
    {
        auto [start_j, offset] = HalfKP::fill_features_sparse(e, features, values, color);
        int j = start_j;
        auto& pos = e.pos;
        {
            // king square factor
            auto ksq = pos.kingSquare(color);
            features[j] = offset + static_cast<int>(orient(color, ksq));
            values[j] = static_cast<float>(start_j);
            ++j;
        }
        offset += K_INPUTS;
        auto pieces = pos.piecesBB() & ~(pos.piecesBB(Piece(PieceType::King, Color::White)) | pos.piecesBB(Piece(PieceType::King, Color::Black)));

        // We order the features so that the resulting sparse
        // tensor is coalesced. Note that we can just sort
        // the parts where values are all 1.0f and leave the
        // halfk feature where it was.
        for(Square sq : pieces)
        {
            auto p = pos.pieceAt(sq);
            auto p_idx = static_cast<int>(p.type()) * 2 + (p.color() != color);
            values[j] = 1.0f;
            features[j] = offset + (p_idx * HalfKP::NUM_SQ) + static_cast<int>(orient(color, sq));
            ++j;
        }

        return { j, INPUTS };
    }
};

struct HalfKA {
    static constexpr int NUM_SQ = 64;
    static constexpr int NUM_PT = 12;
    static constexpr int NUM_PLANES = (NUM_SQ * NUM_PT + 1);
    static constexpr int INPUTS = NUM_PLANES * NUM_SQ;

    static constexpr int MAX_ACTIVE_FEATURES = 32;

    static int feature_index(Color color, Square ksq, Square sq, Piece p)
    {
        auto p_idx = static_cast<int>(p.type()) * 2 + (p.color() != color);
        return 1 + static_cast<int>(orient_flip(color, sq)) + p_idx * NUM_SQ + static_cast<int>(ksq) * NUM_PLANES;
    }

    static std::pair<int, int> fill_features_sparse(const TrainingDataEntry& e, int* features, float* values, Color color)
    {
        auto& pos = e.pos;
        auto pieces = pos.piecesBB();
        auto ksq = pos.kingSquare(color);

        int j = 0;
        for(Square sq : pieces)
        {
            auto p = pos.pieceAt(sq);
            values[j] = 1.0f;
            features[j] = feature_index(color, orient_flip(color, ksq), sq, p);
            ++j;
        }

        return { j, INPUTS };
    }
};

struct HalfKAFactorized {
    // Factorized features
    static constexpr int PIECE_INPUTS = HalfKA::NUM_SQ * HalfKA::NUM_PT;
    static constexpr int INPUTS = HalfKA::INPUTS + PIECE_INPUTS;

    static constexpr int MAX_PIECE_FEATURES = 32;
    static constexpr int MAX_ACTIVE_FEATURES = HalfKA::MAX_ACTIVE_FEATURES + MAX_PIECE_FEATURES;

    static std::pair<int, int> fill_features_sparse(const TrainingDataEntry& e, int* features, float* values, Color color)
    {
        const auto [start_j, offset] = HalfKA::fill_features_sparse(e, features, values, color);
        auto& pos = e.pos;
        auto pieces = pos.piecesBB();

        int j = start_j;
        for(Square sq : pieces)
        {
            auto p = pos.pieceAt(sq);
            auto p_idx = static_cast<int>(p.type()) * 2 + (p.color() != color);
            values[j] = 1.0f;
            features[j] = offset + (p_idx * HalfKA::NUM_SQ) + static_cast<int>(orient_flip(color, sq));
            ++j;
        }

        return { j, INPUTS };
    }
};

struct HalfKAv2 {
    static constexpr int NUM_SQ = 64;
    static constexpr int NUM_PT = 11;
    static constexpr int NUM_PLANES = NUM_SQ * NUM_PT;
    static constexpr int INPUTS = NUM_PLANES * NUM_SQ;

    static constexpr int MAX_ACTIVE_FEATURES = 32;

    static int feature_index(Color color, Square ksq, Square sq, Piece p)
    {
        auto p_idx = static_cast<int>(p.type()) * 2 + (p.color() != color);
        if (p_idx == 11)
            --p_idx; // pack the opposite king into the same NUM_SQ * NUM_SQ
        return static_cast<int>(orient_flip(color, sq)) + p_idx * NUM_SQ + static_cast<int>(ksq) * NUM_PLANES;
    }

    static std::pair<int, int> fill_features_sparse(const TrainingDataEntry& e, int* features, float* values, Color color)
    {
        auto& pos = e.pos;
        auto pieces = pos.piecesBB();
        auto ksq = pos.kingSquare(color);

        int j = 0;
        for(Square sq : pieces)
        {
            auto p = pos.pieceAt(sq);
            values[j] = 1.0f;
            features[j] = feature_index(color, orient_flip(color, ksq), sq, p);
            ++j;
        }

        return { j, INPUTS };
    }
};

struct HalfKAv2Factorized {
    // Factorized features
    static constexpr int NUM_PT = 12;
    static constexpr int PIECE_INPUTS = HalfKAv2::NUM_SQ * NUM_PT;
    static constexpr int INPUTS = HalfKAv2::INPUTS + PIECE_INPUTS;

    static constexpr int MAX_PIECE_FEATURES = 32;
    static constexpr int MAX_ACTIVE_FEATURES = HalfKAv2::MAX_ACTIVE_FEATURES + MAX_PIECE_FEATURES;

    static std::pair<int, int> fill_features_sparse(const TrainingDataEntry& e, int* features, float* values, Color color)
    {
        const auto [start_j, offset] = HalfKAv2::fill_features_sparse(e, features, values, color);
        auto& pos = e.pos;
        auto pieces = pos.piecesBB();

        int j = start_j;
        for(Square sq : pieces)
        {
            auto p = pos.pieceAt(sq);
            auto p_idx = static_cast<int>(p.type()) * 2 + (p.color() != color);
            values[j] = 1.0f;
            features[j] = offset + (p_idx * HalfKAv2::NUM_SQ) + static_cast<int>(orient_flip(color, sq));
            ++j;
        }

        return { j, INPUTS };
    }
};

// ksq must not be oriented
static Square orient_flip_2(Color color, Square sq, Square ksq)
{
    bool h = ksq.file() < fileE;
    if (color == Color::Black)
        sq = sq.flippedVertically();
    if (h)
        sq = sq.flippedHorizontally();
    return sq;
}

struct HalfKAv2_hm {
    static constexpr int NUM_SQ = 64;
    static constexpr int NUM_PT = 11;
    static constexpr int NUM_PLANES = NUM_SQ * NUM_PT;
    static constexpr int INPUTS = NUM_PLANES * NUM_SQ / 2;

    static constexpr int MAX_ACTIVE_FEATURES = 32;

    static constexpr int KingBuckets[64] = {
      -1, -1, -1, -1, 31, 30, 29, 28,
      -1, -1, -1, -1, 27, 26, 25, 24,
      -1, -1, -1, -1, 23, 22, 21, 20,
      -1, -1, -1, -1, 19, 18, 17, 16,
      -1, -1, -1, -1, 15, 14, 13, 12,
      -1, -1, -1, -1, 11, 10, 9, 8,
      -1, -1, -1, -1, 7, 6, 5, 4,
      -1, -1, -1, -1, 3, 2, 1, 0
    };

    static int feature_index(Color color, Square ksq, Square sq, Piece p)
    {
        Square o_ksq = orient_flip_2(color, ksq, ksq);
        auto p_idx = static_cast<int>(p.type()) * 2 + (p.color() != color);
        if (p_idx == 11)
            --p_idx; // pack the opposite king into the same NUM_SQ * NUM_SQ
        return static_cast<int>(orient_flip_2(color, sq, ksq)) + p_idx * NUM_SQ + KingBuckets[static_cast<int>(o_ksq)] * NUM_PLANES;
    }

    static std::pair<int, int> fill_features_sparse(const TrainingDataEntry& e, int* features, float* values, Color color)
    {
        auto& pos = e.pos;
        auto pieces = pos.piecesBB();
        auto ksq = pos.kingSquare(color);

        int j = 0;
        for(Square sq : pieces)
        {
            auto p = pos.pieceAt(sq);
            values[j] = 1.0f;
            features[j] = feature_index(color, ksq, sq, p);
            ++j;
        }

        return { j, INPUTS };
    }
};

struct HalfKAv2_hmFactorized {
    // Factorized features
    static constexpr int NUM_PT = 12;
    static constexpr int PIECE_INPUTS = HalfKAv2_hm::NUM_SQ * NUM_PT;
    static constexpr int INPUTS = HalfKAv2_hm::INPUTS + PIECE_INPUTS;

    static constexpr int MAX_PIECE_FEATURES = 32;
    static constexpr int MAX_ACTIVE_FEATURES = HalfKAv2_hm::MAX_ACTIVE_FEATURES + MAX_PIECE_FEATURES;

    static std::pair<int, int> fill_features_sparse(const TrainingDataEntry& e, int* features, float* values, Color color)
    {
        const auto [start_j, offset] = HalfKAv2_hm::fill_features_sparse(e, features, values, color);
        auto& pos = e.pos;
        auto pieces = pos.piecesBB();
        auto ksq = pos.kingSquare(color);

        int j = start_j;
        for(Square sq : pieces)
        {
            auto p = pos.pieceAt(sq);
            auto p_idx = static_cast<int>(p.type()) * 2 + (p.color() != color);
            values[j] = 1.0f;
            features[j] = offset + (p_idx * HalfKAv2_hm::NUM_SQ) + static_cast<int>(orient_flip_2(color, sq, ksq));
            ++j;
        }

        return { j, INPUTS };
    }
};

template <typename T, typename... Ts>
struct FeatureSet
{
    static_assert(sizeof...(Ts) == 0, "Currently only one feature subset supported.");

    static constexpr int INPUTS = T::INPUTS;
    static constexpr int MAX_ACTIVE_FEATURES = T::MAX_ACTIVE_FEATURES;

    static std::pair<int, int> fill_features_sparse(const TrainingDataEntry& e, int* features, float* values, Color color)
    {
        return T::fill_features_sparse(e, features, values, color);
    }
};

struct SparseBatch
{
    static constexpr bool IS_BATCH = true;

    template <typename... Ts>
    SparseBatch(FeatureSet<Ts...>, const std::vector<TrainingDataEntry>& entries)
    {
        num_inputs = FeatureSet<Ts...>::INPUTS;
        size = entries.size();
        is_white = new float[size];
        outcome = new float[size];
        score = new float[size];
        white = new int[size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES];
        black = new int[size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES];
        white_values = new float[size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES];
        black_values = new float[size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES];
        psqt_indices = new int[size];
        layer_stack_indices = new int[size];

        num_active_white_features = 0;
        num_active_black_features = 0;
        max_active_features = FeatureSet<Ts...>::MAX_ACTIVE_FEATURES;

        for (std::size_t i = 0; i < size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES; ++i)
            white[i] = -1;
        for (std::size_t i = 0; i < size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES; ++i)
            black[i] = -1;
        for (std::size_t i = 0; i < size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES; ++i)
            white_values[i] = 0.0f;
        for (std::size_t i = 0; i < size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES; ++i)
            black_values[i] = 0.0f;

        for(int i = 0; i < entries.size(); ++i)
        {
            fill_entry(FeatureSet<Ts...>{}, i, entries[i]);
        }
    }

    int num_inputs;
    int size;

    float* is_white;
    float* outcome;
    float* score;
    int num_active_white_features;
    int num_active_black_features;
    int max_active_features;
    int* white;
    int* black;
    float* white_values;
    float* black_values;
    int* psqt_indices;
    int* layer_stack_indices;

    ~SparseBatch()
    {
        delete[] is_white;
        delete[] outcome;
        delete[] score;
        delete[] white;
        delete[] black;
        delete[] white_values;
        delete[] black_values;
        delete[] psqt_indices;
        delete[] layer_stack_indices;
    }

private:

    template <typename... Ts>
    void fill_entry(FeatureSet<Ts...>, int i, const TrainingDataEntry& e)
    {
        is_white[i] = static_cast<float>(e.pos.sideToMove() == Color::White);
        outcome[i] = (e.result + 1.0f) / 2.0f;
        score[i] = e.score;
        psqt_indices[i] = (e.pos.piecesBB().count() - 1) / 4;
        layer_stack_indices[i] = psqt_indices[i];
        fill_features(FeatureSet<Ts...>{}, i, e);
    }

    template <typename... Ts>
    void fill_features(FeatureSet<Ts...>, int i, const TrainingDataEntry& e)
    {
        const int offset = i * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES;
        num_active_white_features +=
            FeatureSet<Ts...>::fill_features_sparse(e, white + offset, white_values + offset, Color::White)
            .first;
        num_active_black_features +=
            FeatureSet<Ts...>::fill_features_sparse(e, black + offset, black_values + offset, Color::Black)
            .first;
    }
};

struct AnyStream
{
    virtual ~AnyStream() = default;
};

template <typename StorageT>
struct Stream : AnyStream
{
    using StorageType = StorageT;

    Stream(int concurrency, const std::vector<std::string>& filenames, bool cyclic, std::function<bool(const TrainingDataEntry&)> skipPredicate) :
        m_stream(training_data::open_sfen_input_file_parallel(concurrency, filenames, cyclic, skipPredicate))
    {
    }

    virtual StorageT* next() = 0;

protected:
    std::unique_ptr<training_data::BasicSfenInputStream> m_stream;
};

template <typename StorageT>
struct AsyncStream : Stream<StorageT>
{
    using BaseType = Stream<StorageT>;

    AsyncStream(int concurrency, const std::vector<std::string>& filenames, bool cyclic, std::function<bool(const TrainingDataEntry&)> skipPredicate) :
        BaseType(1, filenames, cyclic, skipPredicate)
    {
    }

    ~AsyncStream()
    {
        if (m_next.valid())
        {
            delete m_next.get();
        }
    }

protected:
    std::future<StorageT*> m_next;
};

template <typename FeatureSetT, typename StorageT>
struct FeaturedBatchStream : Stream<StorageT>
{
    static_assert(StorageT::IS_BATCH);

    using FeatureSet = FeatureSetT;
    using BaseType = Stream<StorageT>;

    static constexpr int num_feature_threads_per_reading_thread = 2;

    FeaturedBatchStream(int concurrency, const std::vector<std::string>& filenames, int batch_size, bool cyclic, std::function<bool(const TrainingDataEntry&)> skipPredicate) :
        BaseType(
            std::max(
                1,
                concurrency / num_feature_threads_per_reading_thread
            ),
            filenames,
            cyclic,
            skipPredicate
        ),
        m_concurrency(concurrency),
        m_batch_size(batch_size)
    {
        m_stop_flag.store(false);

        auto worker = [this]()
        {
            std::vector<TrainingDataEntry> entries;
            entries.reserve(m_batch_size);

            while(!m_stop_flag.load())
            {
                entries.clear();

                {
                    std::unique_lock lock(m_stream_mutex);
                    BaseType::m_stream->fill(entries, m_batch_size);
                    if (entries.empty())
                    {
                        break;
                    }
                }

                auto batch = new StorageT(FeatureSet{}, entries);

                {
                    std::unique_lock lock(m_batch_mutex);
                    m_batches_not_full.wait(lock, [this]() { return m_batches.size() < m_concurrency + 1 || m_stop_flag.load(); });

                    m_batches.emplace_back(batch);

                    lock.unlock();
                    m_batches_any.notify_one();
                }

            }
            m_num_workers.fetch_sub(1);
            m_batches_any.notify_one();
        };

        const int num_feature_threads = std::max(
            1,
            concurrency - std::max(1, concurrency / num_feature_threads_per_reading_thread)
        );

        for (int i = 0; i < num_feature_threads; ++i)
        {
            m_workers.emplace_back(worker);

            // This cannot be done in the thread worker. We need
            // to have a guarantee that this is incremented, but if
            // we did it in the worker there's no guarantee
            // that it executed.
            m_num_workers.fetch_add(1);
        }
    }

    StorageT* next() override
    {
        std::unique_lock lock(m_batch_mutex);
        m_batches_any.wait(lock, [this]() { return !m_batches.empty() || m_num_workers.load() == 0; });

        if (!m_batches.empty())
        {
            auto batch = m_batches.front();
            m_batches.pop_front();

            lock.unlock();
            m_batches_not_full.notify_one();

            return batch;
        }
        return nullptr;
    }

    ~FeaturedBatchStream()
    {
        m_stop_flag.store(true);
        m_batches_not_full.notify_all();

        for (auto& worker : m_workers)
        {
            if (worker.joinable())
            {
                worker.join();
            }
        }

        for (auto& batch : m_batches)
        {
            delete batch;
        }
    }

private:
    int m_batch_size;
    int m_concurrency;
    std::deque<StorageT*> m_batches;
    std::mutex m_batch_mutex;
    std::mutex m_stream_mutex;
    std::condition_variable m_batches_not_full;
    std::condition_variable m_batches_any;
    std::atomic_bool m_stop_flag;
    std::atomic_int m_num_workers;

    std::vector<std::thread> m_workers;
};

// Very simple fixed size string wrapper with a stable ABI to pass to python.
struct Fen
{
    Fen() :
        m_fen(nullptr)
    {
    }

    Fen(const std::string& fen) :
        m_size(fen.size()),
        m_fen(new char[fen.size() + 1])
    {
        std::memcpy(m_fen, fen.c_str(), fen.size() + 1);
    }

    Fen& operator=(const std::string& fen)
    {
        if (m_fen != nullptr)
        {
            delete m_fen;
        }

        m_size = fen.size();
        m_fen = new char[fen.size() + 1];
        std::memcpy(m_fen, fen.c_str(), fen.size() + 1);

        return *this;
    }

    ~Fen()
    {
        delete[] m_fen;
    }

    int m_size;
    char* m_fen;
};

struct FenBatch
{
    FenBatch(const std::vector<TrainingDataEntry>& entries) :
        m_size(entries.size()),
        m_fens(new Fen[entries.size()])
    {
        for (int i = 0; i < m_size; ++i)
        {
            m_fens[i] = entries[i].pos.fen();
        }
    }

    ~FenBatch()
    {
        delete[] m_fens;
    }

    int m_size;
    Fen* m_fens;
};

struct FenBatchStream : Stream<FenBatch>
{
    static constexpr int num_feature_threads_per_reading_thread = 2;

    using BaseType = Stream<FenBatch>;

    FenBatchStream(int concurrency, const std::vector<std::string>& filenames, int batch_size, bool cyclic, std::function<bool(const TrainingDataEntry&)> skipPredicate) :
        BaseType(
            std::max(
                1,
                concurrency / num_feature_threads_per_reading_thread
            ),
            filenames,
            cyclic,
            skipPredicate
        ),
        m_concurrency(concurrency),
        m_batch_size(batch_size)
    {
        m_stop_flag.store(false);

        auto worker = [this]()
        {
            std::vector<TrainingDataEntry> entries;
            entries.reserve(m_batch_size);

            while(!m_stop_flag.load())
            {
                entries.clear();

                {
                    std::unique_lock lock(m_stream_mutex);
                    BaseType::m_stream->fill(entries, m_batch_size);
                    if (entries.empty())
                    {
                        break;
                    }
                }

                auto batch = new FenBatch(entries);

                {
                    std::unique_lock lock(m_batch_mutex);
                    m_batches_not_full.wait(lock, [this]() { return m_batches.size() < m_concurrency + 1 || m_stop_flag.load(); });

                    m_batches.emplace_back(batch);

                    lock.unlock();
                    m_batches_any.notify_one();
                }

            }
            m_num_workers.fetch_sub(1);
            m_batches_any.notify_one();
        };

        const int num_feature_threads = std::max(
            1,
            concurrency - std::max(1, concurrency / num_feature_threads_per_reading_thread)
        );

        for (int i = 0; i < num_feature_threads; ++i)
        {
            m_workers.emplace_back(worker);

            // This cannot be done in the thread worker. We need
            // to have a guarantee that this is incremented, but if
            // we did it in the worker there's no guarantee
            // that it executed.
            m_num_workers.fetch_add(1);
        }
    }

    FenBatch* next()
    {
        std::unique_lock lock(m_batch_mutex);
        m_batches_any.wait(lock, [this]() { return !m_batches.empty() || m_num_workers.load() == 0; });

        if (!m_batches.empty())
        {
            auto batch = m_batches.front();
            m_batches.pop_front();

            lock.unlock();
            m_batches_not_full.notify_one();

            return batch;
        }
        return nullptr;
    }

    ~FenBatchStream()
    {
        m_stop_flag.store(true);
        m_batches_not_full.notify_all();

        for (auto& worker : m_workers)
        {
            if (worker.joinable())
            {
                worker.join();
            }
        }

        for (auto& batch : m_batches)
        {
            delete batch;
        }
    }

private:
    int m_batch_size;
    int m_concurrency;
    std::deque<FenBatch*> m_batches;
    std::mutex m_batch_mutex;
    std::mutex m_stream_mutex;
    std::condition_variable m_batches_not_full;
    std::condition_variable m_batches_any;
    std::atomic_bool m_stop_flag;
    std::atomic_int m_num_workers;

    std::vector<std::thread> m_workers;
};

std::function<bool(const TrainingDataEntry&)> make_skip_predicate(bool filtered, int random_fen_skipping, bool wld_filtered, int early_fen_skipping, int param_index)
{
    if (filtered || random_fen_skipping || wld_filtered || early_fen_skipping)
    {
        return [
            random_fen_skipping,
            prob = double(random_fen_skipping) / (random_fen_skipping + 1),
            filtered,
            wld_filtered,
            early_fen_skipping
            ](const TrainingDataEntry& e){

            // VALUE_NONE from Stockfish.
            // We need to allow a way to skip predetermined positions without
            // having to remove them from the dataset, as otherwise the we lose some
            // compression ability.
            static constexpr int VALUE_NONE = 32002;

            static constexpr double desired_piece_count_weights[33] = {
                1.000000,
                1.121094, 1.234375, 1.339844, 1.437500, 1.527344, 1.609375, 1.683594, 1.750000,
                1.808594, 1.859375, 1.902344, 1.937500, 1.964844, 1.984375, 1.996094, 2.000000,
                1.996094, 1.984375, 1.964844, 1.937500, 1.902344, 1.859375, 1.808594, 1.750000,
                1.683594, 1.609375, 1.527344, 1.437500, 1.339844, 1.234375, 1.121094, 1.000000
            };

            static constexpr double desired_piece_count_weights_total = [](){
                double tot = 0;
                for (auto w : desired_piece_count_weights)
                    tot += w;
                return tot;
            }();

            static thread_local std::mt19937 gen(std::random_device{}());

            // keep stats on passing pieces
            static thread_local double alpha = 1;
            static thread_local double piece_count_history_all[33] = {0};
            static thread_local double piece_count_history_passed[33] = {0};
            static thread_local double piece_count_history_all_total = 0;
            static thread_local double piece_count_history_passed_total = 0;

            // max skipping rate
            static constexpr double max_skipping_rate = 10.0;

            auto do_wld_skip = [&]() {
                std::bernoulli_distribution distrib(1.0 - e.score_result_prob());
                auto& prng = rng::get_thread_local_rng();
                return distrib(prng);
            };

            auto do_skip = [&]() {
                std::bernoulli_distribution distrib(prob);
                auto& prng = rng::get_thread_local_rng();
                return distrib(prng);
            };

            auto do_filter = [&]() {
                return (e.isCapturingMove() || e.isInCheck());
            };

            // Allow for predermined filtering without the need to remove positions from the dataset.
            if (e.score == VALUE_NONE)
                return true;

            if (e.ply <= early_fen_skipping)
                return true;

            if (random_fen_skipping && do_skip())
                return true;

            if (filtered && do_filter())
                return true;

            if (wld_filtered && do_wld_skip())
                return true;

            constexpr bool do_debug_print = false;
            if (do_debug_print) {
                if (uint64_t(piece_count_history_all_total) % 10000 == 0) {
                    std::cout << "Total : " << piece_count_history_all_total << '\n';
                    std::cout << "Passed: " << piece_count_history_passed_total << '\n';
                    for (int i = 0; i < 33; ++i)
                        std::cout << i << ' ' << piece_count_history_passed[i] << '\n';
                }
            }

            const int pc = e.pos.piecesBB().count();
            piece_count_history_all[pc] += 1;
            piece_count_history_all_total += 1;

            // update alpha, which scales the filtering probability, to a maximum rate.
            if (uint64_t(piece_count_history_all_total) % 10000 == 0) {
                double pass = piece_count_history_all_total * desired_piece_count_weights_total;
                for (int i = 0; i < 33; ++i)
                {
                    if (desired_piece_count_weights[pc] > 0)
                    {
                        double tmp = piece_count_history_all_total * desired_piece_count_weights[pc] /
                                     (desired_piece_count_weights_total * piece_count_history_all[pc]);
                        if (tmp < pass)
                            pass = tmp;
                    }
                }
                alpha = 1.0 / (pass * max_skipping_rate);
            }

            double tmp = alpha *  piece_count_history_all_total * desired_piece_count_weights[pc] /
                                 (desired_piece_count_weights_total * piece_count_history_all[pc]);
            tmp = std::min(1.0, tmp);
            std::bernoulli_distribution distrib(1.0 - tmp);
            auto& prng = rng::get_thread_local_rng();
            if (distrib(prng))
                return true;

            piece_count_history_passed[pc] += 1;
            piece_count_history_passed_total += 1;

            return false;
        };
    }

    return nullptr;
}

namespace py = pybind11;
using namespace py::literals;

PYBIND11_MODULE(_core, m) {
    m.doc() = "NNUE dataset Python bindings using pybind11";

    py::class_<Stream<FenBatch>>(m, "FenBatchStreamBase")
        .def("next", &Stream<FenBatch>::next,
             py::return_value_policy::reference);

             py::class_<SparseBatch>(m, "SparseBatch")
             .def_readonly("num_inputs", &SparseBatch::num_inputs)
             .def_readonly("size", &SparseBatch::size)
             .def_readonly("num_active_white_features", &SparseBatch::num_active_white_features)
             .def_readonly("num_active_black_features", &SparseBatch::num_active_black_features)
             .def_readonly("max_active_features", &SparseBatch::max_active_features)
             .def_property_readonly("is_white", [](const SparseBatch& self) {
                 return py::array_t<float>(
                     {self.size},                   // shape
                     {sizeof(float)},               // strides
                     self.is_white,                 // data pointer
                     py::capsule([]() {})           // no ownership (data managed by SparseBatch)
                 );
             })
             .def_property_readonly("outcome", [](const SparseBatch& self) {
                 return py::array_t<float>(
                     {self.size},
                     {sizeof(float)},
                     self.outcome,
                     py::capsule([]() {})
                 );
             })
             .def_property_readonly("score", [](const SparseBatch& self) {
                 return py::array_t<float>(
                     {self.size},
                     {sizeof(float)},
                     self.score,
                     py::capsule([]() {})
                 );
             })
             .def_property_readonly("white", [](const SparseBatch& self) {
                 return py::array_t<int>(
                     {self.size, self.max_active_features},
                     {static_cast<size_t>(self.max_active_features) * sizeof(int), sizeof(int)},
                     self.white,
                     py::capsule([]() {})
                 );
             })
             .def_property_readonly("black", [](const SparseBatch& self) {
                 return py::array_t<int>(
                     {self.size, self.max_active_features},
                     {static_cast<size_t>(self.max_active_features) * sizeof(int), sizeof(int)},
                     self.black,
                     py::capsule([]() {})
                 );
             })
             .def_property_readonly("white_values", [](const SparseBatch& self) {
                 return py::array_t<float>(
                     {self.size, self.max_active_features},
                     {static_cast<size_t>(self.max_active_features) * sizeof(float), sizeof(float)},
                     self.white_values,
                     py::capsule([]() {})
                 );
             })
             .def_property_readonly("black_values", [](const SparseBatch& self) {
                 return py::array_t<float>(
                     {self.size, self.max_active_features},
                     {static_cast<size_t>(self.max_active_features) * sizeof(float), sizeof(float)},
                     self.black_values,
                     py::capsule([]() {})
                 );
             })
             .def_property_readonly("psqt_indices", [](const SparseBatch& self) {
                 return py::array_t<int>(
                     {self.size},
                     {sizeof(int)},
                     self.psqt_indices,
                     py::capsule([]() {})
                 );
             })
             .def_property_readonly("layer_stack_indices", [](const SparseBatch& self) {
                 return py::array_t<int>(
                     {self.size},
                     {sizeof(int)},
                     self.layer_stack_indices,
                     py::capsule([]() {})
                 );
             })
             .def("get_tensors", [](const SparseBatch& self, py::object device) {
                 py::module torch = py::module::import("torch");
                 
                 auto white_values_arr = py::array_t<float>(
                     {self.size, self.max_active_features},
                     {static_cast<size_t>(self.max_active_features) * sizeof(float), sizeof(float)},
                     self.white_values
                 );
                 
                 auto black_values_arr = py::array_t<float>(
                     {self.size, self.max_active_features},
                     {static_cast<size_t>(self.max_active_features) * sizeof(float), sizeof(float)},
                     self.black_values
                 );
                 
                 auto white_indices_arr = py::array_t<int>(
                     {self.size, self.max_active_features},
                     {static_cast<size_t>(self.max_active_features) * sizeof(int), sizeof(int)},
                     self.white
                 );
                 
                 auto black_indices_arr = py::array_t<int>(
                     {self.size, self.max_active_features},
                     {static_cast<size_t>(self.max_active_features) * sizeof(int), sizeof(int)},
                     self.black
                 );
                 
                 auto is_white_arr = py::array_t<float>(
                     {self.size, 1},
                     {sizeof(float), sizeof(float)},
                     self.is_white
                 );
                 
                 auto outcome_arr = py::array_t<float>(
                     {self.size, 1},
                     {sizeof(float), sizeof(float)},
                     self.outcome
                 );
                 
                 auto score_arr = py::array_t<float>(
                     {self.size, 1},
                     {sizeof(float), sizeof(float)},
                     self.score
                 );
                 
                 auto psqt_indices_arr = py::array_t<int>(
                     {self.size},
                     {sizeof(int)},
                     self.psqt_indices
                 );
                 
                 auto layer_stack_indices_arr = py::array_t<int>(
                     {self.size},
                     {sizeof(int)},
                     self.layer_stack_indices
                 );
                 
                 // Convert numpy arrays to torch tensors
                 py::object white_values = torch.attr("from_numpy")(white_values_arr).attr("pin_memory")().attr("to")(device, "non_blocking"_a=true);
                 py::object black_values = torch.attr("from_numpy")(black_values_arr).attr("pin_memory")().attr("to")(device, "non_blocking"_a=true);
                 py::object white_indices = torch.attr("from_numpy")(white_indices_arr).attr("pin_memory")().attr("to")(device, "non_blocking"_a=true);
                 py::object black_indices = torch.attr("from_numpy")(black_indices_arr).attr("pin_memory")().attr("to")(device, "non_blocking"_a=true);
                 py::object us = torch.attr("from_numpy")(is_white_arr).attr("pin_memory")().attr("to")(device, "non_blocking"_a=true);
                 py::object them = torch.attr("tensor")(1.0).attr("sub")(us);
                 py::object outcome = torch.attr("from_numpy")(outcome_arr).attr("pin_memory")().attr("to")(device, "non_blocking"_a=true);
                 py::object score = torch.attr("from_numpy")(score_arr).attr("pin_memory")().attr("to")(device, "non_blocking"_a=true);
                 py::object psqt_indices = torch.attr("from_numpy")(psqt_indices_arr).attr("long")().attr("pin_memory")().attr("to")(device, "non_blocking"_a=true);
                 py::object layer_stack_indices = torch.attr("from_numpy")(layer_stack_indices_arr).attr("long")().attr("pin_memory")().attr("to")(device, "non_blocking"_a=true);

                 return py::make_tuple(us, them, white_indices, white_values, black_indices, black_values, outcome, score, psqt_indices, layer_stack_indices);
             });

    py::class_<FenBatch>(m, "FenBatch")
        .def("__repr__", [](const FenBatch &self) { return "FenBatch object"; })
        .def_readonly("size", &FenBatch::m_size)
        .def("get_fens", [](const FenBatch &self) {
            std::vector<std::string> fens;
            for (int i = 0; i < self.m_size; i++) {
                fens.push_back(self.m_fens[i].m_fen);
            }
            return fens;
        });

    py::class_<Stream<SparseBatch>>(m, "SparseBatchStream")
        .def("next", &Stream<SparseBatch>::next,
             py::return_value_policy::reference);

    py::class_<FenBatchStream, Stream<FenBatch>>(m, "FenBatchStream")
        .def("next", &FenBatchStream::next, py::return_value_policy::reference);

    m.def(
        "get_sparse_batch_from_fens",
        [](const std::string &feature_set, const std::vector<std::string> &fens,
           std::vector<int> &scores, std::vector<int> &plies,
           std::vector<int> &results) -> SparseBatch * {
            if (fens.size() != scores.size() || fens.size() != plies.size() ||
                fens.size() != results.size()) {
                throw std::runtime_error(
                    "Inconsistent input sizes for fens, scores, plies, and "
                    "results");
            }

            std::vector<TrainingDataEntry> entries;
            entries.reserve(fens.size());

            for (size_t i = 0; i < fens.size(); ++i) {
                auto &e = entries.emplace_back();
                e.pos = Position::fromFen(fens[i].c_str());
                movegen::forEachLegalMove(e.pos, [&](Move m) { e.move = m; });
                e.score = scores[i];
                e.ply = plies[i];
                e.result = results[i];
            }

            SparseBatch *batch = nullptr;

            if (feature_set == "HalfKP") {
                batch = new SparseBatch(FeatureSet<HalfKP>{}, entries);
            } else if (feature_set == "HalfKP^") {
                batch =
                    new SparseBatch(FeatureSet<HalfKPFactorized>{}, entries);
            } else if (feature_set == "HalfKA") {
                batch = new SparseBatch(FeatureSet<HalfKA>{}, entries);
            } else if (feature_set == "HalfKA^") {
                batch =
                    new SparseBatch(FeatureSet<HalfKAFactorized>{}, entries);
            } else if (feature_set == "HalfKAv2") {
                batch = new SparseBatch(FeatureSet<HalfKAv2>{}, entries);
            } else if (feature_set == "HalfKAv2^") {
                batch =
                    new SparseBatch(FeatureSet<HalfKAv2Factorized>{}, entries);
            } else if (feature_set == "HalfKAv2_hm") {
                batch = new SparseBatch(FeatureSet<HalfKAv2_hm>{}, entries);
            } else if (feature_set == "HalfKAv2_hm^") {
                batch = new SparseBatch(FeatureSet<HalfKAv2_hmFactorized>{},
                                        entries);
            } else {
                throw std::runtime_error("Unknown feature_set: " + feature_set);
            }

            return batch;
        },
        py::arg("feature_set"), py::arg("fens"), py::arg("scores"),
        py::arg("plies"), py::arg("results"),
        py::return_value_policy::take_ownership,
        "Create a SparseBatch from a list of FENs with associated scores, "
        "plies, "
        "and results");

    m.def(
        "create_fen_batch_stream",
        [](int concurrency, const std::vector<std::string> &filenames,
           int batch_size, bool cyclic, bool filtered = false,
           int random_fen_skipping = 0, bool wld_filtered = false,
           int early_fen_skipping = 0,
           int param_index = 0) -> FenBatchStream * {
            auto skipPredicate =
                make_skip_predicate(filtered, random_fen_skipping, wld_filtered,
                                    early_fen_skipping, param_index);
            return new FenBatchStream(concurrency, filenames, batch_size,
                                      cyclic, skipPredicate);
        },
        py::arg("concurrency"), py::arg("filenames"), py::arg("batch_size"),
        py::arg("cyclic"), py::arg("filtered") = false,
        py::arg("random_fen_skipping") = 0, py::arg("wld_filtered") = false,
        py::arg("early_fen_skipping") = 0, py::arg("param_index") = 0,
        py::return_value_policy::take_ownership,
        "Create a FenBatchStream from a list of filenames");

    m.def(
        "create_sparse_batch_stream",
        [](const std::string &feature_set, int concurrency,
           const std::vector<std::string> &filenames, int batch_size,
           bool cyclic, bool filtered = false, int random_fen_skipping = 0,
           bool wld_filtered = false, int early_fen_skipping = 0,
           int param_index = 0) -> Stream<SparseBatch> * {
            auto skipPredicate =
                make_skip_predicate(filtered, random_fen_skipping, wld_filtered,
                                    early_fen_skipping, param_index);

            Stream<SparseBatch> *stream = nullptr;

            if (feature_set == "HalfKP") {
                stream =
                    new FeaturedBatchStream<FeatureSet<HalfKP>, SparseBatch>(
                        concurrency, filenames, batch_size, cyclic,
                        skipPredicate);
            } else if (feature_set == "HalfKP^") {
                stream = new FeaturedBatchStream<FeatureSet<HalfKPFactorized>,
                                                 SparseBatch>(
                    concurrency, filenames, batch_size, cyclic, skipPredicate);
            } else if (feature_set == "HalfKA") {
                stream =
                    new FeaturedBatchStream<FeatureSet<HalfKA>, SparseBatch>(
                        concurrency, filenames, batch_size, cyclic,
                        skipPredicate);
            } else if (feature_set == "HalfKA^") {
                stream = new FeaturedBatchStream<FeatureSet<HalfKAFactorized>,
                                                 SparseBatch>(
                    concurrency, filenames, batch_size, cyclic, skipPredicate);
            } else if (feature_set == "HalfKAv2") {
                stream =
                    new FeaturedBatchStream<FeatureSet<HalfKAv2>, SparseBatch>(
                        concurrency, filenames, batch_size, cyclic,
                        skipPredicate);
            } else if (feature_set == "HalfKAv2^") {
                stream = new FeaturedBatchStream<FeatureSet<HalfKAv2Factorized>,
                                                 SparseBatch>(
                    concurrency, filenames, batch_size, cyclic, skipPredicate);
            } else if (feature_set == "HalfKAv2_hm") {
                stream = new FeaturedBatchStream<FeatureSet<HalfKAv2_hm>,
                                                 SparseBatch>(
                    concurrency, filenames, batch_size, cyclic, skipPredicate);
            } else if (feature_set == "HalfKAv2_hm^") {
                stream =
                    new FeaturedBatchStream<FeatureSet<HalfKAv2_hmFactorized>,
                                            SparseBatch>(concurrency, filenames,
                                                         batch_size, cyclic,
                                                         skipPredicate);
            } else {
                throw std::runtime_error("Unknown feature_set: " + feature_set);
            }

            return stream;
        },
        py::arg("feature_set"), py::arg("concurrency"), py::arg("filenames"),
        py::arg("batch_size"), py::arg("cyclic"), py::arg("filtered") = false,
        py::arg("random_fen_skipping") = 0, py::arg("wld_filtered") = false,
        py::arg("early_fen_skipping") = 0, py::arg("param_index") = 0,
        py::return_value_policy::take_ownership,
        "Create a SparseBatchStream from a list of filenames");

    m.def(
        "destroy_sparse_batch", [](SparseBatch *batch) { delete batch; },
        "Destroy a SparseBatch object");

    m.def(
        "destroy_fen_batch", [](FenBatch *batch) { delete batch; },
        "Destroy a FenBatch object");

    m.def(
        "destroy_sparse_batch_stream",
        [](Stream<SparseBatch> *stream) { delete stream; },
        "Destroy a SparseBatchStream object");

    m.def(
        "destroy_fen_batch_stream",
        [](FenBatchStream *stream) { delete stream; },
        "Destroy a FenBatchStream object");

    m.def(
        "fetch_next_sparse_batch",
        [](Stream<SparseBatch> *stream) -> SparseBatch * {
            return stream->next();
        },
        py::return_value_policy::reference,
        "Fetch the next SparseBatch from a stream");

    m.def(
        "fetch_next_fen_batch",
        [](Stream<FenBatch> *stream) -> FenBatch * { return stream->next(); },
        py::return_value_policy::reference,
        "Fetch the next FenBatch from a stream");
}

/* benches */ /*
#include <chrono>

int main()
{
    auto stream = create_sparse_batch_stream("HalfKP", 4, { "10m_d3_q_2.binpack" }, 8192, true, false, 0, false, -1, 0);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i)
    {
        if (i % 100 == 0) std::cout << i << '\n';
        destroy_sparse_batch(stream->next());
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << (t1 - t0).count() / 1e9 << "s\n";
}
//*/
