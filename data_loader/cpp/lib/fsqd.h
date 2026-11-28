#pragma once

// Reader for FSQD ("four-square") files: prefiltered training positions, one
// rANS-coded record per position, in indexed blocks of up to 256 records.
// The format is specified in four-square/codec.md and four-square/container.md;
// the reference implementation is the Rust crate in four-square/.
//
// The loader reads an FSQD file in chunks of up to kChunkBlocks consecutive
// blocks of one segment (about 1 MB). FsqdFile reads a chunk's index entries
// and payloads; FsqdChunkReader decodes it into TrainingDataEntry values.
//
// FSQD stores no ply and no move. Decoded entries get ply 0 and a null move,
// except search-tree samples (outcome "no result"), which get ply 9999 and
// result 0 as in the binpacks they come from.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "chess.h"
#include "training_data_entry.h"

namespace fsqd
{
    constexpr int GROUPS = 16;
    constexpr int PHASES = 8;
    constexpr int MAX_DICT = 127;
    constexpr std::uint32_t ESCAPE = 127;
    constexpr int RESULTS = 4;  // loss, draw, win, no result
    constexpr std::uint32_t NO_RESULT = 3;

    // Table layout in codec order.
    constexpr int T_WK = 0;                          // 64
    constexpr int T_BK = T_WK + 1;                   // by white king rank, 64
    constexpr int T_PHASE = T_BK + 8;                // 8
    constexpr int T_TOKEN = T_PHASE + 1;             // phase x group, 128
    constexpr int T_MASK = T_TOKEN + PHASES * GROUPS; // phase x group, 16
    constexpr int T_LIT = T_MASK + PHASES * GROUPS;  // phase x square, 12
    constexpr int T_STATE = T_LIT + PHASES * 64;     // 32
    constexpr int T_EP = T_STATE + 1;                // 65
    constexpr int T_RESULT = T_EP + 1;               // material context, 4
    constexpr int T_SHIGH = T_RESULT + 11;           // context x result, 256
    constexpr int T_SLOW = T_SHIGH + 11 * RESULTS;   // score high byte, 256
    constexpr int N_TABLES = T_SLOW + 256;
    static_assert(T_LIT == 266 && T_SLOW == 835 && N_TABLES == 1091);

    constexpr std::uint32_t PROB_BITS = 10;
    constexpr std::uint32_t PROB_SCALE = 1u << PROB_BITS;
    constexpr std::uint32_t RANS_L = 1u << 23;

    constexpr std::uint64_t BLOCK_RECORDS = 256;
    constexpr std::uint64_t HEADER_LEN = 16;
    constexpr std::uint64_t FOOTER_LEN = 24;
    constexpr std::uint16_t VERSION = 1;
    constexpr int TREE_SAMPLE_PLY = 9999;
    constexpr std::uint8_t EMPTY = 12;

    // Signed material of each piece ID (white positive).
    constexpr int MATERIAL[13] = {1, -1, 3, -3, 3, -3, 5, -5, 9, -9, 0, 0, 0};

    constexpr int alphabet(int t)
    {
        if (t < T_PHASE) return 64;
        if (t < T_TOKEN) return 8;
        if (t < T_MASK) return 128;
        if (t < T_LIT) return 16;
        if (t < T_STATE) return 12;
        if (t < T_EP) return 32;
        if (t < T_RESULT) return 65;
        if (t < T_SHIGH) return RESULTS;
        return 256;
    }

    constexpr int total_symbols()
    {
        int n = 0;
        for (int t = 0; t < N_TABLES; ++t)
            n += alphabet(t);
        return n;
    }

    constexpr std::size_t max_model_len()
    {
        return GROUPS * (1 + 4 * MAX_DICT) + 2 * static_cast<std::size_t>(total_symbols());
    }

    // State plus at most two refill bytes for each of 3 + 16 * 6 + 5 symbols.
    constexpr std::uint64_t max_block_len(std::uint64_t records) { return 4 + 208 * records; }

    // Square of group g, k-th from the lowest rank. Group 2f covers ranks 1-4
    // of file f, group 2f + 1 ranks 5-8.
    constexpr int group_square(int g, int k) { return 8 * ((g & 1) * 4 + k) + (g >> 1); }

    inline std::uint32_t read_u32(const unsigned char* p)
    {
        return std::uint32_t(p[0]) | std::uint32_t(p[1]) << 8 | std::uint32_t(p[2]) << 16
             | std::uint32_t(p[3]) << 24;
    }

    inline std::uint64_t read_u64(const unsigned char* p)
    {
        return std::uint64_t(read_u32(p)) | std::uint64_t(read_u32(p + 4)) << 32;
    }

    struct Model
    {
        struct Word
        {
            std::uint8_t pieces[4];
            std::int8_t material;
            std::uint8_t count;  // non-empty squares
        };

        std::array<std::vector<Word>, GROUPS> dicts;
        // Decode slot: symbol (8 bits), frequency (11 bits), offset (10 bits).
        std::vector<std::array<std::uint32_t, PROB_SCALE>> decode;

        // Parse and validate a serialized model; throws on malformed input.
        explicit Model(const std::vector<unsigned char>& bytes)
        {
            auto fail = [](const std::string& msg) { throw std::runtime_error("FSQD model: " + msg); };
            std::size_t pos = 0;
            for (int g = 0; g < GROUPS; ++g)
            {
                if (pos >= bytes.size())
                    fail("data ends early");
                const std::size_t len = bytes[pos++];
                if (len > MAX_DICT)
                    fail("dictionary longer than 127 words");
                if (bytes.size() - pos < 4 * len)
                    fail("data ends early");
                std::vector<std::uint32_t> seen;
                for (std::size_t i = 0; i < len; ++i, pos += 4)
                {
                    const std::uint32_t w = read_u32(&bytes[pos]);
                    for (auto s : seen)
                        if (s == w)
                            fail("duplicate dictionary word");
                    seen.push_back(w);
                    Word word{};
                    int mat = 0;
                    for (int k = 0; k < 4; ++k)
                    {
                        const std::uint8_t p = bytes[pos + k];
                        if (p > 9 && p != EMPTY)
                            fail("dictionary word with a bad piece");
                        word.pieces[k] = p;
                        mat += MATERIAL[p];
                        word.count += p != EMPTY;
                    }
                    word.material = static_cast<std::int8_t>(mat);
                    dicts[g].push_back(word);
                }
            }
            if (bytes.size() - pos != 2 * static_cast<std::size_t>(total_symbols()))
                fail("wrong length");
            decode.resize(N_TABLES);
            for (int t = 0; t < N_TABLES; ++t)
            {
                std::uint32_t start = 0;
                for (int s = 0; s < alphabet(t); ++s, pos += 2)
                {
                    const std::uint32_t f = bytes[pos] | bytes[pos + 1] << 8;
                    if (f == 0 || start + f > PROB_SCALE)
                        fail("table " + std::to_string(t) + " is not positive with sum 1024");
                    for (std::uint32_t k = 0; k < f; ++k)
                        decode[t][start + k] = std::uint32_t(s) | f << 8 | k << 19;
                    start += f;
                }
                if (start != PROB_SCALE)
                    fail("table " + std::to_string(t) + " is not positive with sum 1024");
            }
        }
    };

    // Models of all files opened through one cache are shared when their
    // serialized bytes are identical.
    using ModelCache = std::map<std::vector<unsigned char>, std::shared_ptr<const Model>>;

    // A chunk handed from the reader to a decode worker. `data` holds
    // u32 block count, u32 records in the last block, (blocks + 1) u32 block
    // offsets into the payload, then the payload. The model outlives the
    // chunk: it is owned by the file, which outlives the loader threads.
    struct Chunk
    {
        const Model* model = nullptr;
        std::vector<unsigned char> data;

        void swap(Chunk& other) noexcept
        {
            std::swap(model, other.model);
            data.swap(other.data);
        }
    };

    struct DecodeError : std::runtime_error
    {
        using std::runtime_error::runtime_error;
    };

    // Decode `n` records of one block payload into `out` (cleared first).
    inline void decode_block(const Model& model, const unsigned char* data, std::size_t len, std::size_t n,
                             std::vector<binpack::TrainingDataEntry>& out)
    {
        using namespace chess;
        auto fail = [](const char* why) { throw DecodeError(std::string("FSQD block: ") + why); };

        out.clear();
        if (len < 4)
            fail("truncated");
        std::uint32_t x = read_u32(data);
        if (x < RANS_L || std::uint64_t(x) >= 256ull * RANS_L)
            fail("initial state out of range");
        std::size_t pos = 4;
        const auto* dec = model.decode.data();

        auto get = [&](int table) -> std::uint32_t {
            const std::uint32_t e = dec[table][x & (PROB_SCALE - 1)];
            x = ((e >> 8) & 0x7ff) * (x >> PROB_BITS) + (e >> 19);
            while (x < RANS_L)
            {
                if (pos >= len)
                    fail("truncated");
                x = x << 8 | data[pos++];
            }
            return e & 0xff;
        };

        for (std::size_t i = 0; i < n; ++i)
        {
            const int wk = static_cast<int>(get(T_WK));
            const int bk = static_cast<int>(get(T_BK + wk / 8));
            if (wk == bk)
                fail("king collision");
            const int phase = static_cast<int>(get(T_PHASE));
            std::uint8_t sq[64];
            std::memset(sq, EMPTY, sizeof(sq));
            int material = 0;
            int nonking = 0;
            for (int g = 0; g < GROUPS; ++g)
            {
                const int table = phase * GROUPS + g;
                const std::uint32_t tok = get(T_TOKEN + table);
                if (tok == ESCAPE)
                {
                    const std::uint32_t mask = get(T_MASK + table);
                    for (int k = 0; k < 4; ++k)
                    {
                        if (mask >> k & 1)
                        {
                            const std::uint32_t p = get(T_LIT + phase * 64 + 4 * g + k);
                            if (p >= 10)
                                fail("king literal");
                            sq[group_square(g, k)] = static_cast<std::uint8_t>(p);
                            material += MATERIAL[p];
                            ++nonking;
                        }
                    }
                }
                else
                {
                    const auto& dict = model.dicts[g];
                    if (tok >= dict.size())
                        fail("missing dictionary id");
                    const auto& w = dict[tok];
                    for (int k = 0; k < 4; ++k)
                        sq[group_square(g, k)] = w.pieces[k];
                    material += w.material;
                    nonking += w.count;
                }
            }
            if (sq[wk] != EMPTY || sq[bk] != EMPTY)
                fail("occupied king destination");
            if (std::min(7, nonking / 8) != phase)
                fail("phase does not match the board");

            const std::uint32_t state = get(T_STATE);
            const bool black = state & 1;
            const std::uint32_t ep = get(T_EP);
            const int m = std::clamp(black ? -material : material, -5, 5) + 5;
            const std::uint32_t r = get(T_RESULT + m);
            const std::uint32_t hi = get(T_SHIGH + m * RESULTS + static_cast<int>(r));
            const std::uint32_t lo = get(T_SLOW + static_cast<int>(hi));

            auto& e = out.emplace_back();
            e.pos = Position();
            e.pos.setCastlingRights(static_cast<CastlingRights>(state >> 1));
            e.pos.setSideToMove(black ? Color::Black : Color::White);
            e.pos.setEpSquareUnchecked(ep == 64 ? Square::none() : Square(static_cast<int>(ep)));
            for (int s = 0; s < 64; ++s)
                if (sq[s] != EMPTY)
                    e.pos.place(Piece::fromId(sq[s]), Square(s));
            e.pos.place(whiteKing, Square(wk));
            e.pos.place(blackKing, Square(bk));
            e.move = Move::null();
            e.score = static_cast<std::int16_t>(static_cast<std::uint16_t>(hi << 8 | lo));
            if (r == NO_RESULT)
            {
                e.ply = TREE_SAMPLE_PLY;
                e.result = 0;
            }
            else
            {
                e.ply = 0;
                e.result = static_cast<std::int16_t>(static_cast<int>(r) - 1);
            }
        }
        if (x != RANS_L || pos != len)
            fail("final state or trailing bytes");
    }

    // Iterates the entries of one Chunk, decoding a block at a time.
    struct ChunkReader
    {
        [[nodiscard]] bool hasNext(const Chunk& chunk)
        {
            while (m_offset >= m_entries.size())
            {
                if (!nextBlock(chunk))
                    return false;
            }
            return true;
        }

        // Decodes the next block if needed. Like binpack::ChunkReader, the
        // loader calls next on a fresh chunk without asking hasNext; every
        // chunk holds at least one record.
        [[nodiscard]] binpack::TrainingDataEntry next(const Chunk& chunk)
        {
            if (m_offset >= m_entries.size() && !hasNext(chunk))
                throw DecodeError("FSQD chunk: read past the end");
            return m_entries[m_offset++];
        }

    private:
        std::vector<binpack::TrainingDataEntry> m_entries;
        std::size_t m_offset = 0;
        std::uint32_t m_block = 0;

        bool nextBlock(const Chunk& chunk)
        {
            const unsigned char* d = chunk.data.data();
            if (chunk.data.size() < 8)
                return false;
            const std::uint32_t blocks = read_u32(d);
            if (m_block >= blocks)
                return false;
            const std::uint32_t last = read_u32(d + 4);
            const unsigned char* offsets = d + 8;
            const unsigned char* payload = offsets + 4 * (std::size_t(blocks) + 1);
            const std::uint32_t start = read_u32(offsets + 4 * m_block);
            const std::uint32_t end = read_u32(offsets + 4 * (m_block + 1));
            const std::size_t n = m_block + 1 == blocks ? last : BLOCK_RECORDS;
            decode_block(*chunk.model, payload + start, end - start, n, m_entries);
            m_offset = 0;
            ++m_block;
            return true;
        }
    };

    // An FSQD file split into chunks of up to kChunkBlocks blocks of one
    // segment. The interface mirrors binpack::CompressedTrainingDataFile.
    struct File
    {
        static constexpr std::uint64_t kChunkBlocks = 384;  // about 1 MB, 98k records

        File(std::string path, ModelCache& cache) :
            m_path(std::move(path)),
            m_file(m_path, std::ios_base::in | std::ios_base::binary)
        {
            if (!m_file)
                throw std::runtime_error("Cannot open " + m_path);
            m_file.seekg(0, std::ios_base::end);
            m_sizeBytes = static_cast<std::uint64_t>(m_file.tellg());
            openSegments(cache);
        }

        [[nodiscard]] bool hasNextChunk() const { return m_next < m_chunks.size(); }

        void seek_to_start() { m_next = 0; }

        [[nodiscard]] bool skipChunks(std::size_t n, std::size_t* skipped_out)
        {
            const std::size_t skipped = std::min<std::size_t>(n, m_chunks.size() - m_next);
            m_next += skipped;
            if (skipped_out)
                *skipped_out = skipped;
            return skipped == n;
        }

        [[nodiscard]] bool skipChunks(std::size_t n) { return skipChunks(n, nullptr); }

        // Read the next chunk's index entries and payload.
        void readNextChunkInto(Chunk& out)
        {
            const ChunkRef& c = m_chunks.at(m_next++);
            const Segment& seg = m_segments[c.segment];
            const bool last = c.first + c.blocks == seg.blocks;
            const std::uint64_t nOffsets = last ? c.blocks : c.blocks + 1;

            std::vector<unsigned char> index(8 * nOffsets);
            readAt(seg.start + seg.indexStart + 8 * c.first, index.data(), index.size());
            std::vector<std::uint64_t> offs(c.blocks + 1);
            for (std::uint64_t i = 0; i < nOffsets; ++i)
                offs[i] = read_u64(&index[8 * i]);
            if (last)
                offs[c.blocks] = seg.indexStart;
            for (std::uint64_t i = 0; i < c.blocks; ++i)
            {
                const std::uint64_t b = c.first + i;
                const std::uint64_t n = blockRecords(seg, b);
                if (offs[i] < seg.firstBlock || offs[i + 1] > seg.indexStart || offs[i] >= offs[i + 1]
                    || offs[i + 1] - offs[i] < 4 || offs[i + 1] - offs[i] > max_block_len(n))
                    fail("block " + std::to_string(b) + " of segment at " + std::to_string(seg.start)
                         + " has an invalid span");
            }

            const std::uint64_t payloadLen = offs[c.blocks] - offs[0];
            const std::size_t head = 8 + 4 * (c.blocks + 1);
            out.model = seg.model.get();
            out.data.resize(head + payloadLen);
            unsigned char* d = out.data.data();
            auto put = [](unsigned char* p, std::uint32_t v) {
                for (int k = 0; k < 4; ++k)
                    p[k] = static_cast<unsigned char>(v >> (8 * k));
            };
            put(d, static_cast<std::uint32_t>(c.blocks));
            put(d + 4, static_cast<std::uint32_t>(blockRecords(seg, c.first + c.blocks - 1)));
            for (std::uint64_t i = 0; i <= c.blocks; ++i)
                put(d + 8 + 4 * i, static_cast<std::uint32_t>(offs[i] - offs[0]));
            readAt(seg.start + offs[0], d + head, payloadLen);
        }

        [[nodiscard]] std::size_t sizeBytes() const { return m_sizeBytes; }
        [[nodiscard]] std::string path() const { return m_path; }
        [[nodiscard]] std::uint64_t records() const { return m_records; }

    private:
        struct Segment
        {
            std::uint64_t start, records, blocks, firstBlock, indexStart;
            std::shared_ptr<const Model> model;
        };

        struct ChunkRef
        {
            std::uint32_t segment;
            std::uint64_t first, blocks;
        };

        std::string m_path;
        std::ifstream m_file;
        std::uint64_t m_sizeBytes = 0;
        std::uint64_t m_records = 0;
        std::vector<Segment> m_segments;
        std::vector<ChunkRef> m_chunks;
        std::size_t m_next = 0;

        [[noreturn]] void fail(const std::string& msg) const
        {
            throw std::runtime_error("Invalid FSQD file " + m_path + ": " + msg);
        }

        static std::uint64_t blockRecords(const Segment& seg, std::uint64_t b)
        {
            return b + 1 < seg.blocks ? BLOCK_RECORDS : seg.records - b * BLOCK_RECORDS;
        }

        void readAt(std::uint64_t offset, unsigned char* dst, std::uint64_t n)
        {
            if (offset > m_sizeBytes || n > m_sizeBytes - offset)
                fail("truncated");
            m_file.clear();
            m_file.seekg(static_cast<std::streamoff>(offset));
            m_file.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(n));
            if (!m_file)
                fail("read error");
        }

        // Walk the segments backwards from the end of the file (container.md).
        void openSegments(ModelCache& cache)
        {
            if (m_sizeBytes < HEADER_LEN + FOOTER_LEN)
                fail("too short");
            std::uint64_t end = m_sizeBytes;
            while (end > 0)
            {
                if (end < HEADER_LEN + FOOTER_LEN)
                    fail("segment too short for a header and footer");
                unsigned char footer[FOOTER_LEN];
                readAt(end - FOOTER_LEN, footer, FOOTER_LEN);
                if (std::memcmp(footer + 20, "FEND", 4) != 0)
                    fail("bad footer magic");
                if (read_u32(footer + 16) != 0)
                    fail("reserved footer bits set");
                Segment seg{};
                seg.records = read_u64(footer);
                seg.indexStart = read_u64(footer + 8);
                seg.blocks = seg.records / BLOCK_RECORDS + (seg.records % BLOCK_RECORDS != 0);
                if (seg.blocks > (UINT64_MAX - FOOTER_LEN) / 8
                    || seg.indexStart > UINT64_MAX - FOOTER_LEN - 8 * seg.blocks)
                    fail("index size overflows");
                const std::uint64_t segLen = seg.indexStart + 8 * seg.blocks + FOOTER_LEN;
                if (segLen > end)
                    fail("segment extends before the start of the file");
                seg.start = end - segLen;

                unsigned char header[HEADER_LEN];
                readAt(seg.start, header, HEADER_LEN);
                if (std::memcmp(header, "FSQD", 4) != 0)
                    fail("bad magic");
                if ((header[4] | header[5] << 8) != VERSION)
                    fail("unsupported version");
                if ((header[6] | header[7] << 8) != 0)
                    fail("unknown flags");
                const std::uint64_t modelLen = read_u64(header + 8);
                if (modelLen > max_model_len() || HEADER_LEN + modelLen > seg.indexStart)
                    fail("model length out of bounds");
                seg.firstBlock = HEADER_LEN + modelLen;
                if (seg.blocks == 0 && seg.indexStart != seg.firstBlock)
                    fail("empty segment has data between model and index");
                if (seg.blocks > 0)
                {
                    unsigned char first[8];
                    readAt(seg.start + seg.indexStart, first, 8);
                    if (read_u64(first) != seg.firstBlock)
                        fail("first block does not follow the model");
                }

                std::vector<unsigned char> bytes(modelLen);
                readAt(seg.start + HEADER_LEN, bytes.data(), modelLen);
                auto it = cache.find(bytes);
                if (it == cache.end())
                {
                    auto model = std::make_shared<const Model>(bytes);
                    it = cache.emplace(std::move(bytes), std::move(model)).first;
                }
                seg.model = it->second;
                end = seg.start;
                m_segments.push_back(std::move(seg));
            }
            std::reverse(m_segments.begin(), m_segments.end());
            for (std::uint32_t s = 0; s < m_segments.size(); ++s)
            {
                m_records += m_segments[s].records;
                for (std::uint64_t b = 0; b < m_segments[s].blocks; b += kChunkBlocks)
                    m_chunks.push_back({s, b, std::min(kChunkBlocks, m_segments[s].blocks - b)});
            }
        }
    };

    inline bool has_extension(const std::string& path)
    {
        return path.size() >= 5 && path.compare(path.size() - 5, 5, ".fsqd") == 0;
    }
}
