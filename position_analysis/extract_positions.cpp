// extract_positions: sample positions from one or more .binpack files into a TSV file.
//
// Positions are drawn from randomly chosen chunks spread over the whole of every input file
// (binpack chunks are independently decodable), so the sample is representative of the file and not
// just of its beginning. Within a chunk, a uniform random subset of positions is taken.
//
// Output columns (tab separated, with header):
//   fen        position (FEN, with rule50 and move number)
//   move       move stored in the binpack (normally the search best move), UCI
//   score      stored search score, side-to-move perspective, internal units
//   ply        game ply
//   result     game result for the side to move (1 win, 0 draw, -1 loss; other values = no result)
//   next_score search score of the next position in the same game, negated to the side to move of this
//              position (empty if the next entry is not a continuation). |score - next_score| measures
//              how much the search value moved after one ply.
//   prev_score search score of the previous position in the same game, negated to the side to move of
//              this position (empty if this entry is not a continuation of the previous one)
//   capture    1 if the stored move is a capture
//   in_check   1 if the side to move is in check
//   source     index of the input file (as passed on the command line)
//
// Build: see build.sh (only needs a C++20 compiler and the repo headers in data_loader/cpp/lib).

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "binpack.h"

namespace {

constexpr int VALUE_NONE = 32002;

struct ChunkRef {
    int           file;
    std::uint64_t offset;  // offset of the chunk payload (after the 8 byte header)
    std::uint32_t size;
};

struct Options {
    std::vector<std::string> files;
    std::string              out          = "positions.tsv";
    std::uint64_t            count        = 1000000;
    std::uint64_t            per_chunk    = 2000;
    std::uint64_t            seed         = 1;
    bool                     filtered     = false;
    int                      max_ply      = -1;
    int                      min_ply      = 0;
    int                      threads      = 0;
    bool                     by_size      = false;
};

void usage() {
    std::cerr
      << "usage: extract_positions [options] file1.binpack [file2.binpack ...]\n"
         "  -o FILE          output TSV (default positions.tsv)\n"
         "  -n N             number of positions to extract (default 1000000)\n"
         "  --per-chunk N    positions taken from each sampled chunk (default 2000); smaller = more\n"
         "                   chunks visited = more diverse sample, but slower\n"
         "  --seed S         random seed (default 1)\n"
         "  --filtered       drop positions the trainer skips with --filtered (capture as stored move\n"
         "                   or side to move in check). By default they are kept and flagged.\n"
         "  --min-ply P      drop positions with ply < P (default 0)\n"
         "  --max-ply P      drop positions with ply > P\n"
         "  --by-size        with several files, sample in proportion to file size (default: the same\n"
         "                   number of chunks from every file, round robin)\n"
         "  -j T             threads (default: hardware concurrency)\n";
}

bool parse(int argc, char** argv, Options& o) {
    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        auto        next = [&]() -> std::string {
            if (i + 1 >= argc)
            {
                std::cerr << "missing value for " << a << "\n";
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "-h" || a == "--help")
            return false;
        else if (a == "-o")
            o.out = next();
        else if (a == "-n")
            o.count = std::stoull(next());
        else if (a == "--per-chunk")
            o.per_chunk = std::max<std::uint64_t>(1, std::stoull(next()));
        else if (a == "--seed")
            o.seed = std::stoull(next());
        else if (a == "--filtered")
            o.filtered = true;
        else if (a == "--min-ply")
            o.min_ply = std::stoi(next());
        else if (a == "--max-ply")
            o.max_ply = std::stoi(next());
        else if (a == "--by-size")
            o.by_size = true;
        else if (a == "-j")
            o.threads = std::stoi(next());
        else if (!a.empty() && a[0] == '-')
        {
            std::cerr << "unknown option " << a << "\n";
            return false;
        }
        else
            o.files.push_back(a);
    }
    return !o.files.empty();
}

std::vector<ChunkRef> index_chunks(const std::string& path, int file_id) {
    std::vector<ChunkRef> chunks;
    std::ifstream         in(path, std::ios::binary);
    if (!in)
    {
        std::cerr << "cannot open " << path << "\n";
        std::exit(1);
    }
    in.seekg(0, std::ios::end);
    const std::uint64_t file_size = in.tellg();
    std::uint64_t       pos       = 0;
    while (pos + 8 <= file_size)
    {
        unsigned char h[8];
        in.seekg(pos);
        in.read(reinterpret_cast<char*>(h), 8);
        if (!in || h[0] != 'B' || h[1] != 'I' || h[2] != 'N' || h[3] != 'P')
        {
            std::cerr << path << ": bad chunk header at offset " << pos << ", stopping index\n";
            break;
        }
        const std::uint32_t size = h[4] | (h[5] << 8) | (h[6] << 16) | (std::uint32_t(h[7]) << 24);
        if (pos + 8 + size > file_size)
            break;
        chunks.push_back({file_id, pos + 8, size});
        pos += 8 + std::uint64_t(size);
    }
    return chunks;
}

// Decode one chunk and return the sampled rows (already formatted).
std::vector<std::string> process_chunk(const Options& o, const ChunkRef& c, std::uint64_t take,
                                       std::uint64_t seed) {
    std::vector<unsigned char> buf(c.size);
    {
        std::ifstream in(o.files[c.file], std::ios::binary);
        in.seekg(c.offset);
        in.read(reinterpret_cast<char*>(buf.data()), c.size);
        if (!in)
            return {};
    }

    std::vector<binpack::TrainingDataEntry> entries;
    entries.reserve(c.size / 2);
    try
    {
        binpack::ChunkReader reader;
        while (reader.hasNext(buf))
            entries.push_back(reader.next(buf));
    } catch (const std::exception& e)
    {
        std::cerr << "chunk decode error (" << e.what() << "), keeping " << entries.size()
                  << " decoded entries\n";
    }

    // Eligible entries
    std::vector<std::uint32_t> idx;
    idx.reserve(entries.size());
    for (std::uint32_t i = 0; i < entries.size(); ++i)
    {
        const auto& e = entries[i];
        if (e.score == VALUE_NONE || std::abs(e.score) > 32000)
            continue;
        if (e.ply < o.min_ply || (o.max_ply >= 0 && e.ply > o.max_ply))
            continue;
        if (o.filtered && (e.isCapturingMove() || e.isInCheck()))
            continue;
        idx.push_back(i);
    }

    std::mt19937_64 rng(seed);
    if (idx.size() > take)
    {
        // partial Fisher-Yates
        for (std::uint64_t k = 0; k < take; ++k)
        {
            std::uniform_int_distribution<std::uint64_t> d(k, idx.size() - 1);
            std::swap(idx[k], idx[d(rng)]);
        }
        idx.resize(take);
        std::sort(idx.begin(), idx.end());
    }

    std::vector<std::string> rows;
    rows.reserve(idx.size());
    for (auto i : idx)
    {
        const auto& e = entries[i];
        std::string prev_score, next_score;
        if (i > 0)
        {
            const auto& p = entries[i - 1];
            if (p.score != VALUE_NONE && binpack::isContinuation(p, e))
                prev_score = std::to_string(-int(p.score));
        }
        if (i + 1 < entries.size())
        {
            const auto& n = entries[i + 1];
            if (n.score != VALUE_NONE && binpack::isContinuation(e, n))
                next_score = std::to_string(-int(n.score));
        }
        std::string line;
        line.reserve(128);
        line += e.pos.fen();
        line += '\t';
        line += chess::uci::moveToUci(e.pos, e.move);
        line += '\t';
        line += std::to_string(int(e.score));
        line += '\t';
        line += std::to_string(int(e.ply));
        line += '\t';
        line += std::to_string(int(e.result));
        line += '\t';
        line += next_score;
        line += '\t';
        line += prev_score;
        line += '\t';
        line += e.isCapturingMove() ? '1' : '0';
        line += '\t';
        line += e.isInCheck() ? '1' : '0';
        line += '\t';
        line += std::to_string(c.file);
        line += '\n';
        rows.push_back(std::move(line));
    }
    return rows;
}

}  // namespace

int main(int argc, char** argv) {
    Options o;
    if (!parse(argc, argv, o))
    {
        usage();
        return 2;
    }

    // Random order of the chunks; we walk it until enough positions are collected.
    std::mt19937_64                    rng(o.seed);
    std::vector<std::vector<ChunkRef>> per_file;
    for (int f = 0; f < int(o.files.size()); ++f)
    {
        auto c = index_chunks(o.files[f], f);
        std::cerr << o.files[f] << ": " << c.size() << " chunks\n";
        std::shuffle(c.begin(), c.end(), rng);
        per_file.push_back(std::move(c));
    }
    std::vector<ChunkRef> chunks;
    if (o.by_size)
    {
        for (auto& c : per_file)
            chunks.insert(chunks.end(), c.begin(), c.end());
        std::shuffle(chunks.begin(), chunks.end(), rng);
    }
    else
    {
        // round robin over the files
        for (std::size_t k = 0;; ++k)
        {
            bool any = false;
            for (auto& c : per_file)
                if (k < c.size())
                {
                    chunks.push_back(c[k]);
                    any = true;
                }
            if (!any)
                break;
        }
    }
    if (chunks.empty())
    {
        std::cerr << "no chunks found\n";
        return 1;
    }

    const int threads =
      o.threads > 0 ? o.threads : std::max(1u, std::thread::hardware_concurrency());

    std::ofstream out(o.out);
    if (!out)
    {
        std::cerr << "cannot write " << o.out << "\n";
        return 1;
    }
    out << "fen\tmove\tscore\tply\tresult\tnext_score\tprev_score\tcapture\tin_check\tsource\n";

    std::atomic<std::size_t>   next_chunk{0};
    std::atomic<std::uint64_t> written{0};
    std::mutex                 out_mutex;

    auto worker = [&]() {
        while (true)
        {
            if (written.load() >= o.count)
                return;
            const std::size_t ci = next_chunk.fetch_add(1);
            if (ci >= chunks.size())
                return;
            auto rows = process_chunk(o, chunks[ci], o.per_chunk, o.seed * 0x9E3779B97F4A7C15ULL + ci);
            std::lock_guard lock(out_mutex);
            for (auto& r : rows)
            {
                if (written.load() >= o.count)
                    break;
                out << r;
                written.fetch_add(1);
            }
        }
    };

    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t)
        pool.emplace_back(worker);
    for (auto& t : pool)
        t.join();

    std::cerr << "wrote " << written.load() << " positions from "
              << std::min(next_chunk.load(), chunks.size()) << " chunks to " << o.out << "\n";
    if (written.load() < o.count)
        std::cerr << "warning: fewer positions than requested (all chunks used)\n";
    return 0;
}
