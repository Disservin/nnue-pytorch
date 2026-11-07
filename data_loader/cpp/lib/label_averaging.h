#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "training_data_entry.h"

namespace binpack {

// Replaces a position's training score with the average of its own score and
// the scores of the next `plies` positions of the same game. Binpacks are read,
// never written: the entries are delayed in a small ring buffer while decoding,
// so the filters still see the original score and only kept entries pay for the
// averaging.
//
// Scores are averaged as win probabilities, otherwise a few huge scores would
// dominate. Positions without a valid score (VALUE_NONE, mate and tablebase
// scores, placeholder zeros) don't contribute, and a position that is itself a
// mate or tablebase score keeps it, since the loss treats those separately.
class ChainLabelAverager {
   public:
    static constexpr int    MaxPlies        = 32;
    static constexpr int    MateThreshold   = 32000 - 245;
    static constexpr double Scale           = 380.0;  // Matches the loss's out-scaling.
    static constexpr double MinProbability  = 1e-6;
    static constexpr int    PlaceholderBase = 100;

    explicit ChainLabelAverager(int plies) :
        m_capacity(static_cast<std::size_t>(std::clamp(plies, 0, MaxPlies)) + 1),
        m_entries(m_capacity) { }

    void push(const TrainingDataEntry& e) {
        m_entries[(m_head + m_size) % m_capacity] = e;
        ++m_size;
    }

    // The front entry's window is complete once the buffer is full. When the
    // chunk ended, the next chunk isn't a continuation, so drain regardless.
    [[nodiscard]] bool ready(bool draining) const {
        return draining ? m_size > 0 : m_size == m_capacity;
    }

    [[nodiscard]] const TrainingDataEntry& front() const { return m_entries[m_head]; }

    void pop() {
        m_head = (m_head + 1) % m_capacity;
        --m_size;
    }

    [[nodiscard]] std::int16_t averaged_front_score() const {
        const TrainingDataEntry& first = front();
        if (!has_usable_score(first.score))
            return first.score;

        double sum   = probability(first.score);
        int    count = 1;
        int    last  = first.score;

        for (std::size_t i = 1; i < m_size; ++i)
        {
            const TrainingDataEntry& previous = at(i - 1);
            const TrainingDataEntry& current  = at(i);

            if (current.ply != previous.ply + 1 || current.result != -previous.result)
                break;
            if (!has_usable_score(current.score))
                continue;
            if (current.score == 0 && current.result != 0 && std::abs(last) > PlaceholderBase)
                continue;

            last = current.score;
            // Scores are relative to the side to move, which alternates.
            sum += probability(i % 2 == 0 ? current.score : -current.score);
            ++count;
        }

        const double p = std::clamp(sum / count, MinProbability, 1.0 - MinProbability);
        return static_cast<std::int16_t>(std::lround(Scale * std::log(p / (1.0 - p))));
    }

   private:
    [[nodiscard]] static bool has_usable_score(int score) {
        return std::abs(score) < MateThreshold;
    }

    [[nodiscard]] static double probability(int score) {
        return 1.0 / (1.0 + std::exp(-score / Scale));
    }

    [[nodiscard]] const TrainingDataEntry& at(std::size_t i) const {
        return m_entries[(m_head + i) % m_capacity];
    }

    std::size_t                    m_capacity;
    std::vector<TrainingDataEntry> m_entries;
    std::size_t                    m_head = 0;
    std::size_t                    m_size = 0;
};

}  // namespace binpack
