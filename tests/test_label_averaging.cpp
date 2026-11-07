// Unit test for ChainLabelAverager. Prints nothing and exits 0 on success.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "label_averaging.h"

using namespace binpack;

namespace {

TrainingDataEntry entry(int score, int ply, int result) {
    TrainingDataEntry e{};
    e.score  = static_cast<std::int16_t>(score);
    e.ply    = static_cast<std::uint16_t>(ply);
    e.result = static_cast<std::int16_t>(result);
    return e;
}

double p(int score) { return 1.0 / (1.0 + std::exp(-score / ChainLabelAverager::Scale)); }

int expected(std::vector<int> whiteRelative) {
    double sum = 0.0;
    for (int s : whiteRelative)
        sum += p(s);
    const double m = sum / static_cast<double>(whiteRelative.size());
    return static_cast<int>(std::lround(ChainLabelAverager::Scale * std::log(m / (1.0 - m))));
}

int run(int plies, const std::vector<TrainingDataEntry>& entries) {
    ChainLabelAverager a(plies);
    for (const auto& e : entries)
        a.push(e);
    return a.averaged_front_score();
}

void check(const char* name, int actual, int want) {
    if (actual != want)
    {
        std::fprintf(stderr, "%s: got %d, want %d\n", name, actual, want);
        std::exit(1);
    }
}

}  // namespace

int main() {
    // Signs alternate with the side to move.
    check("alternating sign", run(2, {entry(100, 30, 1), entry(-200, 31, -1), entry(300, 32, 1)}),
          expected({100, 200, 300}));

    // A settled position is unchanged.
    check("constant", run(2, {entry(150, 30, 1), entry(-150, 31, -1), entry(150, 32, 1)}), 150);

    // Stops at the next game: ply discontinuity, and a result that doesn't flip.
    check("ply break", run(2, {entry(100, 30, 1), entry(-200, 31, -1), entry(500, 5, 1)}),
          expected({100, 200}));
    check("result break", run(2, {entry(100, 30, 1), entry(-200, 31, 1), entry(500, 32, 1)}), 100);

    // Draws flip to themselves and stay in the chain.
    check("draw", run(1, {entry(100, 30, 0), entry(-200, 31, 0)}), expected({100, 200}));

    // Invalid neighbours are skipped without breaking the chain or the sign.
    check("value none", run(2, {entry(100, 30, 1), entry(32002, 31, -1), entry(300, 32, 1)}),
          expected({100, 300}));
    check("mate neighbour", run(1, {entry(100, 30, 1), entry(-31990, 31, -1)}), 100);
    check("placeholder zero", run(2, {entry(500, 30, 1), entry(0, 31, -1), entry(300, 32, 1)}),
          expected({500, 300}));

    // A real zero after a small score, and any zero in a drawn game, still count.
    check("real zero", run(1, {entry(50, 30, 1), entry(0, 31, -1)}), expected({50, 0}));
    check("drawn zero", run(1, {entry(500, 30, 0), entry(0, 31, 0)}), expected({500, 0}));

    // The front's own mate, tablebase or missing score is kept.
    check("own mate", run(1, {entry(31990, 30, 1), entry(-200, 31, -1)}), 31990);
    check("own none", run(1, {entry(32002, 30, 1), entry(-200, 31, -1)}), 32002);

    // Extreme averages stay far below the mate range.
    if (std::abs(run(1, {entry(30000, 30, 1), entry(-30000, 31, -1)}))
        >= ChainLabelAverager::MateThreshold)
        return 1;

    // Delay and drain: the window needs plies + 1 entries, and a chunk end
    // releases what remains using the shorter window.
    ChainLabelAverager a(2);
    a.push(entry(100, 30, 1));
    if (a.ready(false))
        return 1;
    a.push(entry(-200, 31, -1));
    a.push(entry(300, 32, 1));
    if (!a.ready(false))
        return 1;
    check("full window", a.averaged_front_score(), expected({100, 200, 300}));
    a.pop();
    a.push(entry(-400, 33, -1));
    check("ring wraps", a.averaged_front_score(), expected({-200, -300, -400}));
    a.pop();
    if (a.ready(false) || !a.ready(true))
        return 1;
    check("drain", a.averaged_front_score(), expected({300, 400}));
    a.pop();
    check("drain last", a.averaged_front_score(), -400);
    a.pop();
    if (a.ready(true))
        return 1;

    return 0;
}
