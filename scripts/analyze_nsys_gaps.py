"""Summarize gaps between GPU activities in an Nsight Systems SQLite export.

Run on a single-process, single-device-per-rank capture. Timing gaps are
observed gaps between traced kernels/copies, not proof that the GPU is idle:
untraced activity and profiler overhead can change the result.
"""

import argparse
import collections
import json
import sqlite3
import statistics


def summarize(path):
    con = sqlite3.connect(path)
    strings = dict(con.execute("SELECT id, value FROM StringIds"))
    kernels = con.execute(
        "SELECT start,end,correlationId,shortName FROM CUPTI_ACTIVITY_KIND_KERNEL"
    ).fetchall()
    events = [(s, e, corr, strings.get(name, "kernel")) for s, e, corr, name in kernels]
    for table in ("CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
        if con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone():
            events.extend((s, e, corr, table.rsplit("_", 1)[-1].lower())
                          for s, e, corr in con.execute(f"SELECT start,end,correlationId FROM {table}"))
    if not events:
        raise ValueError("no traced GPU activities")
    launches = {corr: (start, tid) for start, tid, corr in con.execute(
        "SELECT start,globalTid,correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME"
    )}
    ranges = [(s, e, tid, text or strings.get(text_id, ""))
              for s, e, tid, text, text_id in con.execute(
                  "SELECT start,end,globalTid,text,textId FROM NVTX_EVENTS WHERE end IS NOT NULL"
              ) if (text or strings.get(text_id, "")).startswith("phase:")]
    events.sort()
    merged_end = events[0][1]
    busy = 0
    start = events[0][0]
    interval_start = start
    gaps = []
    for event_start, event_end, corr, name in events[1:]:
        if event_start > merged_end:
            busy += merged_end - interval_start
            gap = event_start - merged_end
            launch = launches.get(corr)
            labels = [label for s, e, tid, label in ranges
                      if launch and tid == launch[1] and s <= launch[0] <= e]
            gaps.append((gap, name, labels[-1] if labels else "unlabeled"))
            interval_start = event_start
            merged_end = event_end
        else:
            merged_end = max(merged_end, event_end)
    busy += merged_end - interval_start
    total = merged_end - start
    large = [gap for gap in gaps if gap[0] >= 100_000]
    by_phase = collections.Counter()
    for duration, _, label in large:
        by_phase[label] += duration
    return {
        "span_ms": total / 1e6,
        "gpu_activity_union_ms": busy / 1e6,
        "untraced_gap_ms": (total - busy) / 1e6,
        "gaps_ge_0_1ms": len(large),
        "gaps_ge_0_1ms_total_ms": sum(x[0] for x in large) / 1e6,
        "median_gap_ge_0_1ms_ms": statistics.median([x[0] for x in large]) / 1e6 if large else 0,
        "large_gap_ms_by_next_launch_phase": {k: v / 1e6 for k, v in by_phase.most_common()},
        "largest_gaps": [{"ms": n / 1e6, "next_kernel": name, "next_launch_phase": label}
                         for n, name, label in sorted(large, reverse=True)[:10]],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", nargs="+")
    args = parser.parse_args()
    for trace in args.trace:
        print(json.dumps({"trace": trace, **summarize(trace)}))
