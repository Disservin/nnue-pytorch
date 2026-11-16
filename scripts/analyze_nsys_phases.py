"""Summarize per-rank NVTX phase ranges in an Nsight Systems SQLite export.

GPU time is attributed to CPU launch ranges by correlation ID. Phase ranges
are nested; sums are inclusive and cannot be added. NCCL kernel durations can
include waiting for peers. Captures and instrumentation can perturb timing.
"""

import argparse
import bisect
import collections
import json
import sqlite3
import statistics


def summarize(path):
    con = sqlite3.connect(path)
    strings = dict(con.execute("SELECT id, value FROM StringIds"))
    phases = collections.defaultdict(list)
    by_thread = collections.defaultdict(list)
    for start, end, tid, text, text_id in con.execute(
        "SELECT start,end,globalTid,text,textId FROM NVTX_EVENTS WHERE end IS NOT NULL"
    ):
        name = text or strings.get(text_id, "")
        if name.startswith(("phase:", "step:")):
            phases[name].append(end - start)
            by_thread[tid].append((start, end, name))
    indexes = {}
    for tid, ranges in by_thread.items():
        ranges.sort()
        starts, max_ends = [], []
        maximum = 0
        for start, end, _ in ranges:
            starts.append(start)
            maximum = max(maximum, end)
            max_ends.append(maximum)
        indexes[tid] = ranges, starts, max_ends
    launches = {corr: (start, tid) for start, tid, corr in con.execute(
        "SELECT start,globalTid,correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME"
    )}
    device = collections.defaultdict(lambda: [0, 0, 0])
    for start, end, corr, name_id in con.execute(
        "SELECT start,end,correlationId,shortName FROM CUPTI_ACTIVITY_KIND_KERNEL"
    ):
        launch = launches.get(corr)
        if launch is None or launch[1] not in indexes:
            continue
        ranges, starts, max_ends = indexes[launch[1]]
        i = bisect.bisect_right(starts, launch[0]) - 1
        names = set()
        while i >= 0 and max_ends[i] >= launch[0]:
            if ranges[i][1] >= launch[0]:
                names.add(ranges[i][2])
            i -= 1
        is_nccl = "nccl" in strings.get(name_id, "").lower()
        for name in names:
            device[name][0] += end - start
            device[name][1] += (end - start) if is_nccl else 0
            device[name][2] += 1
    result = {}
    for name, samples in sorted(phases.items()):
        samples.sort()
        gpu_ns, nccl_ns, gpu_count = device[name]
        result[name] = {
            "calls": len(samples),
            "host_median_ms": statistics.median(samples) / 1e6,
            "host_p95_ms": samples[int(.95 * (len(samples) - 1))] / 1e6,
            "host_total_ms": sum(samples) / 1e6,
            "attributed_kernel_total_ms": gpu_ns / 1e6,
            "attributed_nccl_total_ms": nccl_ns / 1e6,
            "attributed_kernel_count": gpu_count,
        }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", nargs="+")
    args = parser.parse_args()
    for trace in args.trace:
        print(json.dumps({"trace": trace, "phases": summarize(trace)}))
