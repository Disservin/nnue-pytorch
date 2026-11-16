"""Summarize NCCL kernels and concurrent GPU work in a PyTorch trace.

Kernel overlap is descriptive, not proof of critical-path exposure: NCCL
kernels can wait for other ranks or contend with other GPU work.
"""
import argparse
import json


def union(intervals):
    result = []
    for start, end in sorted(intervals):
        if not result or start > result[-1][1]:
            result.append([start, end])
        else:
            result[-1][1] = max(result[-1][1], end)
    return result


def intersection_length(left, right):
    left, right = union(left), union(right)
    i = j = 0
    total = 0.0
    while i < len(left) and j < len(right):
        total += max(0.0, min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]))
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return total


def analyze(path):
    with open(path, encoding="utf-8") as stream:
        trace = json.load(stream)
    events = [e for e in trace["traceEvents"] if e.get("ph") == "X" and "dur" in e]
    kernels = [e for e in events if e.get("cat") == "kernel"]
    nccl = [e for e in kernels if "nccl" in e.get("name", "").lower()]
    # Synchronization inserted for profiler startup is not training work.
    barriers = [e for e in nccl if e.get("args", {}).get("Collective name") == "barrier"]
    comm = [e for e in nccl if e not in barriers]
    compute = [e for e in kernels if e not in nccl]
    interval = lambda e: (e["ts"], e["ts"] + e["dur"])
    comm_intervals = union([interval(e) for e in comm])
    comm_span = sum(end - start for start, end in comm_intervals)
    overlap = intersection_length(comm_intervals, [interval(e) for e in compute])
    origin = min((e["ts"] for e in kernels), default=0)
    backward = next((e for e in events if e.get("name") == "step:ddp_backward"
                     and e.get("cat") != "gpu_user_annotation"), None)
    backward_ids = set()
    if backward:
        start, end = interval(backward)
        backward_ids = {e.get("args", {}).get("External id") for e in events
                        if e.get("cat") == "cpu_op" and start <= e["ts"] < end}
    backward_compute = [e for e in compute if e.get("args", {}).get("External id") in backward_ids]
    last_compute_event = max(backward_compute, key=lambda e: e["ts"] + e["dur"], default=None)
    last_compute = (last_compute_event["ts"] + last_compute_event["dur"]
                    if last_compute_event is not None else None)
    last_comm = max((e["ts"] + e["dur"] for e in comm), default=None)
    collectives = []
    for event in sorted(comm, key=lambda e: e["ts"]):
        args = event.get("args", {})
        collectives.append({"name": args.get("Collective name", event["name"]),
                            "input_elements": args.get("In msg nelems"),
                            "dtype": args.get("dtype"),
                            "start_ms": (event["ts"] - origin) / 1000,
                            "duration_ms": event["dur"] / 1000,
                            "stream": args.get("stream")})
    return {"file": path, "rank": trace.get("distributedInfo", {}).get("rank"),
            "profiler_barriers_excluded": len(barriers),
            "comm_kernel_union_ms": comm_span / 1000,
            "comm_compute_overlap_ms": overlap / 1000,
            "comm_without_other_kernels_ms": (comm_span - overlap) / 1000,
            "last_comm_minus_last_attributed_backward_compute_ms":
                (last_comm - last_compute) / 1000 if last_comm is not None and last_compute is not None else None,
            "last_attributed_backward_kernel": last_compute_event.get("name") if last_compute_event else None,
            "backward_compute_kernels_attributed": len(backward_compute),
            "collectives": collectives}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", nargs="+")
    args = parser.parse_args()
    for trace_path in args.traces:
        print(json.dumps(analyze(trace_path)))
