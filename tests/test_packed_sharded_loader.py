"""Native packed loader output survives native release and CUDA prefetch."""
from pathlib import Path

import pytest
import torch

from data_loader.dataset import DevicePrefetcher, SparseBatchProvider
from data_loader.distributed_features import gather_packed_features


_BINPACK = Path(__file__).resolve().parents[1] / ".pgo" / "small.binpack"


@pytest.mark.skipif(not torch.cuda.is_available() or not _BINPACK.exists(),
                    reason="CUDA and the local small binpack fixture are required")
def test_packed_loader_cuda_prefetch_and_cached_provider():
    device = torch.device("cuda", min(1, torch.cuda.device_count() - 1))
    ds = SparseBatchProvider("K32H3", [str(_BINPACK)], 128, cyclic=True,
                             num_workers=2, use_pinned_memory=True, packed_sharded=True)
    wrapper = DevicePrefetcher(ds, num_batches=2, queue_size_limit=2, device=device)
    try:
        first = wrapper.next_batch()
        second = wrapper.next_batch()
        assert first[2]["format"] == "native_offsets_v1"
        assert first[2]["packed"].device == device
        assert first[4].device == device and first[5].device == device
        # Churn the caching allocator after the native source batch has been
        # freed; the prefetched data must remain owned and stream-safe.
        for _ in range(10):
            torch.empty_like(first[2]["packed"]).fill_(42)
        for batch in (first, second):
            native = batch[2]
            n, width, nw, nb = (native[k] for k in ("size", "width", "nw", "nb"))
            packed = native["packed"]
            white_offsets = packed[:n+1]
            black_offsets = packed[n+1:2*(n+1)]
            base = 2 * (n + 1)
            actual = gather_packed_features(batch)
            for output, offsets, ids in (
                (actual.white_indices, white_offsets, packed[base:base+nw]),
                (actual.black_indices, black_offsets, packed[base+nw:base+nw+nb]),
            ):
                for row in range(n):
                    start, end = int(offsets[row]), int(offsets[row+1])
                    assert torch.equal(output[row, :end-start], ids[start:end])
                    assert (output[row, end-start:] == -1).all()
            assert actual.white_indices.shape == (n, width)
            assert torch.equal(actual.layer_stack_indices, batch[6])
    finally:
        wrapper._stop_prefetching.set()
        for thread in wrapper._prefetch_threads:
            thread.join(timeout=10)
