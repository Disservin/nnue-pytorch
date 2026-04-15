import threading
import queue

import torch
from torch.utils.data import Dataset

from . import stream
from .config import DataloaderSkipConfig, DataloaderDDPConfig


def _recursive_pin(obj):
    if isinstance(obj, torch.Tensor):
        if obj.is_cuda:
            return obj
        return obj.pin_memory()
    elif isinstance(obj, dict):
        return {k: _recursive_pin(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(_recursive_pin(v) for v in obj)
    return obj


class FenBatchProvider:
    def __init__(
        self,
        filename,
        cyclic,
        num_workers,
        batch_size=None,
        config: DataloaderSkipConfig = DataloaderSkipConfig(),
        ddp_config: DataloaderDDPConfig = None,
    ):
        self.filename = filename
        self.cyclic = cyclic
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.config = config

        if batch_size:
            self.stream = stream.create_fen_batch_stream(
                self.num_workers,
                [self.filename],
                batch_size,
                cyclic,
                config,
                ddp_config,
            )
        else:
            # doesnt work yet
            assert False
            # self.stream = make_fen_batch_stream(
            #     self.num_workers,
            #     [self.filename],
            #     cyclic,
            #     config=config
            # )

    def __iter__(self):
        return self

    def __next__(self):
        v = stream.fetch_next_fen_batch(self.stream)

        if v:
            fens = v.contents.get_fens()
            stream.destroy_fen_batch(v)
            return fens
        else:
            raise StopIteration

    def __del__(self):
        stream.destroy_fen_batch_stream(self.stream)


class TrainingDataProvider:
    def __init__(
        self,
        feature_set: str,
        create_stream,
        destroy_stream,
        fetch_next,
        destroy_part,
        filenames: list[str],
        cyclic,
        num_workers,
        batch_size=None,
        config: DataloaderSkipConfig = DataloaderSkipConfig(),
        ddp_config: DataloaderDDPConfig = None,
        device="cpu",
        buffers=None,
    ):
        self.feature_set = feature_set.encode("utf-8")
        self.create_stream = create_stream
        self.destroy_stream = destroy_stream
        self.fetch_next = fetch_next
        self.destroy_part = destroy_part
        self.filenames = filenames
        self.cyclic = cyclic
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.config = config
        self.device = device
        self.buffers = buffers

        if batch_size:
            self.stream = self.create_stream(
                self.feature_set,
                self.num_workers,
                self.filenames,
                batch_size,
                cyclic,
                config,
                ddp_config,
            )
        else:
            self.stream = self.create_stream(
                self.feature_set,
                self.num_workers,
                self.filenames,
                cyclic,
                config,
                ddp_config,
            )

    def __iter__(self):
        return self

    def __next__(self):
        v = self.fetch_next(self.stream)

        if v:
            return _LazyBatch(v, self.destroy_part)
        else:
            raise StopIteration

    def __del__(self):
        self.destroy_stream(self.stream)


class _LazyBatch:
    def __init__(self, batch_ptr, destroy_fn):
        self.batch_ptr = batch_ptr
        self.destroy_fn = destroy_fn

    def to_tensors(self, device, buffers=None):
        tensors = self.batch_ptr.contents.get_tensors(device, buffers)
        self.destroy_fn(self.batch_ptr)
        self.batch_ptr = None
        return tensors


class SparseBatchProvider(TrainingDataProvider):
    def __init__(
        self,
        feature_set: str,
        filenames: list[str],
        batch_size,
        cyclic=True,
        num_workers=1,
        config: DataloaderSkipConfig = DataloaderSkipConfig(),
        ddp_config: DataloaderDDPConfig = None,
        device="cpu",
        buffers=None,
    ):
        super().__init__(
            feature_set,
            stream.create_sparse_batch_stream,
            stream.destroy_sparse_batch_stream,
            stream.fetch_next_sparse_batch,
            stream.destroy_sparse_batch,
            filenames,
            cyclic,
            num_workers,
            batch_size,
            config,
            ddp_config,
            device,
            buffers,
        )


class SparseBatchDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        feature_set: str,
        filenames: list[str],
        batch_size,
        cyclic=True,
        num_workers=1,
        config: DataloaderSkipConfig = DataloaderSkipConfig(),
        ddp_config: DataloaderDDPConfig = None,
        device="cpu",
        buffers=None,
    ):
        super().__init__()
        self.feature_set = feature_set
        self.filenames = filenames
        self.batch_size = batch_size
        self.cyclic = cyclic
        self.num_workers = num_workers
        self.config = config
        self.ddp_config = ddp_config
        self.device = device
        self.buffers = buffers

    def __iter__(self):
        return SparseBatchProvider(
            self.feature_set,
            self.filenames,
            self.batch_size,
            cyclic=self.cyclic,
            num_workers=self.num_workers,
            config=self.config,
            ddp_config=self.ddp_config,
            device=self.device,
            buffers=self.buffers,
        )


class FixedNumBatchesDataset(Dataset):
    def __init__(
        self,
        dataset,
        num_batches,
        pin_memory=False,
        queue_size_limit=None,
        device="cpu",
        batch_size=None,
        max_active_features=None,
    ):
        super().__init__()
        self.dataset = dataset
        self.iter = None
        self.num_batches = num_batches
        self.pin_memory = pin_memory
        self.device = device
        self.batch_size = batch_size
        self.max_active_features = max_active_features
        if queue_size_limit is None:
            queue_size_limit = 10 if pin_memory else 100

        if self.device != "cpu" and self.batch_size is not None and self.max_active_features is not None:
            queue_size_limit = 1

        self._prefetch_queue = queue.Queue(maxsize=queue_size_limit)
        self._prefetch_thread = None
        self._stop_prefetching = threading.Event()
        self._prefetch_started = False
        self._lock = threading.Lock()
        self._buffers = None
        self._first_batch = None
        if self.device != "cpu" and self.batch_size is not None and self.max_active_features is not None:
            self._create_buffers()

    def _create_buffers(self):
        def make_buffers():
            return {
                "us": torch.empty(
                    self.batch_size, 1, dtype=torch.float32, device=self.device
                ),
                "them": torch.empty(
                    self.batch_size, 1, dtype=torch.float32, device=self.device
                ),
                "white_indices": torch.empty(
                    self.batch_size,
                    self.max_active_features,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "white_values": torch.empty(
                    self.batch_size,
                    self.max_active_features,
                    dtype=torch.float32,
                    device=self.device,
                ),
                "black_indices": torch.empty(
                    self.batch_size,
                    self.max_active_features,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "black_values": torch.empty(
                    self.batch_size,
                    self.max_active_features,
                    dtype=torch.float32,
                    device=self.device,
                ),
                "outcome": torch.empty(
                    self.batch_size, 1, dtype=torch.float32, device=self.device
                ),
                "score": torch.empty(
                    self.batch_size, 1, dtype=torch.float32, device=self.device
                ),
                "psqt_indices": torch.empty(
                    self.batch_size, dtype=torch.long, device=self.device
                ),
                "layer_stack_indices": torch.empty(
                    self.batch_size, dtype=torch.long, device=self.device
                ),
            }
        self._buffers = [make_buffers(), make_buffers()]
        self._buffer_idx = 0

    def _safe_put(self, item):
        while not self._stop_prefetching.is_set():
            try:
                self._prefetch_queue.put(item, timeout=1.0)
                break
            except queue.Full:
                continue

    def _prefetch_worker(self):
        try:
            while not self._stop_prefetching.is_set():
                try:
                    item = next(self.iter)
                    self._safe_put((item, self._buffer_idx))
                    self._buffer_idx = 1 - self._buffer_idx
                except StopIteration:
                    self._safe_put((None, self._buffer_idx))
                    break
        except Exception as e:
            self._safe_put((e, self._buffer_idx))

    def _start_prefetching(self):
        with self._lock:
            if not self._prefetch_started:
                self.dataset.device = self.device
                self.iter = iter(self.dataset)
                self._prefetch_thread = threading.Thread(
                    target=self._prefetch_worker, daemon=True
                )
                self._prefetch_thread.start()
                self._prefetch_started = True
                self._buffer_idx = 0

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        self._start_prefetching()

        try:
            item, buffer_idx = self._prefetch_queue.get(timeout=300.0)

            if item is None:
                raise StopIteration("End of dataset reached")
            elif isinstance(item, Exception):
                raise item

            return item.to_tensors(self.device, self._buffers[buffer_idx] if self._buffers is not None else None)

        except queue.Empty:
            raise RuntimeError("Prefetch timeout - no data available")

    def __del__(self):
        if hasattr(self, "_stop_prefetching"):
            self._stop_prefetching.set()
        if hasattr(self, "_prefetch_thread") and self._prefetch_thread:
            self._prefetch_thread.join(timeout=1.0)
