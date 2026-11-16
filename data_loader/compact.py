"""Filtered binpack batches in Bulletformat with training metadata in extra bytes.

This is an input-preparation API, not yet a tensor-parallel training loader.
Each record is the 32-byte bulletformat 1.8 ChessBoard layout. Scores retain
the binpack's original side-to-move-relative Stockfish units. Extra bytes 29,
30 and 31 hold the rule-50 counter, original black-to-move flag and search-tree
flag respectively. Full ply is used for filtering and is not retained.
"""

import ctypes
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ._native import c_lib
from .config import (
    CDataloaderIOConfig,
    CDataloaderSkipConfig,
    DataloaderIOConfig,
    DataloaderSkipConfig,
)


@dataclass(frozen=True)
class CompactBatch:
    records: np.ndarray  # uint8 [N, 32], Bulletformat bytes

    def extract_features(
        self,
        feature_set="Full_Threats+PP_3Wide+K32H3",
        ls_keys="piececount:queen:pawnhalf:manyoffi",
        *,
        num_workers=8,
        backend="direct",
        device="cpu",
        pin_memory=False,
    ):
        """Return the existing trainer's seven tensors for these records.

        The direct backend reads packed records into feature-only bitboards;
        the native backend reconstructs Position for comparison. Both use the
        same feature formulas, once per perspective and row.
        """
        from ._native import SparseBatch

        if not isinstance(self.records, np.ndarray) or self.records.dtype != np.uint8:
            raise ValueError("records must be a uint8 numpy array with shape [N, 32]")
        if self.records.ndim != 2 or self.records.shape[1] != 32:
            raise ValueError("records must be a uint8 numpy array with shape [N, 32]")
        if not isinstance(num_workers, int) or isinstance(num_workers, bool) or num_workers < 1:
            raise ValueError("num_workers must be a positive integer")
        if backend not in ("direct", "native"):
            raise ValueError("backend must be 'direct' or 'native'")
        records = np.ascontiguousarray(self.records)
        lib = c_lib.dll
        try:
            extract = lib.get_sparse_batch_from_compact_backend
        except AttributeError as exc:
            raise RuntimeError("Rebuild the native data loader to extract compact features") from exc
        extract.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_bool,
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_size_t,
        ]
        extract.restype = ctypes.POINTER(SparseBatch)
        error = ctypes.create_string_buffer(512)
        batch = extract(
            feature_set.encode("utf-8"),
            ls_keys.encode("utf-8"),
            records.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            records.nbytes,
            num_workers,
            backend == "direct",
            error,
            len(error),
        )
        if not batch:
            raise ValueError(error.value.decode("utf-8", errors="replace"))
        try:
            return batch.contents.get_tensors(device, use_pinned_memory=pin_memory)
        finally:
            lib.destroy_sparse_batch(batch)

    def training_targets(self):
        """Return (is_white, outcome, score) as the existing trainer expects.

        Search-tree positions have no game outcome and use the existing 2.0
        score-only sentinel, even though the Bulletformat result is ordinary.
        """
        is_white = (self.records[:, 30] == 0).astype(np.float32)
        outcome = np.where(self.records[:, 31] != 0, 2.0, self.records[:, 26] / 2.0).astype(
            np.float32
        )
        score_bits = self.records[:, 24].astype(np.uint16) | (
            self.records[:, 25].astype(np.uint16) << 8
        )
        score = score_bits.view(np.int16).astype(np.float32)
        return is_white, outcome, score


class CompactBatchStream:
    """One ordered global stream; filtering uses the existing native predicate.

    Use as a context manager or explicitly close it. This API deliberately
    does not automatically shard by distributed rank: one producer should
    share each batch with all tensor-parallel workers.
    """

    def __init__(
        self, filenames, batch_size, *, num_workers=4, cyclic=False, config=None, io_config=None
    ):
        paths = [str(Path(p)) for p in filenames]
        if not paths or any(not Path(p).is_file() for p in paths):
            raise ValueError("filenames must be a nonempty list of existing binpacks")
        if any(not p.endswith(".binpack") for p in paths):
            raise ValueError("compact batches currently support .binpack files only")
        if batch_size <= 0 or num_workers <= 0:
            raise ValueError("batch_size and num_workers must be positive")
        lib = c_lib.dll
        try:
            create = lib.create_compact_batch_stream
        except AttributeError as exc:
            raise RuntimeError("Rebuild the native data loader to use compact batches") from exc
        create.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.c_int,
            ctypes.c_bool,
            CDataloaderSkipConfig,
            CDataloaderIOConfig,
        ]
        create.restype = ctypes.c_void_p
        lib.fetch_next_compact_batch.argtypes = [ctypes.c_void_p]
        lib.fetch_next_compact_batch.restype = ctypes.c_void_p
        lib.destroy_compact_batch_stream.argtypes = [ctypes.c_void_p]
        lib.destroy_compact_batch.argtypes = [ctypes.c_void_p]
        lib.compact_batch_stream_error.argtypes = [ctypes.c_void_p]
        lib.compact_batch_stream_error.restype = ctypes.c_char_p
        lib.compact_batch_size.argtypes = [ctypes.c_void_p]
        lib.compact_batch_size.restype = ctypes.c_size_t
        lib.compact_batch_records.argtypes = [ctypes.c_void_p]
        lib.compact_batch_records.restype = ctypes.POINTER(ctypes.c_uint8)
        self._lib = lib
        encoded = (ctypes.c_char_p * len(paths))(*(p.encode("utf-8") for p in paths))
        self._handle = create(
            num_workers,
            len(paths),
            encoded,
            batch_size,
            cyclic,
            CDataloaderSkipConfig(config or DataloaderSkipConfig()),
            CDataloaderIOConfig(io_config or DataloaderIOConfig()),
        )
        if not self._handle:
            raise RuntimeError("Could not create compact binpack stream")

    def __iter__(self):
        return self

    def __next__(self):
        if not self._handle:
            raise StopIteration
        batch = self._lib.fetch_next_compact_batch(self._handle)
        if not batch:
            error = self._lib.compact_batch_stream_error(self._handle)
            if error:
                raise RuntimeError(error.decode("utf-8", errors="replace"))
            raise StopIteration
        try:
            n = self._lib.compact_batch_size(batch)
            records = (
                np.ctypeslib.as_array(self._lib.compact_batch_records(batch), shape=(n * 32,))
                .copy()
                .reshape(n, 32)
            )
            return CompactBatch(records)
        finally:
            self._lib.destroy_compact_batch(batch)

    def close(self):
        if self._handle:
            self._lib.destroy_compact_batch_stream(self._handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        if getattr(self, "_handle", None):
            self.close()
