from typing import List, Optional, Callable, Any, overload
from typing_extensions import Protocol

class SparseBatch:
    def __repr__(self) -> str: ...
    def get_tensors(self, device: str) -> List[Any]: ...

class FenBatch:
    def __repr__(self) -> str: ...
    def get_fens(self) -> List[str]: ...

class SparseBatchStream:
    def next(self) -> Optional[SparseBatch]: ...

class FenBatchStreamBase:
    def next(self) -> Optional[FenBatch]: ...

class FenBatchStream(FenBatchStreamBase):
    def next(self) -> Optional[FenBatch]: ...

def get_sparse_batch_from_fens(
    feature_set: str,
    fens: List[str],
    scores: List[int],
    plies: List[int],
    results: List[int],
) -> SparseBatch: ...
def create_fen_batch_stream(
    concurrency: int,
    filenames: List[str],
    batch_size: int,
    cyclic: bool,
    filtered: bool = False,
    random_fen_skipping: int = 0,
    wld_filtered: bool = False,
    early_fen_skipping: int = 0,
    param_index: int = 0,
) -> FenBatchStream: ...
def create_sparse_batch_stream(
    feature_set: str,
    concurrency: int,
    filenames: List[str],
    batch_size: int,
    cyclic: bool,
    filtered: bool = False,
    random_fen_skipping: int = 0,
    wld_filtered: bool = False,
    early_fen_skipping: int = 0,
    param_index: int = 0,
) -> SparseBatchStream: ...
def destroy_sparse_batch(batch: SparseBatch) -> None: ...
def destroy_fen_batch(batch: FenBatch) -> None: ...
def destroy_sparse_batch_stream(stream: SparseBatchStream) -> None: ...
def destroy_fen_batch_stream(stream: FenBatchStream) -> None: ...
def fetch_next_sparse_batch(stream: SparseBatchStream) -> Optional[SparseBatch]: ...
def fetch_next_fen_batch(stream: FenBatchStream) -> Optional[FenBatch]: ...
