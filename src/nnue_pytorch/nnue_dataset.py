import numpy as np
import torch
from torch.utils.data import Dataset
import nnue_pytorch._core as _core

class SparseBatch(_core.SparseBatch): ...

class FenBatchProvider:
    def __init__(
        self,
        filename,
        cyclic,
        num_workers,
        batch_size=None,
        filtered=False,
        random_fen_skipping=0,
        early_fen_skipping=-1,
        wld_filtered=False,
        param_index=0):

        self.filename = filename
        self.cyclic = cyclic
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.filtered = filtered
        self.wld_filtered = wld_filtered
        self.random_fen_skipping = random_fen_skipping
        self.early_fen_skipping = early_fen_skipping
        self.param_index = param_index

        batch_size = batch_size or 1
        self.stream = _core.create_fen_batch_stream(
            self.num_workers, [self.filename], batch_size, cyclic, filtered, 
            random_fen_skipping, wld_filtered, early_fen_skipping, param_index)

    def __iter__(self):
        return self

    def __next__(self):
        v = self.stream.next()
        if v:
            fens = v.get_fens()
            _core.destroy_fen_batch(v)
            return fens
        else:
            raise StopIteration

    def __del__(self):
        _core.destroy_fen_batch_stream(self.stream)

class TrainingDataProvider:
    def __init__(
        self,
        feature_set,
        create_stream,
        destroy_stream,
        fetch_next,
        destroy_part,
        filenames,
        cyclic,
        num_workers,
        batch_size=None,
        filtered=False,
        random_fen_skipping=0,
        wld_filtered=False,
        early_fen_skipping=-1,
        param_index=0,
        device='cpu'):

        self.feature_set = feature_set
        self.create_stream = create_stream
        self.destroy_stream = destroy_stream
        self.fetch_next = fetch_next
        self.destroy_part = destroy_part
        self.filenames = filenames
        self.cyclic = cyclic
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.filtered = filtered
        self.wld_filtered = wld_filtered
        self.random_fen_skipping = random_fen_skipping
        self.param_index = param_index
        self.device = device

        batch_size = batch_size or 1
        self.stream = self.create_stream(
            self.feature_set, self.num_workers, self.filenames, batch_size, cyclic,
            filtered, random_fen_skipping, wld_filtered, early_fen_skipping, param_index)

    def __iter__(self):
        return self

    def __next__(self):
        v = self.fetch_next(self.stream)
        if v:
            tensors = v.get_tensors(self.device)
            self.destroy_part(v)
            return tensors
        else:
            raise StopIteration

    def __del__(self):
        self.destroy_stream(self.stream)

class SparseBatchProvider(TrainingDataProvider):
    def __init__(self, feature_set, filenames, batch_size, cyclic=True, num_workers=1,
                 filtered=False, random_fen_skipping=0, wld_filtered=False,
                 early_fen_skipping=-1, param_index=0, device='cpu'):
        super(SparseBatchProvider, self).__init__(
            feature_set,
            _core.create_sparse_batch_stream,
            _core.destroy_sparse_batch_stream,
            _core.fetch_next_sparse_batch,
            _core.destroy_sparse_batch,
            filenames,
            cyclic,
            num_workers,
            batch_size,
            filtered,
            random_fen_skipping,
            wld_filtered,
            early_fen_skipping,
            param_index,
            device)

class SparseBatchDataset(torch.utils.data.IterableDataset):
    def __init__(self, feature_set, filenames, batch_size, cyclic=True, num_workers=1,
                 filtered=False, random_fen_skipping=0, wld_filtered=False,
                 early_fen_skipping=-1, param_index=0, device='cpu'):
        super(SparseBatchDataset, self).__init__()
        self.feature_set = feature_set
        self.filenames = filenames
        self.batch_size = batch_size
        self.cyclic = cyclic
        self.num_workers = num_workers
        self.filtered = filtered
        self.random_fen_skipping = random_fen_skipping
        self.wld_filtered = wld_filtered
        self.early_fen_skipping = early_fen_skipping
        self.param_index = param_index
        self.device = device

    def __iter__(self):
        return SparseBatchProvider(
            self.feature_set, self.filenames, self.batch_size, cyclic=self.cyclic,
            num_workers=self.num_workers, filtered=self.filtered,
            random_fen_skipping=self.random_fen_skipping, wld_filtered=self.wld_filtered,
            early_fen_skipping=self.early_fen_skipping, param_index=self.param_index,
            device=self.device)

class FixedNumBatchesDataset(Dataset):
    def __init__(self, dataset, num_batches):
        super(FixedNumBatchesDataset, self).__init__()
        self.dataset = dataset
        self.iter = iter(self.dataset)
        self.num_batches = num_batches

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        return next(self.iter)

def make_sparse_batch_from_fens(feature_set, fens, scores, plies, results):
    return _core.get_sparse_batch_from_fens(feature_set, fens, scores, plies, results)

print(SparseBatch())