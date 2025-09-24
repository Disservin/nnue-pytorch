# DDP Data Loader Implementation

This document describes the implementation of Distributed Data Parallel (DDP) support for the NNUE training data loader.

## Problem Statement

Previously, all DDP processes would read the same training data, defeating the purpose of distributed training. Each process should read different, non-overlapping portions of the data for optimal training efficiency.

## Solution Overview

Implemented a round-robin chunk allocation system where:
- Process `i` out of `N` total processes only reads chunks where `chunk_index % N == i`
- This ensures different processes get different data segments
- Scales efficiently with any number of processes

## Implementation Details

### Environment Detection

The data loader automatically detects DDP configuration from environment variables:

```python
def _get_ddp_rank_and_world_size():
    """Get DDP rank and world size from environment variables."""
    rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, world_size
```

- `LOCAL_RANK` or `RANK`: Process rank (0, 1, 2, ...)  
- `WORLD_SIZE`: Total number of processes

### Chunk Allocation Algorithm

In the C++ `CompressedTrainingDataEntryParallelReader`:

```cpp
// Each process only reads chunks that belong to its rank
std::size_t chunk_index = m_currentChunkIndex.fetch_add(1);

// Find the next chunk that belongs to this rank
while ((chunk_index % m_world_size) != m_rank)
{
    chunk_index = m_currentChunkIndex.fetch_add(1);
}
```

This ensures perfect data distribution without overlaps.

### API Changes

All API changes are backward compatible:

```python
# Existing usage (still works)
dataset = SparseBatchDataset(
    feature_set="HalfKAv2", 
    filenames=["data.binpack"],
    batch_size=16384
)

# Optional explicit rank specification
dataset = SparseBatchDataset(
    feature_set="HalfKAv2", 
    filenames=["data.binpack"],
    batch_size=16384,
    rank=0,         # Optional
    world_size=4    # Optional
)
```

## File Changes

### Python Files
- `data_loader/stream.py`: Added rank detection and parameter passing
- `data_loader/dataset.py`: Updated all dataset classes to accept rank/world_size
- `data_loader/_native.py`: Updated C function prototypes

### C++ Files  
- `training_data_loader.cpp`: Updated exported functions to accept rank/world_size
- `lib/nnue_training_data_formats.h`: Added DDP support to parallel reader
- `lib/nnue_training_data_stream.h`: Updated stream constructors

## Usage Examples

### PyTorch Lightning (Automatic)

```python
trainer = L.Trainer(
    devices=4,
    strategy="ddp"
)

# Data loader automatically detects DDP environment
# Each process gets different chunks automatically
```

### Manual DDP Setup

```bash
# Process 0
WORLD_SIZE=4 LOCAL_RANK=0 python train.py

# Process 1  
WORLD_SIZE=4 LOCAL_RANK=1 python train.py

# Process 2
WORLD_SIZE=4 LOCAL_RANK=2 python train.py

# Process 3
WORLD_SIZE=4 LOCAL_RANK=3 python train.py
```

## Data Distribution Examples

### 2 Processes, 10 Chunks
- Process 0: chunks [0, 2, 4, 6, 8]
- Process 1: chunks [1, 3, 5, 7, 9]

### 4 Processes, 12 Chunks  
- Process 0: chunks [0, 4, 8]
- Process 1: chunks [1, 5, 9] 
- Process 2: chunks [2, 6, 10]
- Process 3: chunks [3, 7, 11]

## Benefits

✅ **Perfect Data Distribution**: Each process reads different data with no overlaps

✅ **Efficient Scaling**: Works with any number of processes (2, 4, 8, 16, ...)

✅ **Backward Compatibility**: Existing code works unchanged

✅ **Automatic Detection**: Zero configuration needed with standard DDP setups

✅ **Manual Override**: Can specify rank/world_size explicitly if needed

✅ **Load Balancing**: Even distribution of chunks across all processes

## Testing

The implementation includes comprehensive tests for:
- Environment variable detection  
- Round-robin chunk allocation logic
- Data separation between ranks
- Backward compatibility

All tests pass, confirming correct implementation.

## Migration Guide

**No migration needed!** Existing training scripts will automatically work with DDP when the environment variables are set. The data loader will detect DDP and distribute data appropriately.

For PyTorch Lightning users, simply add `strategy="ddp"` to your trainer and the data will be automatically distributed across processes.