# Indexed container, version 1

A file is one or more **segments** stored back to back. Each segment carries one immutable model and is laid out as below. Every block starts at a fresh rANS state and holds at most 256 records. All integers are unsigned and little-endian. Byte offsets inside a segment, in its index and in its footer's index_start, are zero-based and relative to the segment's first byte. A standalone file is a single segment starting at byte 0, so concatenating files (`cat a.fsqd b.fsqd`) produces a valid multi-segment file, and a writer appending to an existing file adds a segment. Within a segment the final block can be short and all earlier blocks contain exactly 256 records, which makes the segment's record ordinal i directly addressable as its block i/256 and local ordinal i%256. Readers number blocks and records across segments in file order.

```text
segment start + 0   + 16        first block                     segment end - 24
| header            | model     | block 0 | ... | block B-1 | index | footer |
                            ^                           ^
                            offsets[0]                  index_start
```

| Region | Fields in wire order |
|---|---|
| Header, 16 bytes | ASCII FSQD, u16 version=1, u16 flags=0, u64 model length |
| Model | 16 dictionaries, then all normalized tables in [codec order](codec.md) |
| Block | u32 final encoder state, then the forward-readable refill bytes |
| Index | One u64 start offset per block in increasing order |
| Footer, 24 bytes | u64 record count N, u64 index_start, u32 reserved=0, ASCII FEND |

Each dictionary starts with a u8 length in 0..127, followed by that many little-endian u32 words in ID order. Each word has four piece bytes, each in 0..9 or 12. Reject duplicate words within one dictionary. Each probability table is its complete alphabet of little-endian u16 frequencies in symbol order. Validate positivity and sum 1024. The grammar fixes the table count and caps the dictionary sizes, so compute the maximum model length from that grammar before allocating. The model region must be consumed exactly.

### Locating segments

Walk the file backwards. Start with end = file length. Read the footer at end - 24 and calculate B = ceil(N/256) using division and a remainder check. The segment length is index_start + 8*B + 24, computed with checked arithmetic; require it to be at most end. The segment starts at end minus its length and must begin with the header magic. Repeat with end = that start until end reaches exactly 0. A bad footer or header anywhere is an error for the whole file, because the walk cannot locate the segments before it. Segments whose serialized models are byte-identical may share one decoded model.

### Reading a range

Within a segment, for block j fetch offsets[j] and offsets[j+1], using index_start as the last endpoint, and add the segment start to obtain file positions. Validate both bounds, their ordering and a span of at least four bytes. The first block starts at 16 + model_length. An empty segment contains the header, model and footer, with index_start immediately after the model. Reject unknown versions, flags, and reserved bits.

For a large archive, page the index or share a read-only mapping. At 12 billion records the index alone is 375 MB decimal. A positional-read implementation can validate the full index once or validate fetched ranges on demand. Expose the distinction in validation options. Bound untrusted block reads before allocation. The grammar permits at most 3 initial symbols, 16 groups of at most 6 symbols each, and 5 metadata symbols per record. A symbol can consume at most two refill bytes with the specified state bounds. A conservative block bound is 4 + 208 times its record count. Zero-length or overlapping payloads are invalid.

A streaming writer only needs a running offset, the model, the pending block, and an index spool. On finish, flush the partial block, append the index and write the footer. A streaming sink can receive the bytes in order. An indexed reader requires a seekable or positional source because the footer carries the record count. The v1 container includes structural validation only. For archival integrity record a digest outside the file.

### Compatibility and accounting

Publish a new version for a different traversal, alphabet, interpretation of metadata, or entropy-state convention. Dictionaries and probabilities can vary between v1 files. Shards using different models can be concatenated as separate segments. Merging them into a single segment instead requires identical serialized models and a record count that is a multiple of 256 in every shard but the last, or re-encoding.

Compute the exact size of a segment as 16 + model_length + sum(block_lengths) + 8*B + 24; a file is the sum of its segments. Report both whole-file and payload-plus-index bytes divided by N. Model overhead matters for small shards. Random access to one record entails decoding the preceding records in that block, at most 255. Shuffling block IDs with an optional bounded record shuffle gives a loader efficient reads.
