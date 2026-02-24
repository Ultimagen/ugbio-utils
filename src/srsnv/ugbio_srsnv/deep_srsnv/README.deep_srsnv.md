# deep_srsnv (BAM-native CNN)

`deep_srsnv` is a BAM-native training path for SRSNV that uses full-read channels instead of engineered parquet-only features.

## Input contract (v1)

- Fixed tensor length `L=300`.
- One read per sample.
- Multi-channel 1D representation (`channels x 300`):
  - per-position channels: `read_base`, `ref_base` (gapped by CIGAR), base quality, `tp`, `t0`, focus one-hot, `softclip_mask`, valid-mask
  - repeated constant channels: strand, MAPQ, RSQ (`rq`), trimming reason (`tm`), `st`, `et`, mixed-indicator
- Padding/truncation:
  - shorter reads are padded
  - longer reads are truncated to 300

## Training entrypoint

Use:

`srsnv_dnn_bam_training`

Required arguments:

- `--positive-bam`, `--negative-bam`
- `--positive-parquet`, `--negative-parquet`
- `--training-regions`
- `--output`

For split parity with XGBoost:

- generate or pass `--split-manifest-in/--split-manifest-out`
- holdout chromosomes are typically `chr21,chr22`
- for single-model mode (no K-fold), use:
  - `--single-model-split`
  - `--val-fraction 0.1`
  - `--split-hash-key RN`
  - with `chr21,chr22` as holdout test set

## Fast preprocess (cached)

The training entrypoint preprocesses parquet+BAM into a cached records stream file by default and trains by streaming cached chunks.

- `--preprocess-cache-dir`: persistent cache location (default: `<output>/deep_srsnv_cache`)
- `--preprocess-num-workers`: parallel shard workers (default: `min(cores-2, 16)`)
- `--preprocess-max-ram-gb`: RAM budget for shard planning (default: `48`)
- `--preprocess-batch-rows`: requested shard size before RAM-cap adaptation
- `--preprocess-storage-mode`: `single_file` (default) or `shards`
- `--preprocess-dry-run`: print/write planned preprocess settings without training
- `--encoder-vocab-source`: `known` (default, schema-based dictionaries) or `scan` (rebuild from cached records)
- `--loader-num-workers`: dataloader workers for training/prediction
- `--loader-prefetch-factor`: batches prefetched per worker
- `--loader-pin-memory`: enable pinned host memory for faster H2D copies
- `--use-amp`: mixed precision training on CUDA
- `--use-tf32`: enable TF32 + cudnn benchmark on Ampere+
- `--autotune-batch-size`: use larger default effective batch for higher GPU occupancy

Cache key includes:

- positive/negative parquet paths + mtimes
- positive/negative BAM paths + mtimes
- split manifest content
- tensor spec (`L`, channels) and batch-row setting

Behavior:

- first run builds cache and writes an index with telemetry
- rerun with matching key is a cache hit and skips BAM/parquet preprocessing
- if worker count and requested shard size exceed RAM budget, effective shard size is reduced automatically

## GPU utilization tips

- Start with:
  - `--use-amp --use-tf32 --loader-pin-memory`
  - `--loader-num-workers 8 --loader-prefetch-factor 4`
  - `--batch-size 256` (or `--autotune-batch-size`)
- Check `training_runtime_metrics` in `*.srsnv_dnn_metadata.json` for:
  - `wait_seconds` vs `compute_seconds`
  - `samples_per_second`
