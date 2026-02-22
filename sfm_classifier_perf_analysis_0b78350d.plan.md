---
name: SFM Classifier Perf Analysis
overview: Create a Jupyter notebook that fetches AWS HealthOmics task logs for 12 classifier runs, parses [PERF] metrics, identifies architectural bottlenecks, and proposes alternative frameworks/approaches for better performance.
todos:
  - id: fetch-logs
    content: Create notebook cells for fetching task logs from 12 HealthOmics runs using get_log_for_task (all tasks, not just failed)
    status: completed
  - id: perf-parser
    content: Write [PERF] log line parser that extracts stage/phase/metrics from task log files into a structured DataFrame
    status: completed
  - id: timing-analysis
    content: Build stage-level timing breakdown and sub-stage deep dive visualizations
    status: completed
  - id: resource-analysis
    content: "Build resource analysis from [PERF] logs: CPU efficiency, memory growth, I/O volumes per stage"
    status: completed
  - id: arch-bottlenecks
    content: "Build architectural bottleneck analysis: subprocess overhead, format conversion costs, tool chain analysis"
    status: completed
  - id: cross-sample
    content: "Build cross-sample comparison: scaling behavior across 12 samples"
    status: completed
  - id: recommendations
    content: Write data-backed architectural recommendations with alternative framework proposals
    status: completed
isProject: false
---

# SFM Classifier Performance Analysis

## Goal

Identify **architectural bottlenecks** in the somatic featuremap classifier pipeline and propose **alternative frameworks and design approaches** for better performance and resource utilization. This is not about tuning scatter/region parameters -- it is about questioning the fundamental tool choices (awk, subprocess pipelines, format conversions) and proposing better alternatives backed by data.

## Context

### Pipeline Stages (4 stages, each with [PERF] logs)

1. **filter_and_annotate_tr** -- filter VCF + TR annotation via bedtools
2. **read_vcf_with_aggregation** -- VCF-to-Parquet (parallel region processing) + pileup feature calculation
3. **run_classifier** -- Polars-to-Pandas conversion + XGBoost predict
4. **annotate_vcf_with_xgb_proba** -- bcftools annotate to write XGB_PROBA back to VCF

### Current Tool Chain (end-to-end)

```
VCF → bcftools view (filter) → bcftools query + bedtools closest + cut + sort + bgzip (TR annotation)
    → bcftools annotate (add TR to VCF)
    → [per region, parallel via ProcessPoolExecutor]:
        bcftools query (per sample) → awk (explode/aggregate lists) → TSV string → Polars DataFrame → Parquet part-file
    → Polars lazy merge (concat parts) → Single Parquet
    → Polars read_parquet → PILEUP feature calculation (Polars expressions)
    → Polars DataFrame.to_pandas() → XGBoost.predict()
    → Polars Series → write_csv (TSV) → bgzip → tabix → bcftools annotate → Final VCF
```

### Key Architectural Choices Under Scrutiny

- **awk scripts** for VCF list field processing (explode/aggregate) inside subprocess pipelines
- **bcftools query | awk** subprocess chain per region per sample (2N subprocesses for N regions x 2 samples)
- **Polars-to-Pandas conversion** for XGBoost (full DataFrame copy)
- **TSV file round-trip** for VCF annotation (DataFrame → TSV → bgzip → tabix → bcftools annotate)
- **ProcessPoolExecutor** with subprocess workers (Python parallelism wrapping shell parallelism)
- **bedtools closest** via shell pipe for TR annotation

### [PERF] Log Metrics Available

All `[PERF]` lines follow: `[PERF] [stage:phase] message | {JSON}` with:

- `elapsed_sec`: wall-clock time
- `mem_mb`, `mem_delta_mb`: memory usage and growth (from `psutil`)
- `cpu_user_sec`, `cpu_system_sec`: CPU time breakdown
- `io_read_mb`, `io_write_mb`: I/O volumes
- `input_size_mb`, `output_size_mb`: file sizes
- `variant_count`, `row_count`: data volumes
- `n_threads`: thread count
- `throughput_regions_per_sec`: processing rate

Logs are in: [somatic_featuremap_classifier.py](src/featuremap/ugbio_featuremap/somatic_featuremap_classifier.py), [somatic_featuremap_utils.py](src/featuremap/ugbio_featuremap/somatic_featuremap_utils.py), [featuremap_to_dataframe.py](src/featuremap/ugbio_featuremap/featuremap_to_dataframe.py), [vcf_utils.py](src/core/ugbio_core/vcf_utils.py).

## Data Source

Only **task logs** (containing `[PERF]` lines) are needed. No MONITORING or manifest logs.

- Fetch via `get_log_for_task(run_id, failed=False)` from [get_omics_log.py](src/omics/ugbio_omics/get_omics_log.py)
- Filter task logs to only those for the classifier task (by task name)
- Save to `/data/Runs/erf_analysis/<run_id>/`

## Notebook Structure

Place at `src/featuremap/notebooks/sfm_classifier_benchmark.ipynb`.

### Section 1: Configuration and Log Fetching

- User provides list of 12 run IDs
- Fetch all task logs using `get_log_for_task(run_id, failed=False)` for each run
- Cache locally so re-runs skip fetching

### Section 2: [PERF] Log Parser

Write a `parse_perf_logs(log_dir) -> pd.DataFrame` function:

- Scan all `.log` files in the output directory
- Extract lines containing `[PERF]`
- Parse with regex: `\[PERF\]\s+\[([^:]+):([^\]]+)\]\s+(.+?)\s+\|\s+(\{.+\})`
  - Group 1: stage name, Group 2: phase (entry/exit/start/end), Group 3: message, Group 4: JSON payload
- Flatten JSON payload (including nested `resources` dict)
- Add `run_id`, `task_name`, `task_id` from filename
- Return consolidated DataFrame: one row per [PERF] event across all runs and shards

### Section 3: Stage-Level Timing Breakdown

**Purpose**: Identify which of the 4 stages dominates wall-clock time.

- Filter to `:exit` events which have `elapsed_sec`
- **Stacked bar chart**: per-shard time broken down by stage (sampled across runs)
- **Box plot**: distribution of each stage's elapsed_sec across all shards
- **Pie chart**: average time share per stage
- **Table**: mean, median, p95, max elapsed_sec per stage

### Section 4: Sub-Stage Deep Dive

**Purpose**: Within the dominant stage(s), identify which sub-operations are slowest.

For `read_vcf_with_aggregation`:

- Break into: `vcf_to_parquet` time, `_run_region_jobs` time, `calculate_pileup_features` time, post-processing time
- Identify: is the bottleneck in subprocess I/O (bcftools+awk), Polars processing, or parquet merge?

For `filter_and_annotate_tr`:

- Break into individual `_run_shell_command` calls (bcftools view, bedtools closest pipeline)
- Measure: `cpu_user_sec` vs `elapsed_sec` per command (CPU-bound vs I/O-bound?)

For `annotate_vcf_with_xgb_proba`:

- Break into: TSV write, bgzip, tabix, bcftools annotate
- Measure: file sizes at each step

For `run_classifier`:

- Break into: Polars-to-Pandas conversion, model load, XGBoost predict
- Measure: memory spike from conversion (`mem_delta_mb`)

### Section 5: Resource Efficiency Analysis (from [PERF] logs)

**Purpose**: Quantify how well the current architecture uses CPU, memory, and I/O.

- **CPU efficiency per stage**: `cpu_user_sec / elapsed_sec` -- values <<1 mean idle/waiting (I/O bound or subprocess overhead); values >1 mean multi-threaded work
- **System vs user CPU**: high `cpu_system_sec` relative to `cpu_user_sec` indicates subprocess/fork overhead
- **Memory growth waterfall**: cumulative `mem_delta_mb` across stages -- identify if memory accumulates
- **I/O volume analysis**: `io_read_mb` and `io_write_mb` per stage and per shell command -- identify redundant I/O from intermediate files
- **Subprocess overhead**: count the number of shell commands executed per shard, correlate total subprocess `elapsed_sec` vs total pipeline time

### Section 6: Architectural Bottleneck Identification

**Purpose**: Map performance data to specific architectural decisions.

Analyze and visualize:

- **awk overhead**: time spent in bcftools+awk pipelines vs. total vcf_to_parquet time. What fraction of processing is awk string manipulation?
- **Subprocess spawn cost**: number of subprocess invocations per shard, cumulative process startup time
- **Format conversion cost**: time and memory cost of Polars-to-Pandas, TSV round-trips for annotation
- **I/O amplification**: ratio of total bytes written (intermediate files) vs. final output size
- **Parallelism efficiency**: `_run_region_jobs` elapsed_sec vs. sum of individual region times (overhead of ProcessPoolExecutor + subprocess per region)

### Section 7: Cross-Sample Scaling

**Purpose**: Understand how performance scales with input size.

- **Scatter plot**: total pipeline time vs. input VCF size (or variant count) across 12 samples
- **Per-stage scaling**: scatter plots of each stage's time vs. its input size
- Fit trend lines to identify linear vs. superlinear scaling (indicates algorithmic issues)

### Section 8: Architectural Recommendations

Markdown cells with data-backed recommendations. The analysis from sections 3-7 will inform which of these are most impactful:

**Candidate recommendations to evaluate against the data:**

1. **Replace awk with Polars native list operations**: Polars has `explode()`, `list.mean()`, `list.max()`, `list.min()`, `list.len()` -- these could replace `explode_lists.awk` and `aggregate_lists.awk` entirely, eliminating subprocess overhead and TSV string intermediaries
2. **Replace bcftools query + awk with cyvcf2 or pysam**: Read VCF fields directly into Polars/numpy arrays using `cyvcf2.VCF` (C-backed, fast) instead of spawning `bcftools query | awk` per region per sample
3. **Eliminate Polars-to-Pandas conversion**: XGBoost supports numpy arrays directly -- extract `.to_numpy()` from Polars columns instead of full DataFrame conversion
4. **In-memory VCF annotation instead of TSV round-trip**: Use `pysam.VariantFile` to write the annotated VCF directly from the DataFrame, avoiding the TSV → bgzip → tabix → bcftools annotate chain
5. **Replace ProcessPoolExecutor+subprocess with streaming**: If bcftools+awk are eliminated, the entire VCF-to-DataFrame step could be a single-pass streaming read with cyvcf2, removing the need for region chunking and process pooling
6. **Replace bedtools closest with pyranges**: The TR annotation step (bedtools closest via shell pipe) could use `pyranges.nearest()` in pure Python, avoiding subprocess overhead

Each recommendation will include:

- Which [PERF] metric supports the change (e.g., "awk pipelines account for X% of vcf_to_parquet time")
- Estimated impact (based on measured overhead)
- Implementation complexity assessment

## Dependencies

- `boto3` (available)
- `pandas`, `plotly` (already used in workspace)
- `ugbio_omics` utilities (workspace dependency)
- Standard `json`, `re`, `pathlib`

No new dependencies needed.
