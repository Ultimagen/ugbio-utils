# ugbio_consensus

Performance & duplex reports for the **ReadFuserAlignSort** (consensus tool)
pipeline at Ultima Genomics.

The consensus step (`read_fuser`) fuses the reads of each UMI/MI family into a
single consensus read and records, on that read, an `rs:B:i` tag:

```
rs = [n_forward_strand_reads, n_reverse_strand_reads]
```

i.e. how many original reads on the forward (+) and reverse (−) strand were
fused. From this single tag the report classifies every consensus read and
measures family size **directly** (no MI re-grouping needed):

| Category | `rs` condition | Family size |
|----------|----------------|-------------|
| both-strands **duplex** | both entries > 0 | `n_fwd + n_rev` |
| **single-strand** duplicate | exactly one entry is 0 | the non-zero entry |
| **singleton** / pass-through | no `rs` tag | 1 |

`sum(rs)` equals the length of the `rn` (fused read-name) list, so the two tag
encodings agree on family size.

## What the report summarises (per sample)

- **Sorter QC** — alignment, duplication and coverage metrics from
  `sorter_stats_csv` (post-consensus).
- **Duplex family metrics** — average MI-family size *and* covered depth for
  duplex families and for single-strand duplicate families, from the `rs` tag.
  These are scanned over one chromosome (`--duplex-chrom`, default `chr20`) — a
  representative sample that avoids reading the whole (very large) CRAM. When a
  targets BED is given, the scan is restricted to the targeted intervals on that
  chromosome.
- **On-target metrics (optional)** — when a `--targets` BED is supplied
  (e.g. an exome capture BED): on-target rate and on-target mean coverage from
  the `bedgraph_mapq0` coverage track. Omit `--targets` for genome-wide coverage
  only. The report is target-agnostic; the exome case is simply "targets given".
- **Consensus tool performance** — when a `--consensus-log` is supplied, the
  counters parsed from the consensus tool stdout log.

All inputs are **local files** — the report does no S3/DB access.

## Usage

```bash
# Single sample, with an exome targets BED (Quotient case)
consensus_report \
    --name Z0315 \
    --cram Z0315.cram --crai Z0315.cram.crai \
    --sorter-stats-csv Z0315.csv --sorter-stats-json Z0315.json \
    --bedgraph Z0315_0.bedGraph.gz --consensus-log Z0315.consensus.stdout.log \
    --reference /data/Runs/genomes/hg38/ref_gen/Homo_sapiens_assembly38.fasta \
    --targets Twist_Alliance_Clinical_Research_Exome_hg38.bed \
    --output report.html

# Multiple samples: one repeatable --sample key=value block per sample
consensus_report \
    --sample name=Z0315 cram=Z0315.cram sorter_stats_csv=Z0315.csv \
        sorter_stats_json=Z0315.json bedgraph=Z0315_0.bedGraph.gz \
    --sample name=Z0316 cram=Z0316.cram sorter_stats_csv=Z0316.csv \
        sorter_stats_json=Z0316.json bedgraph=Z0316_0.bedGraph.gz \
    --reference ref.fasta --targets exome.bed --output run_report.html

# b37 reference: scan chromosome 20 under its b37 name
consensus_report ... --duplex-chrom 20 --output report.html
```

Outputs (alongside `--output`):

- `<output>.html` — the self-contained HTML report.
- `<output>_per_sample.csv` — full per-sample metrics table (provenance).
- `<output>_manifest.csv` — resolved input paths (provenance).

## Modules

| Module | Role |
|--------|------|
| `duplex_metrics.py` | Parse `rs:B:i`, classify families, family size + coverage per category (with an `MI`-tag fallback). |
| `on_target.py` | Genome-wide and optional on-target coverage from a bedGraph + targets BED. |
| `consensus_log.py` | Parse the consensus tool stdout log for performance counters. |
| `consensus_report.py` | CLI orchestration: read local inputs, compute metrics, write the HTML report + CSVs. |

## Requirements

- `bedtools` and `samtools` on `PATH` (used for BED handling and CRAM decoding).
- The reference FASTA matching the run's `reference_genome`.
