# ugbio_consensus

Performance & duplex reports for the **ReadFuserAlignSort** (consensus tool)
pipeline at Ultima Genomics.

The consensus step (`read_fuser`) fuses the reads of each UMI/MI family into a
single consensus read and records, on that read:

```
rn:Z:<comma-separated list of the fused query names>
nf:i:<n_forward_strand_reads>
nr:i:<n_reverse_strand_reads>
```

i.e. which reads were fused, and how many of them were on the forward (+) and
reverse (−) strand. From these tags the report classifies every consensus read
and measures family size **directly** (no MI re-grouping needed):

| Category | Condition | Family size |
|----------|-----------|-------------|
| both-strands **duplex** | `nf > 0` and `nr > 0` | `nf + nr` |
| **single-strand** duplicate | exactly one of `nf`/`nr` is 0 | the non-zero count |
| **singleton** / pass-through | no `rn` tag | 1 |

`nf + nr` equals the number of names in `rn`, so the two encodings agree on
family size; the report cross-checks this and warns on a mismatch.

> **A consensus read is identified by the presence of `rn`, never by a strand
> tag.** Trimmer already emits `rs:i` ("start position in input of segment ...",
> see the `@CO` lines in the input header) on the *input* reads, and that tag
> survives onto the reads the consensus step passes through unchanged. An earlier
> read_fuser wrote the strand counts as `fs`/`rs`, which collided with it and made
> every pass-through read look like a consensus read; the counts were renamed to
> `nf`/`nr` to break the collision. Keying off `rn` is collision-proof and also
> handles the fact that `nf` is simply *absent* (not zero) on pass-through reads —
> a parquet reader would fill that absence as `0`.

## What the report summarises (per sample)

- **Sorter QC** — alignment, duplication and coverage metrics from
  `sorter_stats_csv` (post-consensus).
- **Duplex family metrics** — average MI-family size *and* covered depth for
  duplex families and for single-strand duplicate families, from the `nf`/`nr` tags.
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

# ReadFuserAlignSort three-way comparison: input vs singletons vs consensus CRAM.
# Only the consensus CRAM carries rn/nf/nr, so it is the only one given a cram=
# (the others contribute alignment metrics alone, and need no localised CRAM).
# --no-summary drops the median-across-samples table, meaningless across these three.
consensus_report \
    --sample name=Z0315.input sorter_stats_csv=input.csv sorter_stats_json=input.json \
    --sample name=Z0315.singletons sorter_stats_csv=sing.csv sorter_stats_json=sing.json \
    --sample name=Z0315.consensus cram=cons.cram crai=cons.cram.crai \
        sorter_stats_csv=cons.csv sorter_stats_json=cons.json \
        consensus_log=Z0315.consensus.stdout.log \
    --reference ref.fasta --no-summary --output comparison.html

# b37 reference: scan chromosome 20 under its b37 name
consensus_report ... --duplex-chrom 20 --output report.html
```

Sample rows appear in the order the `--sample` blocks are given, so the
input → singletons → consensus reading order above is preserved in the table.

Outputs (alongside `--output`):

- `<output>.html` — the self-contained HTML report.
- `<output>_per_sample.csv` — full per-sample metrics table (provenance).
- `<output>_manifest.csv` — resolved input paths (provenance).

## Modules

| Module | Role |
|--------|------|
| `duplex_metrics.py` | Parse `rn`/`nf`/`nr`, classify families, family size + coverage per category (with an `MI`-tag fallback). |
| `on_target.py` | Genome-wide and optional on-target coverage from a bedGraph + targets BED. |
| `consensus_log.py` | Parse the consensus tool stdout log for performance counters. |
| `consensus_report.py` | CLI orchestration: read local inputs, compute metrics, write the HTML report + CSVs. |

## Requirements

- `bedtools` and `samtools` on `PATH` (used for BED handling and CRAM decoding).
- The reference FASTA matching the run's `reference_genome`.
