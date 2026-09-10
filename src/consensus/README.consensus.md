# ugbio_consensus

Duplicate marking and performance & duplex reports for the **ReadFuserAlignSort**
(consensus tool) pipeline at Ultima Genomics.

| Entry point | Runs | Role |
|-------------|------|------|
| `mark_duplicates_mi` | *before* `read_fuser` | Assign `MI`/`DS` duplicate families, and with `--use-umi` the `CS` cross-strand duplex tag. |
| `consensus_report` | *after* the pipeline | Per-sample HTML report of alignment, duplication, coverage and duplex family metrics. |

## mark_duplicates_mi

Marks duplicate families on a coordinate-sorted BAM/CRAM and writes, on every
record of a family:

```
MI:Z:<run-id>-<10-digit family id>    the duplicate family (molecular identifier)
DS:i:<family size>                    number of pairs in that family
CS:Z:<run-id>-<10-digit family id>    --use-umi only: the cross-strand duplex group
```

`--use-umi` adds `(R1.u5, R2.u3)` to the dedup key alongside the position, so the
two strands of one duplex molecule land in **separate** `MI` families — which is
what we want, since each strand is consensus-called on its own. The cost is that
the duplex relationship is erased: `read_fuser` groups strictly on `MI`, so
`duplex_consensus` would be 0 by construction. `CS` restores the link.

Two clusters are cross-strand partners iff they share the positional key, their
UMI pairs are exact mutual reverses, **and** they sit on opposite strands. That
last term is read off the FLAG (`read.is_reverse`), not off which mate starts
further left: on a real library the two disagree for 10.5 % of pairs, because
fragments shorter than the read dovetail.

`CS` is **total** — a cluster with no partner gets `CS` equal to its own `MI`
rather than no tag at all. That matters downstream, because `demux --umi=CS`
treats a missing tag as the empty value, and every `CS`-less read at a position
would then collapse into one duplicate family. So `CS == MI` reads as "clustered,
but no cross-strand partner was found", and `CS != MI` as "duplex partner found".

```bash
# whole-file
mark_duplicates_mi input.bam output.bam --use-umi

# sharded, which is what the WDL task runs (--jobs = cpu, shards = panel intervals)
mark_duplicates_mi input.cram output.cram --use-umi \
    --reference Homo_sapiens_assembly38.fasta \
    --jobs 16 --regions xgen-pan-cancer-targets.hg38.bed
```

Family-size histogram goes to stdout; `--stats-only` skips writing the output file.
Re-marking is idempotent, and re-marking **without** `--use-umi` strips any `CS`
the input carried (a stale `CS` claims a duplex link the current key does not
support, which is worse than no `CS`).

### Downstream contract (BIOIN-3068)

`CS` only reaches the consensus reads if every stage is told to carry it:

1. `read_fuser --umi-tags u3,u5,CS` — the default list is `u5,u3`, so **without
   this `CS` is silently dropped** and the run still completes green.
2. `sorter_params`: `umi_tag: "CS"` and `mark_duplicates_ends_read_uncertainty: 50`.
   `umi_tag` must be `CS` **alone** — `read_fuser` copies `u5`/`u3` onto each
   consensus read from its own strand's original read, and the two strands carry
   them swapped, so adding them to the key splits exactly the pair `CS` exists to
   join. The wide 50 bp window is needed because the two strands' consensus reads
   routinely end ~27 bp apart, and it is safe only because `CS` is in the key: no
   `(CS, alignment strand)` cell holds more than 2 reads, so there is nothing to
   over-merge. Measured on 36,591 consensus reads: default recovers 424 of 3,423
   true pairs (12.4 %), `ends_read_uncertainty: 50` recovers 3,307 (96.6 %),
   precision 100 % both ways.
3. For SRSNV/DeepSRSNV, add `"CS:Z"` to `cram_tags_to_copy`.

> ⚠️ This module is verified byte-for-byte against a 12,670,749-record baseline
> (`MI`/`DS` identical to the pre-`CS` output, 55,884 `CS` links, 0 gained / 0
> lost). Do not refactor it — including "just" reordering the `set_tag` calls,
> which fixes the emitted aux order as `MI, DS, CS` — without re-running
> `/data/Runs/BIOIN-3068/cmp_stream.py` and `cs_links.py`.

## consensus_report

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
| `mark_duplicates_mi.py` | Duplicate families (`MI`/`DS`) and the `CS` cross-strand duplex tag; whole-file and sharded writers. |
| `duplex_metrics.py` | Parse `rn`/`nf`/`nr`, classify families, family size + coverage per category (with an `MI`-tag fallback). |
| `on_target.py` | Genome-wide and optional on-target coverage from a bedGraph + targets BED. |
| `consensus_log.py` | Parse the consensus tool stdout log for performance counters. |
| `consensus_report.py` | CLI orchestration: read local inputs, compute metrics, write the HTML report + CSVs. |

## Requirements

- `bedtools` and `samtools` on `PATH` (used for BED handling and CRAM decoding).
- The reference FASTA matching the run's `reference_genome`.
