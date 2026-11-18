# ugbio_sequins

Tools for the sequins analysis workflow (`SequinsAnalysis` in terra_pipeline).

## sequins_report

Writes an HTML report stating which sequins spikes were detected and, for each detected one, the SNP/indel
precision, recall and F1 (from the `rtg vcfeval` TP/FP/FN VCFs) and the discordant variants (the FP and FN records, with genotype, quality, filter and DP).

```
sequins_report --base-file-name NAME --min-median-depth 5 \
    --input-crams a.cram b.cram --target-coverage 40 \
    --genomes sequins_clinical sequins_core \
    --median-depths 30 2 --detected true false \
    --tp-baseline-vcfs tpb.vcf.gz --tp-vcfs tp.vcf.gz --fp-vcfs fp.vcf.gz --fn-vcfs fn.vcf.gz
```

The VCF lists hold only the detected genomes, in the order of `--genomes`.
The report also lists the input CRAMs and the coverage the detected genomes were downsampled to.
Output: `report.html` (override with `--output`). Requires `bcftools` on the PATH (included in `ugbio_base`).

Tests: `uv run pytest src/sequins/tests/`

## sequins_igv

Writes an IGV batch file for a detected sequins genome, which shows the reads, the ground-truth VCF and a BED of all
the discordant (FP and FN) positions, and goes to the sequins contig.

```
sequins_igv --genome-name sequins_core --ref-fasta ref.fasta --cram calibrated.cram \
    --truth-vcf truth.vcf.gz --fp-vcf fp.vcf.gz --fn-vcf fn.vcf.gz --output-prefix NAME_sequins_core
```

Outputs `<prefix>.igv_batch.txt` and `<prefix>.discordant.bed`. The batch refers to the files by name, which IGV resolves
relative to the batch file: put the reference (+ `.fai`), the CRAM (+ `.crai`), the truth VCF (+ `.tbi`) and the BED in the
same directory as the batch file, and run it in IGV with Tools > Run Batch Script.
