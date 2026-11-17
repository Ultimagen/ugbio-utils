# ugbio_sequins

Tools for the sequins analysis workflow (`SequinsAnalysis` in terra_pipeline).

## sequins_report

Writes an HTML report stating which sequins spikes were detected and, for each detected one, the SNP/indel
precision, recall and F1 (from the `rtg vcfeval` TP/FP/FN VCFs) and the discordant variants.

```
sequins_report --base-file-name NAME --min-median-depth 5 \
    --genomes sequins_clinical sequins_core \
    --median-depths 30 2 --mean-depths 31 3 --detected true false \
    --tp-baseline-vcfs tpb.vcf.gz --tp-vcfs tp.vcf.gz --fp-vcfs fp.vcf.gz --fn-vcfs fn.vcf.gz \
    --discordant-tables discordant.tsv
```

The VCF and table lists hold only the detected genomes, in the order of `--genomes`.
Output: `NAME.sequins_report.html`. Requires `bcftools` on the PATH (included in `ugbio_base`).

Tests: `uv run pytest src/sequins/tests/`
