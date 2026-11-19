# ugbio PyPGx

A docker for running [PyPGx](https://github.com/sbslee/pypgx)

## pgx_ngs_pipeline

Runs `pypgx run-ngs-pipeline` for every gene of one sample, `--parallel` genes at a time, then writes the
outputs under the names of the terra_pipeline `pypgx.wdl` (`NGSPipeline` + `ConcatResults`):

```bash
pgx_ngs_pipeline --genes CYP2D6 CYP2C19 --parallel 2 --assembly GRCh38 \
    --variants PASS.vcf.gz --depth-of-coverage depth.zip --control-statistics control.zip \
    --ngs-dir ngs --log-dir logs --output-dir out --tsv out/sample.pypgx.tsv
```

- `ngs/<gene>`: pypgx output (`--force`); `logs/ngs.<gene>.log`: its stdout and stderr.
- `out/<kind>.<gene>.zip` (empty if pypgx wrote none), `out/copy-number-profile.<gene>.png` and
  `out/allele-fraction-profile.<gene>.png` (first PNG, or empty).
- The TSV: one row per gene, the last row of the gene's `results.zip` `data.tsv`.

Every gene runs even if some fail; the command then exits 1, lists the failed genes with their logs and writes nothing to the output directory (no TSV, no per-gene files).
Thread caps (`JAVA_TOOL_OPTIONS`, `OMP_NUM_THREADS`, ...) come from the caller's environment.
