# ugbio_mrd

This module includes MRD (Minimal Residual Disease) python scripts and utils for bioinformatics pipelines.

## Statistical Detection

MRD detection is based on a Binomial test comparing the observed supporting read count at patient signature loci against a background noise model derived from synthetic (db\_control) signatures.

**Detection p-value:** `P(X ≥ observed_reads | Binom(n_effective, p_err))`
where `n_effective = signature_size × mean_coverage × denom_ratio` and `p_err` is the background error rate estimated from synthetic controls via Jeffreys prior.

### Personal LOD

The **personal Limit of Detection (LOD)** is the minimum tumor fraction (TF) at which a sample would be detected with ≥ 95% probability (recall), given this patient's specific assay parameters. It is personal because it depends on the individual signature size, mean coverage, and measured noise rate.

**Derivation:**

1. **Detection threshold** `n_th`: the smallest read count where the null hypothesis is rejected at FPR = 5%:

$$n_{th} = \min\{k : P(X \geq k \mid \mathrm{Binom}(N,\, p_{err})) < 0.05\}$$

2. **LOD** at 95% recall: the smallest TF such that a true positive sample crosses the threshold 95% of the time:

$$\mathrm{LOD} = \min\{\mathrm{TF} : P(X \geq n_{th} \mid \mathrm{Binom}(N,\, p_{err} + \mathrm{TF})) \geq 0.95\}$$

where $N = \text{signature\_size} \times \text{mean\_coverage} \times \text{denom\_ratio}$.

The LOD decreases (improves) with larger signature size, higher coverage, or lower noise rate. It is `None` when no threshold satisfies the FPR constraint (e.g., signature too small or coverage too low).

## QC Checks

Four QC checks are displayed above the Assay Metrics in the report. They do not force an Indeterminate call — they are informational flags.

| Check | Threshold | Rationale |
|---|---|---|
| Signature size | ≥ 500 loci | Too few loci reduce statistical power |
| Mean coverage | ≥ 15× | Low coverage inflates noise rate variance |
| Synthetic controls | ≥ 20 | Fewer controls make null distribution unreliable |
| No multiple read support enrichment | Enrichment prob ≥ 1% | See below |

### No Multiple Read Support Enrichment

This check detects unexpectedly high multi-read loci counts, which may indicate contamination, index hopping, or signature artefacts unrelated to true ctDNA signal.

**Step 1 — per-locus expectation (Poisson):**

Each locus independently receives λ = `mean_coverage × tumor_vaf` reads on average. The probability a single locus has ≥ 2 supporting reads is:

$$p = 1 - (1 + \lambda)\,e^{-\lambda}$$

**Step 2 — across-loci test (Binomial):**

The count of loci with ≥ 2 reads out of `signature_size` total loci follows `Binom(n = signature_size, p = p_above)`. The enrichment p-value is:

$$P(X \geq n_{obs}) = \mathrm{Binom.sf}(n_{obs} - 1,\ n = \mathrm{sig\_size},\ p)$$

The check is flagged (⚠️) when this right-tail p-value < 1%, meaning the observed multi-read count is significantly higher than expected given the measured tumor fraction.
