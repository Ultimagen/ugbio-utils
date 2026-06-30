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

Up to six QC checks are displayed above the Assay Metrics in the report. They do not force an Indeterminate call — they are informational flags.

| Check | Threshold | Rationale |
|---|---|---|
| Signature size | ≥ 500 loci | Too few loci reduce statistical power |
| Mean coverage | ≥ 15× | Low coverage inflates noise rate variance |
| Synthetic controls | ≥ 20 | Fewer controls make null distribution unreliable |
| Expected multi-read support distribution (matched) | 0 outlier loci (Bonferroni-corrected p ≥ 1%) | See below |
| Expected multi-read support distribution (synthetic controls) | 0 outlier loci (Bonferroni-corrected p ≥ 1%) | See below — only shown when synthetic controls are present |
| Expected multi-read support distribution (cohort controls) | 0 outlier loci (Bonferroni-corrected p ≥ 1%) | See below — only shown when cohort controls are present |

### Expected Multi-Read Support Distribution

These checks detect loci with a significantly higher read count than expected under the respective Poisson model. A flagged check may indicate germline variants (matched), contamination, or somatic variants leaking into control signatures.

**Per-locus Poisson test:**

For each locus with `k` observed supporting reads, the right-tail p-value is:

$$p_i = P(X \geq k_i \mid \mathrm{Poisson}(\lambda))$$

The expected rate λ differs by check:

| Check | λ per locus |
|---|---|
| Matched signature | `mean_coverage × matched_vaf` (measured tumor fraction) |
| Synthetic controls | `mean_coverage × p_err` (background noise rate) |
| Cohort controls | `mean_coverage × p_err` (background noise rate) |

**Bonferroni correction for outliers:**

A locus is declared an outlier when its p-value falls below the Bonferroni-corrected threshold:

$$p_i < \frac{\alpha}{N}$$

where α = 1% and $N$ is the number of unique loci being tested.  For the matched signature, $N$ is the matched signature size.  For control signatures, each signature is tested independently using its own locus count as $N$; a locus is declared an outlier if the test fires for any single signature.  This per-signature correction is necessary because cohort control signatures vary widely in size — a shared $N$ derived from the matched signature would under-correct large cohort signatures.  The check is flagged (⚠️) when at least one outlier locus is found.
