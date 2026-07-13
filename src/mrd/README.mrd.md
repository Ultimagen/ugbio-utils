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

2. **LOD** at configurable target recall (default 95%) and FPR (default 5%): the smallest TF such that a true positive sample crosses the threshold at the target recall rate:

$$\mathrm{LOD} = \min\{\mathrm{TF} : P(X \geq n_{th} \mid \mathrm{Binom}(N,\, p_{err} + \mathrm{TF})) \geq 0.95\}$$

where $N = \text{signature\_size} \times \text{mean\_coverage} \times \text{denom\_ratio}$.

The LOD decreases (improves) with larger signature size, higher coverage, or lower noise rate. It is `None` when no threshold satisfies the FPR constraint (e.g., signature too small or coverage too low).

## Locus Filters

Two optional pre-detection filters remove noisy loci before the Binomial test is run.  Both default to enabled (CLI defaults); pass `None` programmatically to disable.

### Noisy-Loci Filter (`--thresh-noise-lq-reads`)

Removes loci where the number of low-quality reads (failing `read_filter_query`) meets or exceeds the threshold.  Targets loci systematically affected by assay noise rather than true ctDNA signal.

### Multi-Read Locus Filter (`--thresh-multi-read-pvalue`)

Removes loci whose per-locus read count is a significant outlier under a Poisson null model.  The primary target is germline or mosaic variants leaking into the matched signature, but the test is applied identically to every `(signature_type, signature)` pair.

**λ estimation:**  To avoid circular bias (germline outliers would inflate λ and mask themselves), λ is estimated only from *background* loci — those with ≤ `_VAF_ESTIMATE_READ_CAP` (= 6) reads:

$$\lambda = \frac{\text{reads at background loci}}{\text{corrected\_coverage}} \times \text{mean\_coverage}$$

A Jeffreys prior (`0.5 / (N+1)`) is applied when no background reads are observed.

**Bonferroni N — per signature:**  Each `(signature_type, signature)` pair is tested independently using **its own locus count** as the family size N.  This is the same logic used by the QC check (see below):

| Signature type | Bonferroni N |
|---|---|
| Matched | matched signature's own locus count |
| Synthetic control (db\_control) | that replicate's own locus count |
| Cohort control | that patient's own signature locus count |

Using the matched `signature_size` as a shared N would under-correct large signatures and over-correct small ones.

A locus is flagged when:

$$p_i \times N < \text{thresh\_multi\_read\_pvalue} \quad \text{and} \quad k_i \geq 2$$

The minimum-reads guard (`k ≥ 2`) prevents single-read loci from being removed; a single read is indistinguishable from background noise regardless of how small λ is.  Flagged loci are removed from **all reads of that signature type**.

## QC Checks

Up to six QC checks are displayed above the Assay Metrics in the report.  They do not force an Indeterminate call — they are informational flags.

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

| Check | λ per locus | Bonferroni N |
|---|---|---|
| Matched signature | `mean_coverage × matched_vaf` (measured tumor fraction) | matched signature size |
| Synthetic controls (db\_control) | `mean_coverage × p_err` (background noise rate) | each synthetic signature's own locus count |
| Cohort controls | `mean_coverage × p_err` (background noise rate) | each cohort signature's own locus count |

**Bonferroni correction for outliers:**

A locus is declared an outlier when its p-value falls below the Bonferroni-corrected threshold:

$$p_i < \frac{\alpha}{N}$$

where α = 1%.  The family size $N$ differs by group:

- **Matched signature**: $N$ = matched signature size (number of filtered loci passing the signature filter).
- **Synthetic controls** (db\_control): each synthetic replicate is tested independently; $N$ = that replicate's own locus count.  A locus is declared an outlier if the test fires for any single replicate.  Synthetic controls are population-panel signatures drawn from unrelated samples, so their locus counts can differ substantially from the matched signature and from each other.
- **Cohort controls**: same per-signature treatment; $N$ = each cohort patient's own signature size.  Cohort controls are other patients' matched signatures evaluated on *this* patient's plasma, so their sizes can vary widely.

Using the matched `signature_size` as a shared $N$ for either control type would be incorrect: under-correcting large signatures and over-correcting small ones.  The check is flagged (⚠️) when at least one outlier locus is found across any signature of that type.
