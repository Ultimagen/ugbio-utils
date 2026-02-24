# XGBoost vs DNN Feature Comparison

## Summary

**Coverage: 25/26** XGBoost features are present in the DNN (explicitly or implicitly).
Only **DUP** (duplicate flag) is truly absent — duplicates are typically removed upstream so it carries little signal.

## Feature Mapping Table

| # | XGBoost Feature | In DNN? | DNN Equivalent | Notes |
|---|-----------------|---------|----------------|-------|
| 1 | **REF** | Yes | ref_base embedding (all positions) | DNN has richer per-position representation |
| 2 | **ALT** | Yes | read_base embedding at focus position | Focus channel marks the variant site |
| 3 | **X_HMER_REF** | Yes (implicit) | ref_base sequence + convolutions | Network can learn to count runs |
| 4 | **X_HMER_ALT** | Yes (implicit) | read_base sequence + convolutions | Network can learn to count runs |
| 5 | **X_PREV1** | Yes | ref_base at focus-1 | Full sequence context available |
| 6 | **X_NEXT1** | Yes | ref_base at focus+1 | Full sequence context available |
| 7 | **X_PREV2** | Yes | ref_base at focus-2 | Full sequence context available |
| 8 | **X_NEXT2** | Yes | ref_base at focus+2 | Full sequence context available |
| 9 | **X_PREV3** | Yes | ref_base at focus-3 | Full sequence context available |
| 10 | **X_NEXT3** | Yes | ref_base at focus+3 | Full sequence context available |
| 11 | **BCSQ** | Yes (implicit) | ref_base + read_base at focus + codon context | Network sees base change in sequence context |
| 12 | **BCSQCSS** | Yes (implicit) | ref_base + read_base at focus + codon context | Severity derivable from consequence |
| 13 | **RL** | Yes (implicit) | sum(mask) | Read length = number of valid positions |
| 14 | **INDEX** | Yes | focus channel | 1 at variant position, 0 elsewhere |
| 15 | **DUP** | No | — | BAM flag, not in alignment data |
| 16 | **REV** | Yes | strand channel | Constant channel (0/1) |
| 17 | **SCST** | Yes (implicit) | softclip_mask channel | Per-position soft-clip indicator |
| 18 | **SCED** | Yes (implicit) | softclip_mask channel | Per-position soft-clip indicator |
| 19 | **MAPQ** | Yes | mapq channel | Constant channel (normalized /60) |
| 20 | **EDIST** | Yes (implicit) | read_base vs ref_base mismatches | Network can count differences |
| 21 | **SMQ_BEFORE** | Yes (implicit) | qual channel before focus | Per-position base quality available |
| 22 | **SMQ_AFTER** | Yes (implicit) | qual channel after focus | Per-position base quality available |
| 23 | **tm** | Yes | tm channel | Constant channel (encoded as int) |
| 24 | **rq** | Yes | rq channel | Constant channel |
| 25 | **st** | Yes | st channel | Constant channel (encoded as int) |
| 26 | **et** | Yes | et channel | Constant channel (encoded as int) |

## Legend

- **Yes** — Feature is explicitly present in the DNN input
- **Yes (implicit)** — Feature can be inferred by the network from its input channels (e.g., homopolymer length from base sequences via convolutions)
- **No** — Feature is not available to the DNN

## Key Observations

1. **Homopolymer features (X_HMER_REF/ALT)**: The DNN sees full ref/read base sequences, so convolutions can learn to count homopolymer runs. However, this requires the network to learn counting, which is non-trivial — explicit features give XGBoost an advantage here.

2. **Sequence context (X_PREV/NEXT)**: The DNN has *richer* context than XGBoost — it sees up to 300 positions vs. XGBoost's ±3 bases.

3. **EDIST**: The DNN can in principle count mismatches across the full read alignment (read_base vs ref_base), though learning exact counting is harder than having the explicit value.

4. **BCSQ/BCSQCSS**: The DNN sees the codon context (ref + read bases around the variant), so it could learn reading frame and infer consequence type. However, it lacks knowledge of gene boundaries and strand.

5. **DUP**: The only truly absent feature. Duplicates are typically removed upstream, so this is unlikely to be important.
