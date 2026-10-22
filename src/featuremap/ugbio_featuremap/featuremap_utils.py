from enum import Enum

import numpy as np
import polars as pl
from ugbio_core.consts import (
    ALT,
    CHROM,
    FILTER,
    POS,
    QUAL,
    REF,
)


class FeatureMapFields(Enum):
    CHROM = CHROM.upper()
    POS = POS.upper()
    REF = REF.upper()
    ALT = ALT.upper()
    QUAL = QUAL.upper()
    FILTER = FILTER.upper()
    ID = "ID"
    XGB_PROBA = "XGB_PROBA"
    SAMPLE = "SAMPLE"
    X_ALT = "X_ALT"
    X_PREV1 = "X_PREV1"
    X_PREV2 = "X_PREV2"
    X_PREV3 = "X_PREV3"
    X_NEXT1 = "X_NEXT1"
    X_NEXT2 = "X_NEXT2"
    X_NEXT3 = "X_NEXT3"
    MQUAL = "MQUAL"
    SNVQ = "SNVQ"
    X_HMER_REF = "X_HMER_REF"
    X_HMER_ALT = "X_HMER_ALT"
    X_INDEL_CLASS = "X_INDEL_CLASS"
    X_INDEL_SEQ = "X_INDEL_SEQ"
    ST = "st"
    ET = "et"
    FS = "fs"  # forward-strand read count (consensus data)
    RS = "rs"  # reverse-strand read count (consensus data)
    EDIST = "EDIST"
    HAMDIST = "HAMDIST"
    HAMDIST_FILT = "HAMDIST_FILT"
    INDEX = "INDEX"  # position in read
    RL = "RL"  # read length
    DP = "DP"  # depth
    BCSQ = "BCSQ"  # base calling quality
    REV = "REV"  # is reverse strand
    FILT = "FILT"  # is reads in this position pass filters
    DUP = "DUP"  # is duplicate read
    ADJ_REF_DIFF = "ADJ_REF_DIFF"  # difference to reference base
    DP_FILT = "DP_FILT"  # depth of reads that pass filters
    RADJ_REF_D = "RADJ_REF_D"  # mean ADJ_REF_DIFF of reads that pass filters
    RAW_VAF = "RAW_VAF"  # raw variant allele frequency
    VAF = "VAF"  # variant allele frequency after filtering
    DP_MAPQ60 = "DP_MAPQ60"  # depth of reads with mapq>=60
    MAPQ = "MAPQ"  # mapping quality of reads supporting the variant
    SCST = "SCST"  # soft clip start
    SCED = "SCED"  # soft clip end
    AD = "AD"  # allelic depths
    GT = "GT"


class FeatureMapFilters(Enum):
    LOW_QUAL = "LowQual"
    SINGLE_READ = "SingleRead"
    PASS = "PASS"  # noqa: S105
    PRE_FILTERED = "PreFiltered"


# ---------------------------------------------------------------------------
# Recalibration cells: variant class x homopolymer run-length bin
# ---------------------------------------------------------------------------
# SNVQ recalibration builds one MQUAL->SNVQ table per "cell". SNV and non-hmer indels are one cell each; hmer
# indels are split by the length of the reference homopolymer run the indel changes (X_HMER_RUN — NOT X_HMER_REF,
# which is measured at the VCF anchor base). The same cell key is used by data prep (raw counts, sampling weights),
# recalibration (tables + prefactors) and inference (table lookup), so it lives here.
CELL_SNV = "snv"
CELL_HMER_INDEL = "hmer_indel"
CELL_NON_HMER_INDEL = "non_hmer_indel"
CELL_SEP = "|"
# (label, min_run, max_run) inclusive; run 0 (an insertion creating a new run) is folded into L1.
HMER_RUN_BINS = (("L1", 0, 1), ("L2", 2, 2), ("L3-4", 3, 4), ("L5-6", 5, 6), ("L7-9", 7, 9), ("L10+", 10, None))


def hmer_bin_cell(bin_label: str) -> str:
    """Cell key of an hmer run-length bin, e.g. ``hmer_indel|L3-4``."""
    return f"{CELL_HMER_INDEL}{CELL_SEP}{bin_label}"


def hmer_run_bin_label(run_length: int) -> str:
    """Bin label for a homopolymer run length (``X_HMER_RUN``)."""
    for label, lo, hi in HMER_RUN_BINS:
        if run_length >= lo and (hi is None or run_length <= hi):
            return label
    return HMER_RUN_BINS[0][0]  # negative/invalid -> shortest bin


def hmer_run_bin_array(run_length: np.ndarray) -> np.ndarray:
    """Vectorized :func:`hmer_run_bin_label` (missing / negative -> shortest bin)."""
    run = np.nan_to_num(np.asarray(run_length, dtype=float), nan=0.0)
    out = np.full(run.shape, HMER_RUN_BINS[0][0], dtype=object)
    for label, lo, hi in HMER_RUN_BINS[1:]:
        out[(run >= lo) & ((run <= hi) if hi is not None else True)] = label
    return out


def cell_key_expr(columns: list[str] | set[str], *, with_bins: bool = True) -> pl.Expr:
    """Polars expression giving each featuremap row its recalibration cell key.

    Class predicate (mirrors srsnv_report.compute_variant_type_column): indel iff ``X_IC`` in {ins, del};
    non-hmer iff ``X_INDEL_CLASS == non_hmer``; other indels are hmer (missing X_INDEL_CLASS -> hmer); else SNV.
    With ``with_bins`` and an ``X_HMER_RUN`` column, hmer indels get ``hmer_indel|<bin>``.
    """
    columns = set(columns)
    if "X_IC" not in columns:
        return pl.lit(CELL_SNV)
    is_indel = pl.col("X_IC").cast(pl.Utf8).str.to_lowercase().is_in(["ins", "del"]).fill_null(value=False)
    if "X_INDEL_CLASS" in columns:
        is_nh = is_indel & (pl.col("X_INDEL_CLASS").cast(pl.Utf8).str.to_lowercase() == "non_hmer").fill_null(
            value=False
        )
    else:
        is_nh = pl.lit(value=False)
    hmer = pl.lit(CELL_HMER_INDEL)
    if with_bins and "X_HMER_RUN" in columns:
        run = pl.col("X_HMER_RUN").cast(pl.Int64, strict=False).fill_null(0)
        expr = None
        for label, lo, hi in reversed(HMER_RUN_BINS):
            cond = (run >= lo) & (run <= hi) if hi is not None else run >= lo
            expr = (
                (pl.when(cond).then(pl.lit(hmer_bin_cell(label))))
                if expr is None
                else expr.when(cond).then(pl.lit(hmer_bin_cell(label)))
            )
        hmer = expr.otherwise(pl.lit(hmer_bin_cell(HMER_RUN_BINS[0][0])))
    return pl.when(is_nh).then(pl.lit(CELL_NON_HMER_INDEL)).when(is_indel).then(hmer).otherwise(pl.lit(CELL_SNV))
