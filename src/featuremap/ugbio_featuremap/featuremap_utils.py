from enum import Enum

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
    ST = "st"
    ET = "et"
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


class FeatureMapFilters(Enum):
    LOW_QUAL = "LowQual"
    SINGLE_READ = "SingleRead"
    PASS = "PASS"  # noqa: S105
    PRE_FILTERED = "PreFiltered"
