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


class FeatureMapFilters(Enum):
    LOW_QUAL = "LowQual"
    SINGLE_READ = "SingleRead"
    PASS = "PASS"  # noqa: S105
    PRE_FILTERED = "PreFiltered"
