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
    MQUAL = "MQUAL"
    SNVQ = "SNVQ"
    X_HMER_REF = "X_HMER_REF"
    X_HMER_ALT = "X_HMER_ALT"
    ST = "st"
    ET = "et"
    EDIST = "EDIST"
    HAMDIST = "HAMDIST"
    HAMDIST_FILT = "HAMDIST_FILT"


class FeatureMapFilters(Enum):
    LOW_QUAL = "LowQual"
    SINGLE_READ = "SingleRead"
    PASS = "PASS"  # noqa: S105
    PRE_FILTERED = "PreFiltered"
