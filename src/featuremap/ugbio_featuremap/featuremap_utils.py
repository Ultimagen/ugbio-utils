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
    CHROM = CHROM
    POS = POS
    REF = REF
    ALT = ALT
    QUAL = QUAL
    FILTER = FILTER
    X_ALT = "X_ALT"


class FeatureMapFilters(Enum):
    LOW_QUAL = "LowQual"
    SINGLE_READ = "SingleRead"
    PASS = "PASS"  # noqa: S105
    PRE_FILTERED = "PreFiltered"
