# noqa: N999
from enum import Enum


class HistogramColumnNames(Enum):
    # Internal names
    COUNT = "count"
    COUNT_NORM = "count_norm"
    STRAND_RATIO_START = "strand_ratio_start"
    STRAND_RATIO_END = "strand_ratio_end"
    STRAND_RATIO_CATEGORY_START = "strand_ratio_category_start"
    STRAND_RATIO_CATEGORY_END = "strand_ratio_category_end"
    STRAND_RATIO_CATEGORY_END_NO_UNREACHED = "strand_ratio_category_end_no_unreached"
    STRAND_RATIO_CATEGORY_CONSENSUS = "strand_ratio_category_consensus"
    LOOP_SEQUENCE_START = "loop_sequence_start"
    LOOP_SEQUENCE_END = "loop_sequence_end"


# Display defaults
STRAND_RATIO_AXIS_LABEL = "MINUS strand ratio"
