from enum import Enum

SPAN_DEL = "*"
IGNORE = -1
MISS = -2


class GtType(Enum):
    APPROXIMATE = "approximate"
    EXACT = "exact"

    def __str__(self):
        return self.value

    @property
    def is_cnv_depth(self) -> bool:
        return self in (VcfType.CNV_DEPTH, VcfType.CNV_DEPTH_CNMOPS)


class VcfType(Enum):
    SINGLE_SAMPLE = "single_sample"
    DEEP_VARIANT = "deep_variant"
    DEEP_VARIANT_WITH_SOFTCLIP_COUNTS = "deep_variant_extended"
    JOINT = "joint_callset"
    CNV_DEPTH = "cnv_depth"
    CNV_DEPTH_CNMOPS = "cnv_depth_cnmops"  # depth CNV calls with cn.mops candidates (source_cnmops feature)

    def __str__(self):
        return self.value

    @property
    def is_cnv_depth(self) -> bool:
        return self in (VcfType.CNV_DEPTH, VcfType.CNV_DEPTH_CNMOPS)
