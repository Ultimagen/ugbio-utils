"""Constants for CNV module including INFO and FILTER tag registries."""

# NOTE: both the id and the column name should appear in the registry
# (after converting table to BED the names are as in the VCF)
INFO_TAG_REGISTRY: dict[str, tuple[str, int | str, str, str, str]] = {
    "CNV_calls_source": (
        "CNV_SOURCE",
        1,
        "String",
        "the tool called this CNV. can be combination of: cn.mops, cnvpytor, gridss",
        "INFO",
    ),
    "CNV_SOURCE": (
        "CNV_SOURCE",
        ".",
        "String",
        "the tool called this CNV. can be combination of: cn.mops, cnvpytor, gridss",
        "INFO",
    ),
    "CNMOPS_SAMPLE_STDEV": (
        "CNMOPS_SAMPLE_STDEV",
        1,
        "Float",
        "Standard deviation of coverage in the CNV region for the sample (cn.mops)",
        "INFO",
    ),
    "CNMOPS_COHORT_MEAN": (
        "CNMOPS_COHORT_MEAN",
        1,
        "Float",
        "Mean coverage in the CNV region across the cohort (cn.mops)",
        "INFO",
    ),
    "CNMOPS_COHORT_STDEV": (
        "CNMOPS_COHORT_STDEV",
        1,
        "Float",
        "Standard deviation of coverage in the CNV region across the cohort (cn.mops)",
        "INFO",
    ),
    "GAP_PERC": ("GAP_PERC", 1, "Float", "Fraction of N bases in the CNV region from reference genome", "INFO"),
}

# the reason filters require special treatment is that they need to be
# unique and should be PASS if none present. In the end filter tags are added to info
# All columns from FILTER_COLUMNS_REGISTRY are aggregated into a single INFO field FILTER_ANNOTATION_NAME
FILTER_COLUMNS_REGISTRY = ["LCR_label_value"]
INFO_TAG_REGISTRY["REGION_ANNOTATIONS"] = (
    "REGION_ANNOTATIONS",
    ".",
    "String",
    "Aggregated region-based annotations for the CNV (e.g., LCR status and other region filters)",
    "INFO",
)

FILTER_TAG_REGISTRY = {
    "Clusters": ("Clusters", None, None, "Overlaps with locations with frequent clusters of CNV", "FILTER"),
    "Coverage-Mappability": (
        "Coverage-Mappability",
        None,
        None,
        "Overlaps with low coverage or low mappability regions",
        "FILTER",
    ),
    "Telomere_Centromere": (
        "Telomere_Centromere",
        None,
        None,
        "Overlaps with telomere or centromere regions",
        "FILTER",
    ),
    "LEN": (
        "LEN",
        None,
        None,
        "CNV length is below the minimum length threshold (cn.mops)",
        "FILTER",
    ),
    "UG-CNV-LCR": (
        "UG-CNV-LCR",
        None,
        None,
        "Overlaps with low-complexity regions as defined by UGBio CNV module",
        "FILTER",
    ),
    "CNMOPS_SHORT_DUPLICATION": (
        "CNMOPS_SHORT_DUPLICATION",
        None,
        None,
        "Duplication length is shorter than the defined threshold in cn.mops calls.",
        "FILTER",
    ),
}
