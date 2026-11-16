"""Constants for CNV module including INFO and FILTER tag registries."""

# NOTE: both the id and the column name should appear in the registry
# (after converting table to BED the names are as in the VCF)
INFO_TAG_REGISTRY: dict[str, tuple[str, int | str, str, str, str]] = {
    "CNV_calls_source": (
        "CNV_SOURCE",
        ".",
        "String",
        "the tool that called this CNV. can be combination of: depth, split_read, pair, cn.mops, gridss",
        "INFO",
    ),
    "CNV_SOURCE": (
        "CNV_SOURCE",
        ".",
        "String",
        "the tool that called this CNV. can be combination of: depth, split_read, pair, cn.mops, gridss",
        "INFO",
    ),
    "CNV_ID": (
        "CNV_ID",
        ".",
        "String",
        "IDs of CNV calls that were replaced by this SV call during merge",
        "INFO",
    ),
    "CNMOPS_SAMPLE_MEAN": (
        "CNMOPS_SAMPLE_MEAN",
        1,
        "Float",
        "Mean coverage in the CNV region for the sample (cn.mops)",
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
    "GAP_PERCENTAGE": (
        "GAP_PERCENTAGE",
        1,
        "Float",
        "Fraction of N bases in the CNV region from reference genome",
        "INFO",
    ),
    "CIPOS": (
        "CIPOS",
        2,
        "Integer",
        "Confidence interval around POS for imprecise variants",
        "INFO",
    ),
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
}

# INFO tags written by the depth CNV caller (depth_cnv_caller, split_read_candidates, breakpoint_refinement).
# Kept apart from INFO_TAG_REGISTRY, whose tags are added to every CNV VCF header (cn.mops, FREEC).
DEPTH_CNV_INFO_TAGS: dict[str, tuple[str, int | str, str, str, str]] = {
    "END": ("END", 1, "Integer", "End position of the variant", "INFO"),
    "CIEND": ("CIEND", 2, "Integer", "Confidence interval around END for imprecise variants", "INFO"),
    "CN_MEAN": ("CN_MEAN", 1, "Float", "Mean copy-number estimate over the call (depth caller)", "INFO"),
    "BASE_CN": (
        "BASE_CN",
        1,
        "Integer",
        "Expected copy number of the contig for this sample (2, or 1 when haploid)",
        "INFO",
    ),
    "Z": ("Z", 1, "Float", "Strongest segment z-score of the copy-number change vs the baseline", "INFO"),
    "LOW_CONF": (
        "LOW_CONF",
        0,
        "Flag",
        "Mostly segdup / low-mappability sequence without a full copy change (a filter feature, not a FILTER)",
        "INFO",
    ),
    "N_MERGED": ("N_MERGED", 1, "Integer", "Number of depth segments merged into this call", "INFO"),
    "SEGDUP_FRAC": ("SEGDUP_FRAC", 1, "Float", "Mean segmental-duplication fraction over the call", "INFO"),
    "UNIQ_FRAC": ("UNIQ_FRAC", 1, "Float", "Mean fraction of depth from reads with MAPQ>=20", "INFO"),
    "CLUSTER_SUPPORT": (
        "CLUSTER_SUPPORT",
        1,
        "Integer",
        "Split reads or discordant pairs in the cluster that created a split_read or pair candidate",
        "INFO",
    ),
    "BREAKPOINT_SOURCE": (
        "BREAKPOINT_SOURCE",
        1,
        "String",
        "Where the breakpoints come from: split_read (junction), pair (deletion shrunk by discordant pairs) or depth",
        "INFO",
    ),
    "BREAKPOINT_SUPPORT": (
        "BREAKPOINT_SUPPORT",
        1,
        "Integer",
        "Split reads or discordant pairs supporting the breakpoints (0 for depth breakpoints)",
        "INFO",
    ),
    "DEPTH_START": ("DEPTH_START", 1, "Integer", "Start of the call from depth, before breakpoint refinement", "INFO"),
    "DEPTH_END": ("DEPTH_END", 1, "Integer", "End of the call from depth, before breakpoint refinement", "INFO"),
}
# CNV_SOURCE values of the depth CNV caller
CNV_SOURCE_DEPTH = "depth"
CNV_SOURCE_SPLIT_READ = "split_read"
CNV_SOURCE_PAIR = "pair"
CNV_SOURCE_CNMOPS = "cn.mops"  # optional cohort caller (cnmops_cnv_candidates)
