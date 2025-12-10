import warnings

import pandas as pd
import pysam

warnings.filterwarnings("ignore")

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
FILTER_ANNOTATION_NAME = "REGION_ANNOTATIONS"
INFO_TAG_REGISTRY[FILTER_ANNOTATION_NAME] = (
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


def add_vcf_header(sample_name: str, fasta_index_file: str) -> pysam.VariantHeader:
    """
    Create a VCF header for CNV calls with the given sample name and reference genome information.

    Parameters
    ----------
    sample_name : str
        The name of the sample.
    fasta_index_file : str
        Path to the reference genome index file (.fai).

    Returns
    -------
    pysam.VariantHeader
        The VCF header with the specified sample name and reference genome information.
    """
    header = pysam.VariantHeader()

    # Add meta-information to the header
    header.add_meta("fileformat", value="VCFv4.2")
    header.add_meta("source", value="ULTIMA_CNV")

    # Add sample names to the header
    header.add_sample(sample_name)

    # Add custom meta information
    header.add_meta("VCF_TYPE", value="ULTIMA_CNV")

    # Add contigs info to the header
    df_genome = pd.read_csv(fasta_index_file, sep="\t", header=None, usecols=[0, 1])
    df_genome.columns = ["chr", "length"]
    for _, row in df_genome.iterrows():
        chr_id = row["chr"]
        length = row["length"]
        header.contigs.add(chr_id, length=length)

    # Add ALT (using add_line as ALT records don't have a direct add method)
    header.add_line('##ALT=<ID=CNV,Description="Copy number variant region">')
    header.add_line('##ALT=<ID=DEL,Description="Deletion relative to the reference">')
    header.add_line('##ALT=<ID=DUP,Description="Region of elevated copy number relative to the reference">')

    # Add FILTER (PASS is automatically included by pysam)
    for filter_tag in FILTER_TAG_REGISTRY.values():
        if filter_tag[-1] == "FILTER":
            header.filters.add(filter_tag[0], None, None, filter_tag[-2])

    # Add INFO
    header.info.add("CopyNumber", 1, "Float", "average copy number detected from cn.mops")
    header.info.add("RoundedCopyNumber", 1, "Integer", "rounded copy number detected from cn.mops")
    header.info.add("SVLEN", ".", "Integer", "CNV length")
    header.info.add("SVTYPE", 1, "String", "CNV type. can be DUP or DEL")

    # Add INFO tags from registry, avoiding duplicates
    added_info_ids = set()
    for info_tag in INFO_TAG_REGISTRY.values():
        if info_tag[-1] == "INFO" and info_tag[0] not in added_info_ids:
            header.info.add(info_tag[0], info_tag[1], info_tag[2], info_tag[3])
            added_info_ids.add(info_tag[0])

    # Add FORMAT
    header.formats.add("GT", 1, "String", "Genotype")

    return header


def process_filter_columns(
    row: pd.Series,
    filter_columns_registry: list = FILTER_COLUMNS_REGISTRY,
    filter_tags_registry: dict = FILTER_TAG_REGISTRY,
) -> str:
    """
    Process filter columns for a single row, handling multiple filter values separated by | or ;.

    Parameters
    ----------
    row : pd.Series
        A row from the DataFrame containing filter columns.
    filter_columns_registry : list, optional
        A list of column names to process for filters. Default is FILTER_COLUMNS_REGISTRY.
    filter_tags_registry : dict, optional
        A dictionary of valid filter tags. Default is FILTER_TAG_REGISTRY.

    Returns
    -------
    str
        Comma-separated string of unique filter values, or 'PASS' if no filters apply.
    """
    filter_values = []

    for col in filter_columns_registry:
        if col in row and pd.notna(row[col]):
            value = str(row[col])
            # Handle multiple values separated by | or ;
            value = value.replace("|", ";")
            parts = [part for part in value.split(";") if part and part not in ("PASS", ".")]
            filter_values.extend(parts)

    # Remove duplicates while preserving order
    unique_filters = list(set(filter_values))
    if [x for x in unique_filters if x not in filter_tags_registry.keys()]:
        raise ValueError(
            f"Unknown filter values found: {[x for x in unique_filters if x not in filter_tags_registry.keys()]}"
        )

    unique_filters = [x for x in unique_filters if x in filter_tags_registry.keys()]

    # Return 'PASS' if no filters, otherwise return comma-separated filter list
    return "PASS" if len(unique_filters) == 0 else ",".join(sorted(unique_filters))


def _create_base_vcf_record(vcf_out: pysam.VariantFile, row: pd.Series) -> pysam.VariantRecord:
    """
    Create a base VCF record with chromosome, position, and variant type information.

    Parameters
    ----------
    vcf_out : pysam.VariantFile
        VCF file handle for creating new records.
    row : pd.Series
        DataFrame row containing CNV information.

    Returns
    -------
    pysam.VariantRecord
        Base VCF record with basic information set.
    """
    record = vcf_out.new_record()
    record.contig = str(row["chr"])
    record.start = row["start"]
    record.stop = row["end"]
    record.ref = "N"
    record.alts = (f"<{row['SVTYPE']}>",)

    # Set ID if present in the dataframe
    if "ID" in row and pd.notna(row["ID"]):
        record.id = str(row["ID"])

    return record


def _add_filters_to_record(record: pysam.VariantRecord, filter_value: str) -> None:
    """
    Add filter information to a VCF record.

    Parameters
    ----------
    record : pysam.VariantRecord
        VCF record to modify.
    filter_value : str
        Filter value from the dataframe.
    """
    if filter_value == "PASS":
        record.filter.add("PASS")
    else:
        filters = filter_value.split(",")
        for f in filters:
            record.filter.add(f)


def _get_possible_column_names(original_col: str) -> list[str]:
    """
    Get possible column names for backward compatibility.

    Parameters
    ----------
    original_col : str
        The original column name.

    Returns
    -------
    list[str]
        List of possible column names including backward compatibility aliases.
    """
    possible_cols = [original_col]
    if original_col == "CNV_SOURCE":
        possible_cols.append("CNV_calls_source")
    return possible_cols


def _add_registry_info_field(
    record: pysam.VariantRecord, row: pd.Series, vcf_tag: str, possible_cols: list[str], data_type: str
) -> None:
    """
    Add a single INFO field from the registry to the VCF record.

    Parameters
    ----------
    record : pysam.VariantRecord
        VCF record to modify.
    row : pd.Series
        DataFrame row containing CNV information.
    vcf_tag : str
        VCF INFO tag name.
    possible_cols : list[str]
        List of possible column names to check in the dataframe.
    data_type : str
        Data type for the INFO field (e.g., 'Integer', 'Float', 'String').
    """
    for col in possible_cols:
        if col in row and pd.notna(row[col]):
            if data_type == "Integer":
                record.info[vcf_tag] = int(row[col])
            elif data_type == "Float":
                record.info[vcf_tag] = float(row[col])
            else:
                record.info[vcf_tag] = str(row[col])
            break  # Only use the first available column


def _add_standard_info_fields(record: pysam.VariantRecord, row: pd.Series) -> None:
    """
    Add standard INFO fields to the VCF record.

    Parameters
    ----------
    record : pysam.VariantRecord
        VCF record to modify.
    row : pd.Series
        DataFrame row containing CNV information.
    """
    if pd.notna(row.get("CopyNumber")):
        record.info["CopyNumber"] = float(row["CopyNumber"])
    if pd.notna(row.get("RoundedCopyNumber")):
        record.info["RoundedCopyNumber"] = int(row["RoundedCopyNumber"])
    if pd.notna(row.get("SVLEN")):
        record.info["SVLEN"] = int(row["SVLEN"])
    if pd.notna(row.get("SVTYPE")):
        record.info["SVTYPE"] = str(row["SVTYPE"])


def _add_info_fields_to_record(record: pysam.VariantRecord, row: pd.Series) -> None:
    """
    Add INFO field information to a VCF record directly from dataframe row.

    Parameters
    ----------
    record : pysam.VariantRecord
        VCF record to modify.
    row : pd.Series
        DataFrame row containing CNV information.
    """
    # Add INFO fields from registry first, handling both possible column names
    for original_col, (vcf_tag, _, data_type, _, _) in INFO_TAG_REGISTRY.items():
        possible_cols = _get_possible_column_names(original_col)
        _add_registry_info_field(record, row, vcf_tag, possible_cols, data_type)

    # Add standard INFO fields after registry fields
    _add_standard_info_fields(record, row)


def _determine_genotype(copy_number_value: str | None) -> tuple[int | None, int]:
    """
    Determine the genotype based on copy number value.

    Parameters
    ----------
    copy_number_value : str | None
        String representation of copy number.

    Returns
    -------
    tuple[int | None, int]
        Genotype tuple for VCF format.
    """
    gt = [None, 1]
    if copy_number_value is not None:
        try:
            cn_float = float(copy_number_value)
            rounded_cn = int(round(cn_float))
            if rounded_cn == 1:
                gt = [0, 1]
            elif rounded_cn == 0:
                gt = [1, 1]
        except (ValueError, TypeError):
            pass  # Keep default gt
    return (gt[0], gt[1])


def _add_genotype_to_record(record: pysam.VariantRecord, sample_name: str, row: pd.Series) -> None:
    """
    Add genotype information to a VCF record.

    Parameters
    ----------
    record : pysam.VariantRecord
        VCF record to modify.
    sample_name : str
        Name of the sample.
    row : pd.Series
        DataFrame row containing copy number information.
    """
    copy_number_value = row.get("CopyNumber", None)
    if pd.notna(copy_number_value):
        copy_number_value = str(copy_number_value)
    else:
        copy_number_value = None
    gt = _determine_genotype(copy_number_value)
    record.samples[sample_name]["GT"] = gt


def write_cnv_vcf(outfile: str, cnv_df: pd.DataFrame, sample_name: str, fasta_index_file: str) -> None:
    """
    Write CNV calls directly from dataframe to a VCF file.

    Parameters
    ----------
    outfile : str
        Path to the output VCF file.
    cnv_df : pd.DataFrame
        Dataframe containing processed CNV data.
    sample_name : str
        The name of the sample.
    fasta_index_file : str
        Path to the reference genome index file (.fai).
    """
    header = add_vcf_header(sample_name, fasta_index_file)

    with pysam.VariantFile(outfile, mode="w", header=header) as vcf_out:
        for _, row in cnv_df.iterrows():
            # Create base VCF record with basic information
            record = _create_base_vcf_record(vcf_out, row)

            # Add filter information to the record
            _add_filters_to_record(record, row["filter"])

            # Add INFO field information to the record
            _add_info_fields_to_record(record, row)

            # Add genotype information to the record
            _add_genotype_to_record(record, sample_name, row)

            # Write the completed record to the VCF file
            vcf_out.write(record)
