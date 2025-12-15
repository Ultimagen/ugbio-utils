"""Utilities for processing and writing CNV calls from cn.mops and other CNV callers."""

import os
import warnings

import pandas as pd
import pysam
from ugbio_cnv.cnv_vcf_consts import (
    FILTER_COLUMNS_REGISTRY,
    FILTER_TAG_REGISTRY,
    INFO_TAG_REGISTRY,
)
from ugbio_core import bed_utils

warnings.filterwarnings("ignore")


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


def merge_filter_files(original_bed: str, filter_files: list[str], output_file: str) -> None:
    """Merges multiple filter bed files into a single sorted bed file.

    Parameters
    ----------
    original_bed : str
        Original bed: four columns (chr, start, end, copy-number)
    filter_files : list[str]
        BED files with filters added to merge. Each file should have four columns (chr, start, end, CN2|LEN etc.)
        Some lines are empty
    output_file : str
        Output file: filters will be combined by ;, if all filters are empty, the result will be just CN
    """
    original_bed_df = pd.read_csv(original_bed, sep="\t", header=None, names=["chr", "start", "end", "copy-number"])
    original_bed_df = original_bed_df.set_index(["chr", "start", "end"])
    filter_dfs = []
    for filter_file in filter_files:
        filter_df = pd.read_csv(filter_file, sep="\t", header=None, names=["chr", "start", "end", "filter"])
        filter_df = filter_df.set_index(["chr", "start", "end"])
        filter_dfs.append(filter_df)
    merged_df = pd.concat((original_bed_df, *filter_dfs), axis=1, join="outer").fillna("")
    cols = ["copy-number"] + [f"filter_{i}" for i in range(len(filter_dfs))]
    merged_df.columns = cols
    merged_df["combine_filters"] = merged_df[cols[1:]].apply(lambda x: ";".join([y for y in x if y]), axis=1)
    merged_df["combined_cn"] = merged_df.apply(
        lambda x: x["copy-number"] if x["combine_filters"] == "" else x["combine_filters"], axis=1
    )
    merged_df = merged_df.reset_index()
    merged_df.to_csv(output_file, sep="\t", header=False, index=False, columns=["chr", "start", "end", "combined_cn"])


def aggregate_annotations_in_df(
    primary_bed_file: str, coverage_annotations: list[tuple[str, str, str]]
) -> pd.DataFrame:
    """
    Aggregate multiple annotation bed files into a single DataFrame.

    This function reads a primary bed file and merges coverage annotations from multiple
    bed files into a single DataFrame. The primary bed file's 4th column contains
    semicolon-separated annotations, each of which has a copy number and optional filters.
    The copy number is extracted as an integer, and filters are stored as a tuple.

    Parameters
    ----------
    primary_bed_file : str
        Path to the primary bed file with 4 columns (chr, start, end, annotation).
        The 4th column contains annotations like "CN2", "CN3|UG-CNV-LCR", or
        "CN1|LEN;CN1|UG-CNV-LCR" (semicolon-separated, CN is always the same).
    coverage_annotations : list of tuple
        List of tuples, each containing (sample_name, operation, bed_file_path).
        Example: [('cov', 'mean', 'file1.bed'), ('cov', 'std', 'file2.bed')]

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: chr, start, end, CopyNumber (int), FILTER (tuple),
        and additional columns for each coverage annotation (e.g., CNMOPS_SAMPLE_MEAN,
        CNMOPS_SAMPLE_STDEV, CNMOPS_COHORT_MEAN, CNMOPS_COHORT_STD).

    Notes
    -----
    - All bed files are assumed to have the same regions in the same order
    - The function performs sorting to ensure proper alignment
    - Coverage annotation column names are formatted as CNMOPS_{SAMPLE}_{OPERATION}
      in uppercase (e.g., CNMOPS_SAMPLE_MEAN)
    - CopyNumber is converted to an integer by removing the "CN" prefix
    - filter is a tuple of filter names (("PASS",) tuple when no filters present)
    """
    # Read the primary bed file (no header, 4 columns)
    cnv_df = pd.read_csv(primary_bed_file, sep="\t", header=None, names=["chr", "start", "end", "annotation"])

    # Sort the DataFrame to ensure consistent ordering
    cnv_df = cnv_df.sort_values(["chr", "start", "end"]).reset_index(drop=True)

    # Parse the annotation column
    # Format: "CN2", "CN3|UG-CNV-LCR", or "CN1|LEN;CN1|UG-CNV-LCR"
    def parse_annotation(annotation_str):
        # Split by semicolon to get all annotations
        parts = annotation_str.split(";")

        # Extract copy number from first part (remove "CN" prefix)
        first_part = parts[0].split("|")
        copy_number = float(first_part[0].replace("CN", ""))

        # Collect all filters from all parts
        filters = []
        for part in parts:
            part_items = part.split("|")
            # Skip the CN part (first item) and add any filters
            filters.extend(part_items[1:])

        # Return copy number and tuple of filters (PASS tuple if no filters)
        return copy_number, ",".join(filters) if filters else "PASS"

    neutral = 2
    cnv_df[["CopyNumber", "filter"]] = cnv_df["annotation"].apply(lambda x: pd.Series(parse_annotation(x)))
    cnv_df["SVTYPE"] = cnv_df["CopyNumber"].apply(
        lambda x: "DUP" if x > neutral else ("DEL" if x < neutral else "NEUTRAL")
    )
    cnv_df = cnv_df.drop(columns=["annotation"])

    # Process each coverage annotation file
    for sample_name, operation, bed_file_path in coverage_annotations:
        # Read the coverage annotation bed file
        cov_df = pd.read_csv(bed_file_path, sep="\t", header=None)

        # Assign column names for sorting
        cov_df.columns = ["chrom", "start", "end", "filter-name", "cov"]

        # Sort to match the primary bed file (first 3 columns are chr, start, end)
        cov_df = cov_df.sort_values(by=["chrom", "start", "end"]).reset_index(drop=True)

        # Extract the last column (the coverage value)
        coverage_values = cov_df["cov"]

        # Create column name in the format CNMOPS_{SAMPLE}_{OPERATION} (uppercase)
        col_name = f"CNMOPS_{sample_name.upper()}_{operation.upper()}"

        # Add the coverage values as a new column
        cnv_df[col_name] = coverage_values.to_numpy()

    return cnv_df


def add_ids(cnmops_cnv_df: pd.DataFrame) -> pd.DataFrame:
    """Add IDs to the CNV DataFrame in the format cnmops_<svtype>_<count>.

    Parameters
    ----------
    cnmops_cnv_df : pd.DataFrame
        Input

    Returns
    -------
    pd.DataFrame
        Output, ID added
    """
    # Add IDs in the format cnmops_<svtype>_<count>
    svtype_counts = {}
    ids = []
    for _, row in cnmops_cnv_df.iterrows():
        svtype = row["SVTYPE"].lower()
        svtype_counts[svtype] = svtype_counts.get(svtype, 0) + 1
        ids.append(f"cnmops_{svtype}_{svtype_counts[svtype]}")
    cnmops_cnv_df["ID"] = ids
    return cnmops_cnv_df


def annotate_bed(bed_file: str, lcr_cutoff: float, lcr_file: str, prefix: str, length_cutoff: int = 10000) -> str:
    """
    Annotate bed file with filters: lcr and length
    Parameters
    ----------
    bed_file : str
        Path to the input bed file.
    lcr_cutoff : float
        Intersection cutoff for LCR filtering.
    lcr_file : str
        Path to the UG-CNV-LCR bed file.
    prefix : str
        Prefix for output files.
    length_cutoff : int, optional
        Minimum CNV length for filtering, by default 10000.
    Returns
    -------
    str
        Path to the annotated bed file.
    """
    # get filters regions
    filter_files = []
    bu = bed_utils.BedUtils()

    if lcr_file is not None:
        lcr_bed_file = bu.filter_by_bed_file(bed_file, lcr_cutoff, lcr_file, prefix, "UG-CNV-LCR")
        filter_files.append(lcr_bed_file)

    if length_cutoff is not None and length_cutoff > 0:
        length_bed_file = bu.filter_by_length(bed_file, length_cutoff, prefix)
        filter_files.append(length_bed_file)

    if not filter_files:
        # No filters to apply, just return sorted bed file
        out_bed_file_sorted = prefix + os.path.splitext(os.path.basename(bed_file))[0] + ".annotate.bed"
        bu.bedtools_sort(bed_file, out_bed_file_sorted)
        return out_bed_file_sorted

    out_combined_info = prefix + os.path.splitext(os.path.basename(bed_file))[0] + ".unsorted.annotate.combined.bed"
    merge_filter_files(bed_file, filter_files, out_combined_info)
    # merge all filters and sort

    out_annotate = prefix + os.path.splitext(os.path.basename(bed_file))[0] + ".annotate.bed"
    bu.bedtools_sort(out_combined_info, out_annotate)
    os.unlink(out_combined_info)
    for f in filter_files:
        os.unlink(f)

    return out_annotate


def aggregate_coverages(
    annotated_bed_file: str, sample_norm_coverage_file: str, cohort_avg_coverage_file: str, tempdir: str
) -> list[tuple[str, str, str]]:
    """
    Prepare coverage annotations for aggregation.
    Parameters
    ----------
    annotated_bed_file : str
        Path to the annotated bed file.
    sample_norm_coverage_file : str
        Path to the sample normalized coverage bed file.
    cohort_avg_coverage_file : str
        Path to the cohort average coverage bed file.
    tempdir : str
        Directory to store intermediate files.
    Returns
    -------
    list of tuple
        List of tuples containing (sample/cohort cvg type, operation, bed_file_path) for coverage annotations.
    """
    coverage_annotations = []
    # annotate with coverage info
    input_sample = ["sample", "cohort"]
    output_param = ["mean", "stdev"]

    for isamp in input_sample:
        for oparam in output_param:
            out_annotate_bed_file_cov = annotated_bed_file.replace(".annotate.bed", f".annotate.{isamp}.{oparam}.bed")
            input_cov_file = sample_norm_coverage_file if isamp == "sample" else cohort_avg_coverage_file
            bed_utils.BedUtils().bedtools_map(
                a_bed=annotated_bed_file,
                b_bed=input_cov_file,
                output_bed=out_annotate_bed_file_cov,
                operation=oparam,
                presort=True,
                tempdir_prefix=tempdir,
                column=5,
            )
            coverage_annotations.append((isamp, oparam, out_annotate_bed_file_cov))
    return coverage_annotations
