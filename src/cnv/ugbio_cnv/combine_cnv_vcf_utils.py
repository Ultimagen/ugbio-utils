# utilities for combining VCFs
from os.path import join as pjoin
from pathlib import Path

import numpy as np
import pandas as pd
import pysam
from ugbio_core import misc_utils as mu
from ugbio_core.logger import logger
from ugbio_core.vcf_utils import VcfUtils
from ugbio_core.vcfbed import vcftools

# Module-level configuration for CNV INFO field aggregation
CNV_AGGREGATION_ACTIONS = {
    "weighted_avg": [
        "CNMOPS_SAMPLE_MEAN",
        "CNMOPS_SAMPLE_STDEV",
        "CNMOPS_COHORT_MEAN",
        "CNMOPS_COHORT_STDEV",
        "CopyNumber",
    ],
    "max": [
        "GAP_PERCENTAGE",
        "JALIGN_DEL_SUPPORT",
        "JALIGN_DUP_SUPPORT",
        "JALIGN_DEL_SUPPORT_STRONG",
        "JALIGN_DUP_SUPPORT_STRONG",
        "TREE_SCORE",
    ],
    "aggregate": ["CNV_SOURCE"],
    "minlength": ["CIPOS"],
}


# Helper function to check and add metadata records
def _add_metadata_records(
    record_dict1: pysam.VariantHeaderMetadata,
    record_dict2: pysam.VariantHeaderMetadata,
    record_type: str,
    enforced_info_specs: dict,
    combined_header: pysam.VariantHeader,
) -> None:
    """Add metadata records (INFO/FORMAT/FILTER) to combined header. (service function), modifies the header in place
    Parameters
    ----------
    record_dict1 : pysam.VariantHeaderMetadata
        Dictionary of metadata records from the first header
    record_dict2 : pysam.VariantHeaderMetadata
        Dictionary of metadata records from the second header
    record_type : str
        Type of metadata records ("INFO", "FORMAT", or "FILTER")
    enforced_info_specs : dict
        Dictionary of enforced specifications for certain INFO fields
    combined_header : pysam.VariantHeader
        The combined VCF header to add records to
    """

    # Collect all unique keys from both headers
    all_keys = set(record_dict1.keys()) | set(record_dict2.keys())

    for key in all_keys:
        # Check if this field has enforced specifications
        if record_type == "INFO" and key in enforced_info_specs:
            number, data_type, description = enforced_info_specs[key]
            combined_header.info.add(key, number, data_type, description)
            continue

        # Handle fields that exist in both headers
        if key in record_dict1 and key in record_dict2:
            record1 = record_dict1[key]
            record2 = record_dict2[key]

            # Check if type or number differ
            if hasattr(record1, "type") and hasattr(record2, "type"):
                if record1.type != record2.type:
                    msg = f"{record_type} field '{key}' has conflicting types: '{record1.type}' vs '{record2.type}'"
                    raise RuntimeError(msg)
            if hasattr(record1, "number") and hasattr(record2, "number"):
                if record1.number != record2.number:
                    msg = (
                        f"{record_type} field '{key}' has conflicting numbers: '{record1.number}' vs '{record2.number}'"
                    )
                    raise RuntimeError(msg)

            # If we get here, they're compatible - use the first one
            combined_header.add_line(str(record1.record))

        # Handle fields that only exist in header1
        elif key in record_dict1:
            combined_header.add_line(str(record_dict1[key].record))

        # Handle fields that only exist in header2
        elif key in record_dict2:
            combined_header.add_line(str(record_dict2[key].record))


def update_vcf_contig(
    vcf_utils: VcfUtils,
    input_vcf: str,
    fasta_index: str,
    output_directory: str,
    index: int = 0,
) -> str:
    """
    Update a VCF header with contigs from FASTA index.

    Parameters
    ----------
    vcf_utils : VcfUtils
        VcfUtils instance for VCF operations
    input_vcf : str
        Path to input VCF file
    fasta_index : str
        Path to FASTA index file
    output_directory : str
        Output directory for temporary files
    index : int, optional
        Index to append to filename for uniqueness (default: 0)

    Returns
    -------
    str
        Path to updated VCF file
    """

    # Generate unique output filename based on input filename and index
    input_basename = Path(input_vcf).name.replace(".vcf.gz", "").replace(".vcf", "")
    output_vcf = pjoin(output_directory, f"{input_basename}.{index}.updated_contigs.vcf.gz")

    logger.info(f"Updating VCF header with contigs from FASTA index: {input_vcf}")
    vcf_utils.update_vcf_contigs_from_fai(input_vcf, output_vcf, fasta_index)

    return output_vcf


def write_vcf_records_with_source(
    vcf_in: pysam.VariantFile,
    vcf_out: pysam.VariantFile,
    combined_header: pysam.VariantHeader,
    source_name: str,
    make_ids_unique: bool = False,
    seen_ids: set | None = None,
) -> set:
    """
    Write VCF records to output file with CNV_SOURCE annotation.

    Note: This function clears all FILTER values from input records since
    FILTER definitions are not included in the combined header.

    Parameters
    ----------
    vcf_in : pysam.VariantFile
        Input VCF file to read records from
    vcf_out : pysam.VariantFile
        Output VCF file to write records to
    combined_header : pysam.VariantHeader
        Combined header for creating new records
    source_name : str
        Source name to add to CNV_SOURCE INFO field
    make_ids_unique : bool, optional
        If True, ensure all IDs are unique by appending suffixes (default: False)
    seen_ids : set, optional
        Set of already-seen IDs for tracking uniqueness (default: None, creates new set)

    Returns
    -------
    set
        The updated set of seen IDs
    """
    logger.info(f"Writing records from {source_name} VCF")
    if seen_ids is None:
        seen_ids = set()

    for record in vcf_in:
        # Clear filters - we remove filters imposed by the previous pipelines
        record.filter.clear()
        record.filter.add("PASS")
        # Create new record with combined header
        new_record = VcfUtils.copy_vcf_record(record, combined_header)

        # Ensure unique ID if requested
        if make_ids_unique:
            original_id = new_record.id
            if original_id is None or original_id == ".":
                # Generate ID based on position if none exists
                original_id = f"{new_record.chrom}_{new_record.start}_{new_record.stop}"

            # Make ID unique if it already exists
            unique_id = original_id
            suffix = 1
            while unique_id in seen_ids:
                unique_id = f"{original_id}_{suffix}"
                suffix += 1

            new_record.id = unique_id
            seen_ids.add(unique_id)
        # Note: When make_ids_unique is False, IDs are not tracked and duplicates are allowed
        # This preserves the original behavior where duplicate IDs may exist

        # Add source tag if not already present
        if "CNV_SOURCE" not in new_record.info:
            new_record.info["CNV_SOURCE"] = (source_name,)
        vcf_out.write(new_record)

    return seen_ids


def combine_vcf_headers_for_cnv(
    header1: pysam.VariantHeader, header2: pysam.VariantHeader, *, keep_filters: bool = False
) -> pysam.VariantHeader:
    """
    Combine two VCF headers into a single header for CNV/SV variant calls.

    This function is specifically designed for merging headers from different CNV/SV
    callers (e.g., cn.mops and CNVpytor). It creates a new header that includes all
    metadata from both input headers with special handling for structural variant fields.

    For INFO and FORMAT fields with the same ID:
    - If type and number match, use the first definition
    - If type or number differ, raise RuntimeError

    FILTER fields are cleared unless keep_filters is set to True.

    Special enforcement for CNV/SV-related fields to ensure VCF spec compliance:
    - SVLEN: enforces Number="." (variable-length array) per VCF 4.2 spec for structural variants
    - SVTYPE: enforces Number=1 (single value) per VCF 4.2 spec for structural variants

    Parameters
    ----------
    header1 : pysam.VariantHeader
        The first VCF header (takes precedence in case of identical collisions)
    header2 : pysam.VariantHeader
        The second VCF header to merge
    keep_filters : bool, optional
        Whether to keep FILTER fields from both headers. Default is False (FILTERs are cleared).

    Returns
    -------
    pysam.VariantHeader
        A new combined VCF header with unified metadata

    Raises
    ------
    RuntimeError
        If INFO/FORMAT IDs collide with different type or number specifications
        (except for SVLEN and SVTYPE which are enforced to standard values)

    Examples
    --------
    >>> import pysam
    >>> header1 = pysam.VariantFile("cnmops.vcf.gz").header
    >>> header2 = pysam.VariantFile("cnvpytor.vcf.gz").header
    >>> combined = combine_vcf_headers_for_cnv(header1, header2)
    """
    # Enforced specifications for CNV/SV fields per VCF 4.2 spec
    enforced_info_specs = {
        "SVLEN": (".", "Integer", "Length of structural variant (SV) per VCF 4.2 specification"),
        "SVTYPE": (1, "String", "Type of structural variant (SV); e.g. DEL, DUP, etc. (per VCF 4.2 specification)"),
    }

    # Start with a copy of the first header
    combined_header = pysam.VariantHeader()

    # Copy contigs from header1 (assuming they're the same)
    for contig in header1.contigs.values():
        combined_header.contigs.add(contig.name, length=contig.length)

    # Add INFO fields
    _add_metadata_records(header1.info, header2.info, "INFO", enforced_info_specs, combined_header)

    # Add FORMAT fields
    _add_metadata_records(header1.formats, header2.formats, "FORMAT", enforced_info_specs, combined_header)

    if keep_filters:
        # Add FILTER fields
        _add_metadata_records(header1.filters, header2.filters, "FILTER", {}, combined_header)
    if tuple(header1.samples) != tuple(header2.samples):
        raise RuntimeError("Input VCF headers have different samples; cannot combine.")
    # Add samples from both headers
    for sample in header1.samples:
        if sample in combined_header.samples:
            continue
        combined_header.add_sample(sample)

    return combined_header


def cnv_vcf_to_bed(input_vcf: str, output_bed: str, *, assign_id=True) -> None:
    """Convert CNV VCF file to BED format with chr, start, end, ID columns."""
    logger.info(f"Converting CNV VCF to BED: {input_vcf} -> {output_bed}")
    with pysam.VariantFile(input_vcf) as vcf_in, open(output_bed, "w") as bed_out:
        for idx, record in enumerate(vcf_in):
            if assign_id:
                name = f"CNV_{idx:09d}"
                bed_out.write(f"{record.contig}\t{record.start}\t{record.stop}\t{name}\n")
            else:
                bed_out.write(f"{record.contig}\t{record.start}\t{record.stop}\n")


def _prepare_update_dataframe(removed_vcf: str, aggregation_fields: list[str], *, ignore_filter: bool) -> pd.DataFrame:
    """
    Load and prepare dataframe of collapsed variants for aggregation.

    Parameters
    ----------
    removed_vcf : str
        Path to the VCF file containing removed/collapsed variants
    aggregation_fields : list[str]
        List of INFO field names to include in the dataframe
    ignore_filter : bool
        Whether to include filtered variants in the update dataframe

    Returns
    -------
    pd.DataFrame
        Filtered dataframe with matchid, pos, end columns ready for aggregation
    """
    update_df = vcftools.get_vcf_df(
        str(removed_vcf), custom_info_fields=aggregation_fields + ["SVLEN", "MatchId"]
    ).sort_index()

    update_df["matchid"] = update_df["matchid"].apply(lambda x: x[0]).astype(float)
    update_df["end"] = update_df["pos"] + update_df["svlen"].apply(lambda x: x[0]) - 1

    # When ignore_filter=False, exclude filtered variants from aggregation/boundary adjustment.
    # Treat NaN, empty string, ".", and "PASS" as not filtered, to match
    # _remove_overlapping_filtered_variants() behavior.
    if not ignore_filter:
        unselect = (
            update_df["filter"].notna()
            & (update_df["filter"] != "PASS")
            & (update_df["filter"] != "")
            & (update_df["filter"] != ".")
        )
        update_df = update_df.loc[~unselect]

    return update_df


def _aggregate_record_info_fields(
    record: pysam.VariantRecord,
    update_records: pd.DataFrame,
    aggregation_actions: dict[str, list[str]],
    header: pysam.VariantHeader,
) -> None:
    """
    Aggregate INFO fields for a single record based on collapsed variants.

    Modifies the record in-place by:
    1. Updating start/stop boundaries to span all collapsed variants
    2. Aggregating INFO field values using specified actions
    3. Updating SVLEN based on new boundaries
    4. Removing collapse metadata fields

    Parameters
    ----------
    record : pysam.VariantRecord
        The variant record to update
    update_records : pd.DataFrame
        Dataframe containing collapsed variants for this record
    aggregation_actions : dict[str, list[str]]
        Dictionary mapping aggregation action to list of INFO field names
    header : pysam.VariantHeader
        VCF header containing INFO field definitions
    """
    # Update boundaries to span all collapsed variants
    record.start = min(update_records["pos"].min(), record.start)
    record.stop = max(update_records["end"].max(), record.stop)

    # Aggregate INFO fields based on defined actions
    for action, fields in aggregation_actions.items():
        for field in fields:
            if field in header.info:
                _value_aggregator(
                    record,
                    update_records,
                    field,
                    action,
                    header.info[field].number,
                    header.info[field].type,
                )

    # Update SVLEN to match new boundaries
    record.info["SVLEN"] = (record.stop - record.start,)

    # Clean up collapse metadata fields
    collapse_info_fields = ["CollapseId", "NumCollapsed", "NumConsolidated"]
    for field_name in collapse_info_fields:
        if field_name in record.info:
            del record.info[field_name]


def _aggregate_collapsed_vcf(
    input_vcf_path: str,
    output_vcf_path: str,
    update_df: pd.DataFrame,
    aggregation_actions: dict[str, list[str]],
) -> None:
    """
    Read collapsed VCF, aggregate INFO fields, and write to output.

    Parameters
    ----------
    input_vcf_path : str
        Path to the collapsed VCF file to read
    output_vcf_path : str
        Path where aggregated VCF will be written
    update_df : pd.DataFrame
        Dataframe containing collapsed variant information for aggregation
    aggregation_actions : dict[str, list[str]]
        Dictionary mapping aggregation action to list of INFO field names
    """
    with pysam.VariantFile(input_vcf_path) as vcf_in:
        header = vcf_in.header
        with pysam.VariantFile(output_vcf_path, "w", header=header) as vcf_out:
            for record in vcf_in:
                if "CollapseId" in record.info:
                    cid = float(record.info["CollapseId"])
                    update_records = update_df[update_df["matchid"] == cid]
                    if not update_records.empty:
                        _aggregate_record_info_fields(record, update_records, aggregation_actions, header)
                vcf_out.write(record)


def merge_cnvs_in_vcf(
    input_vcf: str,
    output_vcf: str,
    distance: int = 1000,
    *,
    ignore_sv_type: bool = False,
    ignore_filter: bool = True,
    pick_best: bool = False,
) -> None:
    """
    Merge CNV variants in a VCF file that are within a specified distance.

    This function performs a two-stage merge process:
    1. Collapse overlapping variants using VcfUtils.collapse_vcf, which groups variants
       within the specified distance into single representative records.
    2. Aggregate INFO fields from collapsed variants using weighted averaging (for means/stdevs),
       max (for quality/support scores), or concatenation (for sources).
    3. Adjust variant boundaries based on the minimum start and maximum end positions
       of all variants in each collapsed group.

    Variants are considered for merging based on:
    - Reference distance: Variants within `distance` bp are candidates for merging
    - SV type matching: Unless `ignore_sv_type=True`, only variants with the same SVTYPE merge
    - Filter status: When `ignore_filter=False`, only PASS variants participate in collapsing
      and aggregation. Filtered variants overlapping with PASS variants are removed in a
      second-stage filtering step.

    When `ignore_filter=False`, the function performs an additional filtering pass to remove
    any filtered variants that overlap with PASS variants, ensuring that high-confidence
    calls take precedence over low-confidence calls.

    Parameters
    ----------
    input_vcf : str
        Path to the input VCF file containing CNV variants.
    output_vcf : str
        Path to the output VCF file where merged variants will be written.
    distance : int, optional
        Maximum distance between CNV variants to consider them for merging, by default 1000.
    ignore_sv_type : bool, optional
        Whether to ignore SVTYPE when collapsing variants, by default False.
    ignore_filter: bool, optional
        Whether to ignore FILTER status when collapsing and aggregating variants, by default True.
        When False, filtered variants are excluded from both the initial collapse and from
        aggregation/boundary adjustments. Filtered variants overlapping with PASS variants
        are also removed in a second-stage filtering step.
    pick_best : bool, optional
        Whether to pick the best variant (by QUAL) among those being merged (or the first: False), by default False.
    Returns
    -------
    None
        Writes the merged VCF to output_vcf and creates an index.
    """

    # Stage 1: Collapse overlapping variants into representative records
    output_vcf_collapse = output_vcf + ".collapse.tmp.vcf.gz"
    temporary_files = [output_vcf_collapse]

    vu = VcfUtils()
    removed_vcf = vu.collapse_vcf(
        vcf=input_vcf,
        output_vcf=output_vcf_collapse,
        refdist=distance,
        pctseq=0.0,
        pctsize=0.0,
        maxsize=-1,
        ignore_filter=ignore_filter,
        ignore_sv_type=ignore_sv_type,
        pick_best=pick_best,
        erase_removed=False,
    )
    temporary_files.extend([str(removed_vcf), output_vcf_collapse])

    # Stage 2: Prepare dataframe and aggregate INFO fields
    all_fields = sum(CNV_AGGREGATION_ACTIONS.values(), [])
    update_df = _prepare_update_dataframe(str(removed_vcf), all_fields, ignore_filter=ignore_filter)

    output_vcf_unsorted = output_vcf.replace(".vcf.gz", ".unsorted.vcf.gz")
    temporary_files.append(output_vcf_unsorted)

    _aggregate_collapsed_vcf(output_vcf_collapse, output_vcf_unsorted, update_df, CNV_AGGREGATION_ACTIONS)

    # Stage 3: Remove overlapping filtered variants (if applicable)
    if not ignore_filter:
        output_vcf_unsorted = _remove_overlapping_filtered_variants(
            input_vcf,
            output_vcf_unsorted,
            output_vcf,
            distance=distance,
            ignore_sv_type=ignore_sv_type,
            pick_best=pick_best,
        )
        temporary_files.append(output_vcf_unsorted)

    # Stage 4: Sort and cleanup
    vu.sort_vcf(output_vcf_unsorted, output_vcf)
    mu.cleanup_temp_files(temporary_files)


def _remove_overlapping_filtered_variants(
    original_vcf: str, merged_vcf: str, output_vcf: str, distance: int, *, ignore_sv_type: bool, pick_best: bool
) -> str:
    """
    Remove merged records that overlap variants filtered by a second pass.

    This function performs a "second-stage" filtering step after an initial
    merge/collapse of CNV calls. It re-runs `VcfUtils.collapse_vcf` on the
    original input VCF to identify overlapping variants and their filters,
    then removes the corresponding records from the already merged VCF.
    """
    output_vcf_collapse = output_vcf + ".collapse.tmp.2.vcf.gz"
    vu = VcfUtils()
    removed_vcf = vu.collapse_vcf(
        vcf=original_vcf,
        output_vcf=output_vcf_collapse,
        refdist=distance,
        pctseq=0.0,
        pctsize=0.0,
        maxsize=-1,
        ignore_filter=True,
        ignore_sv_type=ignore_sv_type,
        pick_best=pick_best,
        erase_removed=False,
    )
    removed_df = vcftools.get_vcf_df(str(removed_vcf))
    filtered_out = ~(
        pd.isna(removed_df["filter"])
        | (removed_df["filter"] == "")
        | (removed_df["filter"] == ".")
        | (removed_df["filter"] == "PASS")
    )
    # TODO [BIOIN-2653]: make sure the IDs are unique at this point, so no need to do complicated ID
    # handling. Use itertuples for efficiency on large DataFrames instead of DataFrame.apply(axis=1).
    filtered_id_pos = removed_df.loc[filtered_out, ["id", "chrom", "pos"]]
    remove_ids = set(filtered_id_pos.itertuples(index=False, name=None))
    with pysam.VariantFile(merged_vcf) as vcf_in:
        hdr = vcf_in.header
        with pysam.VariantFile(output_vcf_collapse, "w", header=hdr) as vcf_out:
            for record in vcf_in:
                if (record.id, record.contig, record.pos) in remove_ids:
                    continue
                vcf_out.write(record)
    mu.cleanup_temp_files([str(removed_vcf)])
    return output_vcf_collapse


def _aggregate_weighted_avg(
    record: pysam.VariantRecord,
    update_records: pd.DataFrame,
    field: str,
    values: list,
) -> None:
    """Aggregate field using weighted average based on SVLEN."""
    lengths = np.array(list(update_records["svlen"].apply(lambda x: x[0])) + [record.info["SVLEN"][0]])
    values_array = np.array(values)
    if all(pd.isna(x) for x in values):
        return
    drop = np.isnan(values_array) | np.isnan(lengths)
    values_array = values_array[~drop]
    lengths = lengths[~drop]
    weighted_avg = np.sum(values_array * lengths) / np.sum(lengths)
    record.info[field] = round(weighted_avg, 3)


def _aggregate_max(
    record: pysam.VariantRecord,
    field: str,
    values: list,
    val_type: str | None,
) -> None:
    """Aggregate field using maximum value."""
    if all(pd.isna(x) for x in values):
        return
    if str(val_type) == "Float":
        record.info[field] = float(np.nanmax(values))
    elif str(val_type) == "Integer":
        record.info[field] = int(np.nanmax(values))
    else:
        raise ValueError(f"Unsupported value type for max aggregation: {val_type}")


def _aggregate_min(
    record: pysam.VariantRecord,
    field: str,
    values: list,
    val_type: str | None,
) -> None:
    """Aggregate field using element-wise minimum for tuples."""
    valid_values = [v for v in values if v is not None and not (isinstance(v, float) and pd.isna(v))]
    if not valid_values:
        return
    values_array = np.array(valid_values)
    min_values = np.min(values_array, axis=0)
    if str(val_type) == "Float":
        record.info[field] = tuple([float(x) for x in min_values])
    elif str(val_type) == "Integer":
        record.info[field] = tuple([int(x) for x in min_values])
    else:
        raise ValueError(f"Unsupported value type for min aggregation: {val_type}")


def _aggregate_minlength(
    record: pysam.VariantRecord,
    field: str,
    values: list,
    val_type: str | None,
) -> None:
    """Aggregate field by selecting tuple with minimum interval length (x[1] - x[0])."""
    valid_values = [v for v in values if v is not None and not (isinstance(v, float) and pd.isna(v))]
    if not valid_values:
        return
    min_tuple = min(valid_values, key=lambda x: x[1] - x[0])
    if str(val_type) == "Float":
        record.info[field] = tuple([float(x) for x in min_tuple])
    elif str(val_type) == "Integer":
        record.info[field] = tuple([int(x) for x in min_tuple])
    else:
        raise ValueError(f"Unsupported value type for minlength aggregation: {val_type}")


def _aggregate_set(
    record: pysam.VariantRecord,
    field: str,
    values: list,
) -> None:
    """Aggregate field by collecting unique values."""
    record.info[field] = tuple(set(values))


def _value_aggregator(
    record: pysam.VariantRecord,
    update_records: pd.DataFrame,
    field: str,
    action: str,
    val_number: str | None,
    val_type: str | None,
):
    """
    Helper function to aggregate INFO field values based on specified action.

    Parameters
    ----------
    record : pysam.VariantRecord
        VCF record to update
    update_records : pd.DataFrame
        DataFrame containing records to aggregate
    field : str
        INFO field name to aggregate
    action : str
        Aggregation action to perform (weighted_avg, max, min, minlength, aggregate)
    val_number : str | None
        Number specification for the field
    val_type : str | None
        Type specification for the field
    """
    if field.lower() not in update_records.columns:
        return

    # Collect values based on field number specification
    values = []
    if val_number == 1:
        values = list(update_records[field.lower()]) + [record.info.get(field, np.nan)]
    elif str(val_number) == ".":
        values = list(update_records[field.lower()]) + [record.info.get(field, (None,))]
        values = [item for sublist in values for item in sublist if item is not None]
    elif isinstance(val_number, int) and val_number > 1:
        values = list(update_records[field.lower()]) + [record.info.get(field, None)]
    else:
        raise ValueError(f"Unsupported value number for aggregation: {val_number}")

    # Dispatch to appropriate aggregation function
    if action == "weighted_avg":
        _aggregate_weighted_avg(record, update_records, field, values)
    elif action == "max":
        _aggregate_max(record, field, values, val_type)
    elif action == "min":
        _aggregate_min(record, field, values, val_type)
    elif action == "minlength":
        _aggregate_minlength(record, field, values, val_type)
    elif action == "aggregate":
        _aggregate_set(record, field, values)
    else:
        raise ValueError(f"Unsupported aggregation action: {action}")
