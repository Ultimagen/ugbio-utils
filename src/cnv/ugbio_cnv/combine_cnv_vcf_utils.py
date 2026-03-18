# utilities for combining VCFs
import os
import shutil
from collections import defaultdict
from os.path import join as pjoin
from pathlib import Path

import numpy as np
import pandas as pd
import pysam
from ugbio_cnv.cnv_vcf_consts import INFO_TAG_REGISTRY
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
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
    "weighted_majority": ["SVTYPE"],
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
    *,
    make_ids_unique: bool = False,
    seen_ids: set | None = None,
) -> set[str]:
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


def _prepare_update_dataframe(
    removed_vcf: str, collapsed_vcf: str, aggregation_fields: list[str], *, ignore_filter: bool
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and prepare dataframe of collapsed variants for aggregation.

    Parameters
    ----------
    removed_vcf : str
        Path to the VCF file containing removed variants from collapsing
    collapsed_vcf : str
        Path to the VCF file containing collapsed variants
    aggregation_fields : list[str]
        List of INFO field names to include in the dataframe
    ignore_filter : bool
        Whether to include filtered variants in the update dataframe

    Returns
    -------
    pd.DataFrame, pd.DataFrame
        Filtered dataframe with matchid, pos, end columns ready for aggregation,
        collapsed_df in the same format
    """
    update_df = vcftools.get_vcf_df(str(removed_vcf), custom_info_fields=aggregation_fields + ["SVLEN", "MatchId"])
    collapsed_df = vcftools.get_vcf_df(str(collapsed_vcf), custom_info_fields=aggregation_fields + ["SVLEN", "MatchId"])

    update_df["matchid"] = update_df["matchid"].apply(lambda x: x[0]).astype(float)
    update_df["end"] = update_df["pos"] + update_df["svlen"].apply(lambda x: x[0])

    # Process collapsed_df - only need "end" column, not matchid (we use index instead)
    collapsed_df["end"] = collapsed_df["pos"] + collapsed_df["svlen"].apply(lambda x: x[0])

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

    return update_df, collapsed_df


def _aggregate_record_info_fields(
    record: pysam.VariantRecord,
    update_records: pd.DataFrame,
    aggregation_actions: dict[str, list[str]],
    header: pysam.VariantHeader,
) -> None:
    """
    Aggregate INFO fields for a single record based on collapsed/merged variants.

    Used by both distance-based collapse (Stage 1) and size-scaled gap smoothing (Stage 3).
    The CIPOS-based boundary selection works correctly for both overlapping CNVs (collapse)
    and non-overlapping CNVs (smoothing) because the 2500bp window captures all relevant
    candidates in both scenarios.

    Modifies the record in-place by:
    1. Updating start/stop boundaries using CIPOS window selection
    2. Aggregating INFO field values using specified actions
    3. Updating SVLEN based on new boundaries
    4. Removing collapse metadata fields (no-op for smoothing records)

    Parameters
    ----------
    record : pysam.VariantRecord
        The variant record to update (used as template for merged result)
    update_records : pd.DataFrame
        Dataframe containing variants to aggregate into this record.
        For collapse: includes removed records + collapsed record's original position
        For smoothing: includes all records in the smoothing group
    aggregation_actions : dict[str, list[str]]
        Dictionary mapping aggregation action to list of INFO field names
    header : pysam.VariantHeader
        VCF header containing INFO field definitions
    """
    record.pos, record.stop = _select_breakpoints_by_cipos_window(update_records)

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
    record.info["SVLEN"] = (record.stop - record.start - 1,)

    # Update genotype to match aggregated SVTYPE
    _update_genotype_from_svtype(record)

    # Clean up collapse metadata fields
    collapse_info_fields = ["CollapseId", "NumCollapsed", "NumConsolidated"]
    for field_name in collapse_info_fields:
        if field_name in record.info:
            del record.info[field_name]


def _select_breakpoints_by_cipos_window(
    update_records: pd.DataFrame,
    window: int = 2500,
) -> tuple[int, int]:
    """Select start/end breakpoints near current boundaries using tightest CIPOS."""
    candidates = update_records[["pos", "end", "cipos"]].copy()

    def _best_breakpoint(df: pd.DataFrame, boundary_col: str, target: int) -> int:
        nearby = df[(df[boundary_col] - target).abs() <= window]
        cipos_lengths = nearby["cipos"].apply(lambda x: x[1] - x[0] - 1).astype(float)
        min_len = cipos_lengths.min()
        tied = nearby[cipos_lengths == min_len]
        if boundary_col == "pos":
            return int(tied[boundary_col].min())
        return int(tied[boundary_col].max())

    new_start = _best_breakpoint(candidates, "pos", candidates["pos"].min())
    new_stop = _best_breakpoint(candidates, "end", candidates["end"].max())
    return new_start, new_stop


def _aggregate_collapsed_vcf(
    input_vcf_path: str,
    output_vcf_path: str,
    collapse_df: pd.DataFrame,
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
    collapse_df: pd.DataFrame
        input_vcf in dataframe form, needed for the aggregation
    update_df : pd.DataFrame
        Dataframe containing collapsed variant information for aggregation
    aggregation_actions : dict[str, list[str]]
        Dictionary mapping aggregation action to list of INFO field names
    """
    with pysam.VariantFile(input_vcf_path) as vcf_in:
        header = vcf_in.header
        with pysam.VariantFile(output_vcf_path, "w", header=header) as vcf_out:
            for i, record in enumerate(vcf_in):
                if "CollapseId" in record.info:
                    cid = float(record.info["CollapseId"])
                    update_records = update_df[update_df["matchid"] == cid]
                    if not update_records.empty:
                        # Concatenate update_records with collapsed record's original position
                        update_records_with_collapsed = pd.concat(
                            (update_records, collapse_df.iloc[[i]]), ignore_index=True
                        )
                        _aggregate_record_info_fields(
                            record, update_records_with_collapsed, aggregation_actions, header
                        )
                vcf_out.write(record)


def _verify_unique_ids(vcf_path: str) -> None:
    """
    Verify that all variant IDs in a VCF file are unique.

    Parameters
    ----------
    vcf_path : str
        Path to the VCF file to check

    Raises
    ------
    ValueError
        If duplicate IDs are found in the VCF file

    Notes
    -----
    This validation ensures that variant IDs can be used reliably for matching
    and tracking variants through merge operations. Records with missing IDs
    (None or ".") are allowed and not counted as duplicates.
    """
    seen_ids = set()

    with pysam.VariantFile(vcf_path) as vcf_in:
        for record in vcf_in:
            variant_id = record.id
            # Skip records with missing IDs
            if variant_id is None or variant_id == ".":
                raise ValueError(
                    "Collapsing callsets only works when the variants have unique IDs. Found record with missing ID."
                )

            if variant_id in seen_ids:
                raise ValueError(
                    f"VCF file contains duplicate variant IDs: {variant_id}. "
                    "All variant IDs must be unique for merge operations. "
                    "Consider using the --make_ids_unique flag when combining VCF files."
                )
            else:
                seen_ids.add(variant_id)


def merge_cnvs_in_vcf(  # noqa: PLR0915
    input_vcf: str,
    output_vcf: str,
    distance: int = 1000,
    *,
    ignore_sv_type: bool = False,
    ignore_filter: bool = True,
    pick_best: bool = False,
    enable_smoothing: bool = False,
    max_gap_absolute: int = 50000,
    gap_scale_fraction: float = 0.05,
    cipos_threshold: int = 50,
) -> None:
    """
    Merge CNV variants in a VCF file that are within a specified distance.

    This function performs a multi-stage merge process:

    **Standard Mode (enable_smoothing=False):**
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

    **Smoothing Mode (enable_smoothing=True):**
    Adds additional post-collapse smoothing using size-scaled gap rules:
    1. Stage 1-3: Standard distance-based collapse (as above)
    3. Stage 4: Identify additional smoothing candidates based on size-scaled gaps
    4. Stage 5: Merge smoothing candidates
    5. Stage 6: Final sort

    Size-scaled smoothing uses the formula: gap_threshold = min(max_gap_absolute, gap_scale_fraction × larger_CNV)
    This allows large CNVs (>1 Mb) to merge across proportionally larger gaps while the absolute cap
    prevents over-merging of small CNVs.

    Smoothing Criteria (all must be satisfied):
    - Same SVTYPE (unless ignore_sv_type=True)
    - CIPOS length >= cipos_threshold for both CNVs (prevents merging high-confidence breakpoints)
    - Gap between CNVs <= size-scaled threshold
    - FILTER status compatible (if ignore_filter=False, only PASS CNVs considered)


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
    enable_smoothing : bool, optional
        Enable size-scaled gap smoothing (BIOIN-2622), by default False.
        When True, performs additional post-collapse smoothing based on CNV size.
    max_gap_absolute : int, optional
        Absolute maximum gap for merging CNVs in smoothing mode (bp), by default 50000.
        This caps the maximum gap regardless of CNV size.
    gap_scale_fraction : float, optional
        Gap as fraction of larger CNV length for smoothing, by default 0.05 (5%).
        Formula: gap_threshold = min(max_gap_absolute, gap_scale_fraction × larger_CNV)
    cipos_threshold : int, optional
        Minimum CIPOS length required for smoothing (bp), by default 50.
        CNVs with CIPOS length < threshold have high-confidence breakpoints and won't be
        smoothed. CIPOS length = max - min - 1(e.g., (-50, 51) → 100 bp).

    Returns
    -------
    None
        Writes the merged VCF to output_vcf and creates an index.

    Raises
    ------
    ValueError
        If the input VCF contains duplicate variant IDs
        If enable_smoothing=True and any CNV is missing CIPOS values

    Examples
    --------
    >>> # Standard merge (backward compatible)
    >>> merge_cnvs_in_vcf("input.vcf.gz", "output.vcf.gz", distance=1000)

    >>> # Size-scaled gap smoothing
    >>> merge_cnvs_in_vcf(
    ...     "input.vcf.gz",
    ...     "output.vcf.gz",
    ...     distance=1000,
    ...     enable_smoothing=True,
    ...     max_gap_absolute=50000,
    ...     gap_scale_fraction=0.05,
    ...     cipos_threshold=50,
    ... )
    """

    logger.info(f"Validating unique variant IDs in {input_vcf}")
    _verify_unique_ids(input_vcf)

    # Log mode selection
    if enable_smoothing:
        logger.info(
            f"Using size-scaled gap smoothing (max_gap={max_gap_absolute}bp, "
            f"scale={gap_scale_fraction}, cipos_threshold={cipos_threshold}bp)"
        )
    else:
        logger.info("Using standard merge (enable_smoothing=False)")

    # Stage 1: Distance-based collapse and aggregation (common to both modes)
    vu = VcfUtils()
    temporary_files = []

    output_vcf_collapse = output_vcf + (".distance_merge.tmp.vcf.gz" if enable_smoothing else ".collapse.tmp.vcf.gz")
    temporary_files.append(output_vcf_collapse)

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
    temporary_files.append(str(removed_vcf))

    # Stage 1.25: Aggregate INFO fields (common to both modes)
    all_fields = sum(CNV_AGGREGATION_ACTIONS.values(), [])
    update_df, collapsed_df = _prepare_update_dataframe(
        str(removed_vcf), output_vcf_collapse, all_fields, ignore_filter=ignore_filter
    )

    output_vcf_unsorted = output_vcf.replace(".vcf.gz", ".unsorted.vcf.gz")
    temporary_files.append(output_vcf_unsorted)

    _aggregate_collapsed_vcf(output_vcf_collapse, output_vcf_unsorted, collapsed_df, update_df, CNV_AGGREGATION_ACTIONS)

    # Stage 1.5: Remove overlapping filtered variants if applicable
    # TODO [BIOIN-2745]: it should be actually possible to do _remove_overlapping_filtered_variants only once.
    # Need to refactor the code
    if not ignore_filter:
        output_vcf_unsorted = _remove_overlapping_filtered_variants(
            input_vcf,
            output_vcf_unsorted,
            distance=distance,
            ignore_sv_type=ignore_sv_type,
            pick_best=pick_best,
        )
        temporary_files.append(output_vcf_unsorted)

    # Standard mode: sort and finish
    logger.info("Stage 1.5: Sorting collapsed VCF")
    output_vcf_sorted = output_vcf if not enable_smoothing else output_vcf + ".stage1.sorted.vcf.gz"
    vu.sort_vcf(output_vcf_unsorted, output_vcf_sorted)
    vu.index_vcf(output_vcf_sorted)
    if not enable_smoothing:
        mu.cleanup_temp_files(temporary_files)
        return
    else:
        temporary_files.append(output_vcf_sorted)
        temporary_files.append(output_vcf_sorted + ".tbi")

    # Stage 2: Identify size-scaled gap smoothing candidates
    logger.info("Stage 2: Identifying size-scaled gap smoothing candidates")
    smoothing_candidates = identify_smoothing_candidates(
        output_vcf_sorted,
        max_gap_absolute,
        gap_scale_fraction,
        ignore_sv_type=ignore_sv_type,
        ignore_filter=ignore_filter,
        cipos_threshold=cipos_threshold,
    )

    if not smoothing_candidates:
        logger.info("No smoothing candidates found - using Stage 1 output")
        shutil.copy(output_vcf_sorted, output_vcf)
        shutil.copy(output_vcf_sorted + ".tbi", output_vcf + ".tbi")
        mu.cleanup_temp_files(temporary_files)
        return

    # Stage 3: Apply smoothing
    logger.info(f"Stage 3: Applying smoothing to {len(smoothing_candidates)} candidate pairs")
    smoothing_groups = _group_candidates_transitively(smoothing_candidates)

    # Log smoothing statistics
    with pysam.VariantFile(output_vcf_sorted) as vcf_in:
        cnvs_before_smoothing = sum(1 for _ in vcf_in)
    total_cnvs_in_groups = sum(len(group) for group in smoothing_groups)
    cnvs_after_smoothing = cnvs_before_smoothing - total_cnvs_in_groups + len(smoothing_groups)
    logger.info(
        f"Smoothing will merge {total_cnvs_in_groups} CNVs into {len(smoothing_groups)} groups "
        f"(reducing {cnvs_before_smoothing} → {cnvs_after_smoothing} CNVs)"
    )

    output_vcf_smoothed = output_vcf + ".smoothed.unsorted.vcf.gz"
    temporary_files.append(output_vcf_smoothed)
    _apply_smoothing_merges(
        output_vcf_sorted, output_vcf_smoothed, smoothing_groups, CNV_AGGREGATION_ACTIONS, ignore_filter=ignore_filter
    )

    # Stage 4: Remove overlapping filtered variants again
    if not ignore_filter:
        output_vcf_smoothed = _remove_overlapping_filtered_variants(
            output_vcf_sorted,
            output_vcf_smoothed,
            distance=distance,
            ignore_sv_type=ignore_sv_type,
            pick_best=pick_best,
        )
        temporary_files.append(output_vcf_smoothed)

    # Stage 4: Final sort and cleanup
    logger.info("Stage 4: Final sort")
    vu.sort_vcf(output_vcf_smoothed, output_vcf)
    vu.index_vcf(output_vcf)
    mu.cleanup_temp_files(temporary_files)


def _remove_overlapping_filtered_variants(
    original_vcf: str, merged_vcf: str, distance: int, *, ignore_sv_type: bool, pick_best: bool
) -> str:
    """
    Remove merged records that overlap variants filtered by a second pass.

    This function performs a "second-stage" filtering step after an initial
    merge/collapse of CNV calls. It re-runs `VcfUtils.collapse_vcf` on the
    original input VCF to identify overlapping variants and their filters,
    then removes the corresponding records from the already merged VCF.
    """
    output_vcf_collapse = merged_vcf + ".collapse.tmp.2.vcf.gz"
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
    # With unique IDs enforced, we can simply use ID for matching
    remove_ids = set(removed_df.loc[filtered_out, "id"])
    with pysam.VariantFile(merged_vcf) as vcf_in:
        hdr = vcf_in.header
        with pysam.VariantFile(output_vcf_collapse, "w", header=hdr) as vcf_out:
            for record in vcf_in:
                if record.id in remove_ids:
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
    lengths = np.array(list(update_records["svlen"].apply(lambda x: x[0])))
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
    update_records: pd.DataFrame,
    field: str,
    values: list,
    val_type: str | None,
    window: int = 2500,
) -> None:
    """
    Aggregate field by selecting tuple with minimum interval length from breakpoints within window.

    For CIPOS aggregation, only considers breakpoints that are within the specified window
    of the current record's boundaries (after they have been updated by _select_breakpoints_by_cipos_window).
    """
    valid_values = [v for v in values if v is not None and not (isinstance(v, float) and pd.isna(v))]
    if not valid_values:
        return

    # For CIPOS field, apply window filtering
    if field == "CIPOS":
        # Create candidates DataFrame with pos, end, and cipos from update_records
        # Note: update_records now includes the collapsed record's ORIGINAL position
        # (from collapse_df added in _aggregate_collapsed_vcf before boundary update)
        candidates = update_records[["pos", "end", "cipos"]].copy()

        # Filter to only breakpoints within window of current boundaries
        # A record is included if either its start OR end is within window
        near_start = (candidates["pos"] - record.pos).abs() <= window
        near_end = (candidates["end"] - record.stop).abs() <= window
        filtered_candidates = candidates[near_start | near_end]

        # Select CIPOS with minimum length from filtered candidates
        if not filtered_candidates.empty:
            valid_cipos = [
                c for c in filtered_candidates["cipos"] if c is not None and not (isinstance(c, float) and pd.isna(c))
            ]
            if valid_cipos:
                min_tuple = min(valid_cipos, key=lambda x: x[1] - x[0])
            else:
                # Fallback if no valid CIPOS in filtered candidates
                min_tuple = min(valid_values, key=lambda x: x[1] - x[0])
        else:
            # Fallback if no candidates in window
            min_tuple = min(valid_values, key=lambda x: x[1] - x[0])
    else:
        # For non-CIPOS fields, use original behavior
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


def _aggregate_weighted_majority(
    record: pysam.VariantRecord,
    update_records: pd.DataFrame,
    field: str,
    values: list,
) -> None:
    """
    Aggregate field using SVLEN-weighted majority vote.

    For SVTYPE: computes total SVLEN for each SVTYPE value and selects
    the SVTYPE with the highest cumulative length.

    Parameters
    ----------
    record : pysam.VariantRecord
        VCF record to update
    update_records : pd.DataFrame
        DataFrame containing records to aggregate
    field : str
        INFO field name to aggregate
    values : list
        List of field values from all records being merged
    """
    # Collect SVLEN values from update_records
    # Note: update_records now includes collapsed record's original SVLEN
    # (added in _aggregate_collapsed_vcf), so don't add record.info["SVLEN"] again
    lengths = list(update_records["svlen"].apply(lambda x: x[0]))

    # Filter out None values
    valid_pairs = [
        (v, abs(length)) for v, length in zip(values, lengths, strict=True) if v is not None and not pd.isna(v)
    ]

    if not valid_pairs:
        return

    # Group by field value and sum absolute SVLENs
    value_weights = defaultdict(int)
    for value, length in valid_pairs:
        value_weights[value] += length

    # Select value with highest total SVLEN
    majority_value = max(value_weights.items(), key=lambda x: x[1])[0]
    record.info[field] = majority_value


def _update_genotype_from_svtype(record: pysam.VariantRecord) -> None:
    """
    Update genotype (GT) field based on SVTYPE.

    - DEL → GT=(0,1) heterozygous deletion
    - DUP → GT=(None,1) duplication

    Parameters
    ----------
    record : pysam.VariantRecord
        VCF record to update genotype for
    """
    if "SVTYPE" not in record.info or not record.samples:
        return

    svtype = record.info["SVTYPE"]
    sample_name = list(record.samples.keys())[0]  # CNV records have single sample

    if svtype == "DEL":
        record.alts = ("<DEL>",)
        record.samples[sample_name]["GT"] = (0, 1)
    elif svtype == "DUP":
        record.alts = ("<DUP>",)
        record.samples[sample_name]["GT"] = (None, 1)


def _collect_field_values(
    update_records: pd.DataFrame,
    field: str,
    val_number: str | None,
) -> list:
    """Collect field values from records based on field number specification."""
    if val_number == 1:
        return list(update_records[field.lower()])
    if str(val_number) == ".":
        values = list(update_records[field.lower()])
        return [item for sublist in values for item in sublist if item is not None]
    if isinstance(val_number, int) and val_number > 1:
        return list(update_records[field.lower()])
    raise ValueError(f"Unsupported value number for aggregation: {val_number}")


def _dispatch_aggregation(
    action: str,
    record: pysam.VariantRecord,
    update_records: pd.DataFrame,
    field: str,
    values: list,
    val_type: str | None,
):
    """Dispatch to appropriate aggregation function based on action."""
    if action == "weighted_avg":
        _aggregate_weighted_avg(record, update_records, field, values)
    elif action == "max":
        _aggregate_max(record, field, values, val_type)
    elif action == "min":
        _aggregate_min(record, field, values, val_type)
    elif action == "minlength":
        _aggregate_minlength(record, update_records, field, values, val_type)
    elif action == "aggregate":
        _aggregate_set(record, field, values)
    elif action == "weighted_majority":
        _aggregate_weighted_majority(record, update_records, field, values)
    else:
        raise ValueError(f"Unsupported aggregation action: {action}")


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

    values = _collect_field_values(update_records, field, val_number)
    _dispatch_aggregation(action, record, update_records, field, values, val_type)


def calculate_size_scaled_gap_threshold(
    cnv1_length: int,
    cnv2_length: int,
    max_gap_absolute: int,
    gap_scale_fraction: float,
) -> int:
    """
    Calculate size-scaled maximum gap between two CNVs for smoothing.

    This function implements the size-scaled gap rule for CNV smoothing, which allows
    larger CNVs to be merged across larger gaps while preventing inappropriate merging
    of smaller CNVs.

    Formula: min(max_gap_absolute, gap_scale_fraction × max(L1, L2))

    The threshold is based on the larger of the two CNVs to allow merging of CNVs
    of different sizes when the gap is proportional to the larger CNV.

    Parameters
    ----------
    cnv1_length : int
        Length of the first CNV in base pairs
    cnv2_length : int
        Length of the second CNV in base pairs
    max_gap_absolute : int
        Absolute maximum gap allowed regardless of CNV size (cap), in base pairs
    gap_scale_fraction : float
        Fraction of the larger CNV length to use as the gap threshold (e.g., 0.05 for 5%)

    Returns
    -------
    int
        Maximum gap threshold in base pairs

    """
    larger_length = max(cnv1_length, cnv2_length)
    scaled_gap = int(gap_scale_fraction * larger_length)
    return min(max_gap_absolute, scaled_gap)


def _find_candidate_pairs(
    chrom_df: pd.DataFrame, max_gap_absolute: int, gap_scale_fraction: float, *, ignore_sv_type: bool
) -> set[tuple[str, str]]:
    """
    Find candidate pairs within a chromosome using windowed search.

    Parameters
    ----------
    chrom_df : pd.DataFrame
        DataFrame of CNVs on a single chromosome, sorted by position.
        Must have columns: id, pos, end, svlen_int, svtype (optional)
    max_gap_absolute : int
        Absolute maximum gap for merging CNVs (bp)
    gap_scale_fraction : float
        Gap as fraction of larger CNV length
    ignore_sv_type : bool
        If False, only CNVs with the same SVTYPE are considered

    Returns
    -------
    set[tuple[str, str]]
        Set of (cnv1_id, cnv2_id) tuples representing candidate pairs
    """
    pairs = set()
    n = len(chrom_df)

    for i in range(n):
        cnv1 = chrom_df.iloc[i]

        # Define search window
        window_end = cnv1["end"] + max_gap_absolute

        # Get CNVs in window using boolean indexing
        window_df = chrom_df[(chrom_df["pos"] >= cnv1["end"]) & (chrom_df["pos"] <= window_end) & (chrom_df.index > i)]

        for _, cnv2 in window_df.iterrows():
            # SVTYPE filter
            if not ignore_sv_type and cnv1.get("svtype") != cnv2.get("svtype"):
                logger.debug(
                    f"Skipping smoothing: Different SVTYPE {cnv1['id']} ({cnv1.get('svtype')}) "
                    f"vs {cnv2['id']} ({cnv2.get('svtype')})"
                )
                continue

            gap = cnv2["pos"] - cnv1["end"]
            if gap <= 0:
                pairs.add((cnv1["id"], cnv2["id"]))
                logger.debug(f"Smoothing adjacent CNVs: {cnv1['id']} - {cnv2['id']} (gap={gap})")
                continue

            # Size-scaled threshold check
            max_gap = calculate_size_scaled_gap_threshold(
                cnv1["svlen_int"], cnv2["svlen_int"], max_gap_absolute, gap_scale_fraction
            )

            if gap <= max_gap:
                pairs.add((cnv1["id"], cnv2["id"]))
                logger.debug(
                    f"Smoothing candidate: {cnv1['id']} - {cnv2['id']} "
                    f"(gap={gap}bp, threshold={max_gap}bp, CNV_sizes={cnv1['svlen_int']}bp,{cnv2['svlen_int']}bp)"
                )

    return pairs


def identify_smoothing_candidates(
    vcf_path: str,
    max_gap_absolute: int,
    gap_scale_fraction: float,
    *,
    ignore_sv_type: bool,
    ignore_filter: bool,
    cipos_threshold: int,
) -> set[tuple[str, str]]:
    """
    Identify pairs of CNVs that should be merged based on size-scaled gap criteria.

    This function is called AFTER standard distance-based merge AND sorting to identify
    additional CNV pairs that should be merged based on size-scaled gap rules. It uses
    chromosome-level blocking and windowed search to find candidate pairs efficiently.

    IMPORTANT: Assumes vcf_path is already sorted by chromosome and position.

    Smoothing Criteria (all must be satisfied):
    1. Same chromosome
    2. Same SVTYPE (unless ignore_sv_type=True)
    4. CIPOS length >= cipos_threshold for both CNVs (prevents merging high-confidence breakpoints)
    5. Gap between CNVs <= size-scaled threshold
    6. FILTER status compatible (if ignore_filter=False, only PASS CNVs considered)

    Parameters
    ----------
    vcf_path : str
        Path to VCF file (MUST be sorted by chromosome and position)
    max_gap_absolute : int
        Absolute maximum gap for merging CNVs (bp), used as cap in size-scaled formula
    gap_scale_fraction : float
        Gap as fraction of smaller CNV length (e.g., 0.05 for 5%)
    ignore_sv_type : bool
        If False, only CNVs with the same SVTYPE are considered for merging
    ignore_filter : bool
        If False, only PASS CNVs are considered for smoothing.
        Filtered CNVs are excluded from smoothing candidates.
    cipos_threshold : int
        Minimum CIPOS length required for merging (bp). If either CNV has
        CIPOS length < threshold, the pair won't be merged due to high-confidence
        breakpoint. CIPOS length = cipos_max - cipos_min - 1 (e.g., (-50, 50) → 99 bp).

    Returns
    -------
    set[tuple[str, str]]
        Set of (cnv1_id, cnv2_id) tuples representing CNV pairs to merge.
        Empty set if no candidates found.

    Examples
    --------
    >>> # Large CNVs with small gap
    >>> candidates = identify_smoothing_candidates(
    ...     "large_cnvs.vcf.gz",
    ...     max_gap_absolute=50_000,
    ...     gap_scale_fraction=0.05,
    ...     ignore_sv_type=False,
    ...     ignore_filter=True,
    ...     cipos_threshold=50,
    ... )
    >>> # Returns pairs like: {('del1', 'del2'), ('dup3', 'dup4')}

    Notes
    -----
    - The function assumes the input VCF is already sorted. If unsorted, adjacent CNVs
      in the genome may not be adjacent in the iteration, causing missed smoothing.
    - Copy number compatibility is NOT checked - only SVTYPE matters.
    - Uses windowed search for efficiency: only compares CNVs within max possible gap distance.
    """
    # Load VCF as DataFrame (assumes already sorted)
    cnv_df = vcftools.get_vcf_df(vcf_path, custom_info_fields=["CIPOS", "SVLEN", "SVTYPE"])

    # Expand CIPOS tuple into columns for vectorized operations
    cnv_df["cipos_length"] = cnv_df["cipos"].apply(lambda x: x[1] - x[0] - 1)

    # Extract SVLEN (handle tuple format)
    cnv_df["svlen_int"] = cnv_df["svlen"].apply(lambda x: x[0])
    cnv_df["end"] = cnv_df["pos"] + cnv_df["svlen_int"]

    # Filter by FILTER status
    if not ignore_filter:
        pass_mask = pd.isna(cnv_df["filter"]) | cnv_df["filter"].isin(["PASS", ".", ""])
        cnv_df = cnv_df[pass_mask].copy()
        logger.info(f"Smoothing with ignore_filter=False: considering {len(cnv_df)} PASS CNVs")

    # Filter out high-confidence breakpoints
    cnv_df = cnv_df[cnv_df["cipos_length"] >= cipos_threshold].copy()
    logger.info(f"After CIPOS filter (>={cipos_threshold}bp): {len(cnv_df)} CNVs remain")

    # Process by chromosome (natural blocking)
    candidates = set()
    for _, chrom_df in cnv_df.groupby("chrom", sort=False):
        chrom_df_sorted = chrom_df.sort_values("pos").reset_index(drop=True)
        pairs = _find_candidate_pairs(
            chrom_df_sorted, max_gap_absolute, gap_scale_fraction, ignore_sv_type=ignore_sv_type
        )
        candidates.update(pairs)

    logger.info(f"Identified {len(candidates)} smoothing candidate pairs")
    return candidates


def _group_candidates_transitively(candidates: set[tuple[str, str]]) -> list[set[str]]:
    """
    Group candidate pairs into connected components using scipy graph algorithms.

    If (A, B) and (B, C) are candidates, they form a connected component {A, B, C}
    that should be merged as a single group. This function identifies all such
    transitive relationships using scipy's optimized connected components algorithm.

    Parameters
    ----------
    candidates : set[tuple[str, str]]
        Set of CNV ID pairs that are candidates for merging

    Returns
    -------
    list[set[str]]
        List of groups, where each group is a set of CNV IDs to merge together.
        Empty list if no candidates provided.

    """
    if not candidates:
        return []

    # Extract unique IDs and create index mapping
    all_ids = sorted({cnv_id for pair in candidates for cnv_id in pair})
    id_to_idx = {cnv_id: i for i, cnv_id in enumerate(all_ids)}

    # Build sparse adjacency matrix (undirected graph)
    rows, cols = [], []
    for a, b in candidates:
        i, j = id_to_idx[a], id_to_idx[b]
        rows.extend([i, j])
        cols.extend([j, i])

    adjacency = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(all_ids), len(all_ids)))

    # Find connected components using scipy's optimized algorithm
    n_components, labels = connected_components(adjacency, directed=False)

    # Group IDs by component label
    groups = [set() for _ in range(n_components)]
    for cnv_id, label in zip(all_ids, labels, strict=True):
        groups[label].add(cnv_id)

    logger.info(f"Grouped {len(candidates)} candidate pairs into {n_components} transitive groups")
    return groups


def _apply_smoothing_merges(
    input_vcf: str,
    output_vcf: str,
    smoothing_groups: list[set[str]],
    aggregation_actions: dict[str, list[str]],
    *,
    ignore_filter: bool,
) -> None:
    """
    Merge CNVs in each smoothing group and write to output VCF.

    For each smoothing group:
    - Load all CNV records in the group
    - Merge boundaries (min start, max end)
    - Aggregate INFO fields using aggregation_actions
    - Write merged record with the ID of the first CNV in the group

    CNVs not in any group are written unchanged to preserve all variants.

    Parameters
    ----------
    input_vcf : str
        Path to input VCF file (post-Stage 1 collapse, sorted)
    output_vcf : str
        Path to output VCF file where smoothed variants will be written
    smoothing_groups : list[set[str]]
        List of CNV ID groups to merge, each group is a set of CNV IDs
    aggregation_actions : dict[str, list[str]]
        Dictionary mapping aggregation action to list of INFO field names
        (e.g., {"weighted_avg": ["CopyNumber"], "max": ["TREE_SCORE"]})
    ignore_filter : bool
        Should the PASS filter be ignored (otherwise only PASS variants will be output)

    Notes
    -----
    - Uses existing aggregation logic from CNV_AGGREGATION_ACTIONS
    - Preserves VCF header from input
    - CNVs are merged in the order they appear in the input VCF
    - The first CNV ID in each group (by file order) is used for the merged record ID
    """
    # Create ID-to-group mapping for fast lookup
    id_to_group = {}
    for group in smoothing_groups:
        for cnv_id in group:
            id_to_group[cnv_id] = group

    # Track which groups have been processed
    processed_groups = set()

    with pysam.VariantFile(input_vcf) as vcf_in:
        header = vcf_in.header

        with pysam.VariantFile(output_vcf, "w", header=header) as vcf_out:
            # Collect all records by ID for grouping
            all_records = {}
            for record in vcf_in:
                # skip non-PASS records
                if not ignore_filter and not vcftools.is_pass_record(record):
                    vcf_out.write(record)
                    continue
                all_records[record.id] = record

            # Process records in input order
            for record_id, record in all_records.items():
                # Check if this record is part of a smoothing group
                if record_id in id_to_group:
                    group = id_to_group[record_id]
                    group_key = frozenset(group)

                    # Only process each group once (first CNV in group wins)
                    if group_key in processed_groups:
                        continue
                    processed_groups.add(group_key)

                    # Collect all records in the group
                    group_records = [all_records[cnv_id] for cnv_id in sorted(group) if cnv_id in all_records]

                    if len(group_records) == 1:
                        # Single CNV in group - write unchanged
                        vcf_out.write(group_records[0])
                        continue

                    # Merge the group
                    merged_record = _merge_cnv_group(group_records, header, aggregation_actions)
                    vcf_out.write(merged_record)

                else:
                    # Not in any group - write unchanged
                    vcf_out.write(record)

    total_cnvs_merged = sum(len(group) for group in smoothing_groups)
    logger.info(
        f"Applied smoothing: merged {total_cnvs_merged} CNVs into {len(smoothing_groups)} groups "
        f"(net reduction: {total_cnvs_merged - len(smoothing_groups)} CNVs)"
    )


def _merge_cnv_group(
    records: list[pysam.VariantRecord],
    header: pysam.VariantHeader,
    aggregation_actions: dict[str, list[str]],
) -> pysam.VariantRecord:
    """
    Merge a group of CNV records into a single record.

    Parameters
    ----------
    records : list[pysam.VariantRecord]
        List of CNV records to merge (sorted by position)
    header : pysam.VariantHeader
        VCF header for creating new record
    aggregation_actions : dict[str, list[str]]
        Dictionary mapping aggregation action to list of INFO field names

    Returns
    -------
    pysam.VariantRecord
        Merged CNV record with aggregated INFO fields

    """
    # Use first record as template
    merged = records[0].copy()

    # Create DataFrame using get_vcf_df with tuple input (header, records)
    # This ensures exact format consistency with _prepare_update_dataframe
    all_fields = sum(aggregation_actions.values(), [])
    df_all = vcftools.get_vcf_df((header, records), custom_info_fields=all_fields + ["SVLEN"])

    # Calculate 'end' column (same logic as _prepare_update_dataframe)
    df_all["end"] = df_all["pos"] + df_all["svlen"].apply(lambda x: x[0])

    # Reuse the same aggregation logic as distance-based collapse
    _aggregate_record_info_fields(merged, df_all, aggregation_actions, header)

    return merged


def _postprocess_collapsed_vcf(  # noqa: C901, PLR0912, PLR0915
    collapsed_vcf: str,
    removed_vcf: str,
    output_vcf: str,
) -> None:
    """
    Post-process collapsed VCF to merge CNV_SOURCE from removed variants and add CNV_ID.

    This function:
    1. Reads removed variants into a list (indexed by position)
    2. For each variant in collapsed VCF that has NumCollapsed > 0:
       - Finds overlapping removed variants
       - Merges CNV_SOURCE from removed variants
       - Adds CNV_ID with IDs of removed variants
       - Removes truvari tags (NumCollapsed, NumConsolidated, CollapseId)

    Parameters
    ----------
    collapsed_vcf : str
        Path to collapsed VCF from truvari
    removed_vcf : str
        Path to removed variants VCF from truvari
    output_vcf : str
        Path to output post-processed VCF
    """
    # Read all removed variants into a list for overlap detection
    removed_variants = []
    with pysam.VariantFile(removed_vcf) as vcf_in:
        for record in vcf_in:
            removed_variants.append(
                {
                    "chrom": record.chrom,
                    "start": record.start,
                    "stop": record.stop,
                    "id": record.id,
                    "cnv_source": record.info.get("CNV_SOURCE", []),
                    "svtype": record.info.get("SVTYPE", ""),
                }
            )
    logger.info(f"Read {len(removed_variants)} removed variants")

    # Process collapsed VCF
    collapsed_count = 0
    num_collapsed_count = 0
    updated_count = 0

    # First pass: collect records to update
    records_to_process = []
    with pysam.VariantFile(collapsed_vcf) as vcf_in:
        header = vcf_in.header.copy()

        # Add CNV_ID tag if not present
        if "CNV_ID" not in header.info:
            header.info.add(*INFO_TAG_REGISTRY["CNV_ID"][:-1])

        # Read all records
        for record in vcf_in:
            records_to_process.append(record)

    # Second pass: write updated records
    with pysam.VariantFile(output_vcf, "w", header=header) as vcf_out:
        for old_record in records_to_process:
            collapsed_count += 1

            # Create new record with updated header
            record = vcf_out.new_record()
            record.chrom = old_record.chrom
            record.pos = old_record.pos
            record.id = old_record.id
            record.ref = old_record.ref
            record.alts = old_record.alts
            record.qual = old_record.qual
            if old_record.filter:
                record.filter.add(*old_record.filter)
            # Copy all INFO fields from old record
            for key in old_record.info:
                record.info[key] = old_record.info[key]
            # Explicitly set stop from old_record to preserve END coordinate
            record.stop = old_record.stop
            # Copy samples
            for sample in old_record.samples:
                for key in old_record.samples[sample]:
                    record.samples[sample][key] = old_record.samples[sample][key]

            # Check if this variant has NumCollapsed > 0 (meaning it replaced others)
            num_collapsed = record.info.get("NumCollapsed", 0)
            if num_collapsed > 0:
                num_collapsed_count += 1

                # Find overlapping removed variants (same chrom, overlap in position, same SVTYPE)
                record_chrom = record.chrom
                record_start = record.start
                record_stop = record.stop
                record_svtype = record.info.get("SVTYPE", "")

                overlapping = []
                for removed in removed_variants:
                    if (
                        removed["chrom"] == record_chrom
                        and removed["svtype"] == record_svtype
                        and removed["start"] < record_stop
                        and removed["stop"] > record_start
                    ):
                        overlapping.append(removed)

                if overlapping:
                    updated_count += 1

                    # Collect CNV_SOURCE values
                    all_sources = set()
                    current_sources = record.info.get("CNV_SOURCE", [])
                    if isinstance(current_sources, str):
                        all_sources.add(current_sources)
                    else:
                        all_sources.update(current_sources)

                    for removed in overlapping:
                        removed_sources = removed["cnv_source"]
                        if isinstance(removed_sources, str):
                            all_sources.add(removed_sources)
                        else:
                            all_sources.update(removed_sources)

                    # Update CNV_SOURCE
                    record.info["CNV_SOURCE"] = sorted(all_sources)

                    # Add CNV_ID with removed variant IDs
                    removed_ids = [v["id"] for v in overlapping if v["id"]]
                    if removed_ids:
                        record.info["CNV_ID"] = removed_ids

            # Remove truvari tags
            for tag in ["NumCollapsed", "NumConsolidated", "CollapseId"]:
                if tag in record.info:
                    del record.info[tag]

            vcf_out.write(record)

    logger.info(
        f"Post-processing complete: {collapsed_count} total variants, "
        f"{num_collapsed_count} with NumCollapsed>0, "
        f"{updated_count} updated with CNV_ID"
    )


def merge_cnv_sv_vcfs(  # noqa: PLR0915
    cnv_vcf: str,
    sv_vcf: str,
    output_vcf: str,
    min_sv_length: int = 1000,
    max_sv_length: int | None = 5000000,
    min_sv_qual: float = 0,
    distance: int = 0,
    pctsize: float = 0.5,
    output_directory: str | None = None,
) -> str:
    """
    Merge CNV VCF with filtered SV VCF, replacing overlapping CNVs with SV calls.

    This function implements a 3-stage pipeline:
    1. Filter SV VCF for PASS DEL/DUP calls with min_sv_length <= |SVLEN| <= max_sv_length
    2. Combine VCFs with source annotation
    3. Sort and collapse overlaps using pick_best=True (SV wins due to higher QUAL)

    The key insight: SV calls from tools like GRIDSS have QUAL scores 10-100x higher
    than CNV calls from CNVpytor/cn.mops. Using collapse_vcf with pick_best=True
    automatically selects the SV call in overlapping regions, effectively "replacing"
    the lower-quality CNV call.

    Parameters
    ----------
    cnv_vcf : str
        Path to CNV VCF file (from cn.mops, CNVpytor, or combined)
    sv_vcf : str
        Path to SV VCF file (e.g., from GRIDSS)
    output_vcf : str
        Path for output merged VCF file
    min_sv_length : int, optional
        Minimum absolute SVLEN for SV calls to include, by default 1000
    max_sv_length : int, optional
        Maximum absolute SVLEN for SV calls to include, by default 5000000 (5Mb).
        Useful for excluding very large SVs that may have low specificity.
    min_sv_qual : float, optional
        Minimum QUAL score for SV calls to include, by default 0 (no minimum).
        Useful for filtering low-quality SV calls.
    distance : int, optional
        Distance threshold for collapsing overlapping variants, by default 0 (exact overlaps only)
    pctsize : float, optional
        Minimum size similarity (0.0-1.0) required for collapsing, by default 0.5 (50% match).
        Prevents giant SVs (e.g., 68Mb) from replacing many small CNVs they happen to overlap.
    output_directory : str, optional
        Directory for temporary files. If None, uses output_vcf directory.
        If provided, must already exist.

    Returns
    -------
    str
        Path to final merged VCF file

    Raises
    ------
    FileNotFoundError
        If input VCF files do not exist
    ValueError
        If output_directory is provided but does not exist
    RuntimeError
        If VCF processing fails

    Examples
    --------
    >>> merge_cnv_sv_vcfs(
    ...     cnv_vcf="combined_cnv.vcf.gz",
    ...     sv_vcf="gridss_output.vcf.gz",
    ...     output_vcf="merged_cnv_sv.vcf.gz",
    ...     fasta_index="hg38.fa.fai",
    ...     min_sv_length=1000,
    ...     distance=0
    ... )

    Notes
    -----
    - Only PASS variants from SV VCF are included
    - Only DEL and DUP SVTYPE values are selected from SV VCF
    - SV calls automatically replace overlapping CNV calls due to higher QUAL scores
    - Non-overlapping CNV calls are preserved in the output
    - All INFO fields from both VCFs are preserved in the merged header
    """
    # Set output directory (require it exists if provided)
    if output_directory is None:
        output_directory = os.path.dirname(os.path.abspath(output_vcf))
        os.makedirs(output_directory, exist_ok=True)
    elif not os.path.isdir(output_directory):
        # User provided output_directory - require it exists
        raise ValueError(f"Output directory does not exist: {output_directory}")

    logger.info(f"Starting CNV+SV merge: CNV={cnv_vcf}, SV={sv_vcf}")
    max_len_msg = f" to {max_sv_length}bp" if max_sv_length else ""
    qual_msg = f", Min QUAL: {min_sv_qual}" if min_sv_qual > 0 else ""
    logger.info(f"Length threshold: {min_sv_length}bp{max_len_msg}{qual_msg}, Distance: {distance}bp")

    vcf_utils = VcfUtils()
    temporary_files = []

    # Stage 1: Filter SV VCF for PASS DEL/DUP with length and quality constraints
    logger.info("Stage 1: Filtering SV VCF for PASS DEL/DUP calls")
    filtered_sv_vcf = pjoin(output_directory, "filtered_sv.vcf.gz")
    filter_expr = f'(SVTYPE="DEL" | SVTYPE="DUP") & abs(SVLEN) >= {min_sv_length}'
    if max_sv_length is not None:
        filter_expr += f" & abs(SVLEN) <= {max_sv_length}"
    if min_sv_qual > 0:
        filter_expr += f" & QUAL >= {min_sv_qual}"
    vcf_utils.view_vcf(
        input_vcf=sv_vcf,
        output_vcf=filtered_sv_vcf,
        extra_args=f"-f PASS -i '{filter_expr}'",
    )
    vcf_utils.index_vcf(filtered_sv_vcf)
    temporary_files.append(filtered_sv_vcf)

    # Log filtered SV count
    with pysam.VariantFile(filtered_sv_vcf) as vcf_in:
        sv_count = sum(1 for _ in vcf_in)
    length_range = f">= {min_sv_length}bp"
    if max_sv_length is not None:
        length_range = f"between {min_sv_length}bp and {max_sv_length}bp"
    qual_suffix = f" with QUAL >= {min_sv_qual}" if min_sv_qual > 0 else ""
    logger.info(f"Filtered SV VCF contains {sv_count} PASS DEL/DUP calls {length_range}{qual_suffix}")

    # Stage 2: Combine VCF headers
    logger.info("Stage 2: Combining VCF headers")
    vcf_cnv = pysam.VariantFile(cnv_vcf)
    vcf_sv = pysam.VariantFile(filtered_sv_vcf)

    # Merge headers with keep_filters=True to preserve GRIDSS filter definitions
    combined_header = combine_vcf_headers_for_cnv(
        vcf_cnv.header,
        vcf_sv.header,
        keep_filters=True,
    )

    # Ensure CNV_SOURCE tag exists
    if "CNV_SOURCE" not in combined_header.info:
        combined_header.info.add(*INFO_TAG_REGISTRY["CNV_SOURCE"][:-1])

    # Write combined records with source annotation
    logger.info("Writing combined VCF with source annotations")
    temp_combined_vcf = pjoin(output_directory, "temp_combined.vcf.gz")
    with pysam.VariantFile(temp_combined_vcf, "w", header=combined_header) as vcf_out:
        # Write SV records with "gridss" source
        seen_ids = write_vcf_records_with_source(
            vcf_sv, vcf_out, combined_header, "gridss", make_ids_unique=True, seen_ids=set()
        )

        # Extract existing CNV_SOURCE from CNV VCF (cn.mops, cnvpytor, or combined)
        vcf_cnv.close()
        vcf_cnv = pysam.VariantFile(cnv_vcf)  # Reopen after iteration
        cnv_source = "unknown"
        for record in vcf_cnv:
            if "CNV_SOURCE" in record.info:
                cnv_source = record.info["CNV_SOURCE"][0]
            break

        # Write CNV records with original source
        vcf_cnv.close()
        vcf_cnv = pysam.VariantFile(cnv_vcf)
        write_vcf_records_with_source(
            vcf_cnv, vcf_out, combined_header, cnv_source, make_ids_unique=True, seen_ids=seen_ids
        )

    vcf_cnv.close()
    vcf_sv.close()
    temporary_files.append(temp_combined_vcf)

    # Stage 3: Sort combined VCF
    logger.info("Stage 3: Sorting combined VCF")
    temp_sorted_vcf = pjoin(output_directory, "temp_sorted.vcf.gz")
    vcf_utils.sort_vcf(temp_combined_vcf, temp_sorted_vcf)
    vcf_utils.index_vcf(temp_sorted_vcf)
    temporary_files.append(temp_sorted_vcf)

    # Stage 4: Collapse overlaps with pick_best=True and size filtering
    logger.info("Stage 4: Collapsing overlapping variants (SV replaces CNV via QUAL)")
    logger.info(f"Collapse parameters: refdist={distance}bp, pctsize={pctsize * 100}%")

    # Call collapse_vcf directly for better control over parameters
    collapsed_vcf_tmp = pjoin(output_directory, "collapsed_tmp.vcf.gz")
    removed_vcf = vcf_utils.collapse_vcf(
        vcf=temp_sorted_vcf,
        output_vcf=collapsed_vcf_tmp,
        refdist=distance,
        pctseq=0.0,  # Don't require sequence similarity
        pctsize=pctsize,  # Require size similarity (default 50%)
        maxsize=-1,  # No size limit
        ignore_filter=False,  # Only PASS variants
        ignore_sv_type=False,  # Only merge same SVTYPE (DEL with DEL, DUP with DUP)
        pick_best=True,  # Choose by QUAL → SV wins
        erase_removed=False,  # Keep removed VCF for post-processing
    )
    temporary_files.extend([collapsed_vcf_tmp, removed_vcf])

    # Post-process: merge CNV_SOURCE from removed variants and add CNV_ID
    logger.info("Post-processing: merging CNV_SOURCE and adding CNV_ID from removed variants")
    postprocessed_vcf = pjoin(output_directory, "postprocessed.vcf.gz")
    _postprocess_collapsed_vcf(collapsed_vcf_tmp, removed_vcf, postprocessed_vcf)
    temporary_files.append(postprocessed_vcf)

    # Sort final output
    vcf_utils.sort_vcf(postprocessed_vcf, output_vcf)
    vcf_utils.index_vcf(output_vcf)

    # Cleanup temporary files
    mu.cleanup_temp_files(temporary_files)
    logger.info(f"Successfully created merged CNV+SV VCF: {output_vcf}")

    return output_vcf
