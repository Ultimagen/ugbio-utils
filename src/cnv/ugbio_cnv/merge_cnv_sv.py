"""Merge CNV and SV VCF files: combine them (headers, source tags), then collapse overlapping calls."""

import argparse
import os
import sys
from os.path import join as pjoin
from pathlib import Path

import pysam
from ugbio_cnv.cnv_vcf_consts import INFO_TAG_REGISTRY
from ugbio_core import misc_utils as mu
from ugbio_core.logger import logger
from ugbio_core.vcf_utils import VcfUtils
from ugbio_core.vcfbed import vcftools


# ---------------------------------------------------------------- combining the VCFs
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
    source_name: str | None,
    *,
    make_ids_unique: bool = False,
    seen_ids: set | None = None,
    clear_filters: bool = True,
) -> set[str]:
    """
    Write VCF records to output file with CNV_SOURCE annotation.

    Parameters
    ----------
    vcf_in : pysam.VariantFile
        Input VCF file to read records from
    vcf_out : pysam.VariantFile
        Output VCF file to write records to
    combined_header : pysam.VariantHeader
        Combined header for creating new records
    source_name : str | None
        Source name to add to CNV_SOURCE INFO field. If None, preserves existing
        CNV_SOURCE values from the input records without modification.
    make_ids_unique : bool, optional
        If True, ensure all IDs are unique by appending suffixes (default: False)
    seen_ids : set, optional
        Set of already-seen IDs for tracking uniqueness (default: None, creates new set)
    clear_filters : bool, optional
        If True, clear all FILTER values and set to PASS (default: True for backward compatibility)

    Returns
    -------
    set
        The updated set of seen IDs
    """
    source_desc = source_name if source_name is not None else "VCF (preserving existing CNV_SOURCE)"
    logger.info(f"Writing records from {source_desc}")
    if seen_ids is None:
        seen_ids = set()

    for record in vcf_in:
        if clear_filters:
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

        # Add source tag if source_name is provided and tag not already present
        if source_name is not None and "CNV_SOURCE" not in new_record.info:
            new_record.info["CNV_SOURCE"] = (source_name,)
        # If source_name is None, preserve existing CNV_SOURCE value from record

        vcf_out.write(new_record)

    return seen_ids


def combine_vcf_headers_for_cnv(
    header1: pysam.VariantHeader, header2: pysam.VariantHeader, *, keep_filters: bool = False
) -> pysam.VariantHeader:
    """
    Combine two VCF headers into a single header for CNV/SV variant calls.

    This function is specifically designed for merging headers from different CNV/SV
    callers (e.g., the depth CNV caller and GRIDSS). It creates a new header that includes all
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
    >>> header2 = pysam.VariantFile("gridss.vcf.gz").header
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


def combine_cnv_or_sv_vcf_files(
    vcf_files: list[tuple[str, str | None]],
    output_vcf: str,
    fasta_index: str,
    *,
    preserve_filters: bool = False,
    make_ids_unique: bool = False,
    output_directory: str | None = None,
) -> str:
    """
    Combine multiple CNV or SV VCF files with source annotations into a single sorted VCF.

    This is a general-purpose VCF combination function that:
    - Updates all VCF headers to match the FASTA index contigs
    - Combines headers from all input VCFs
    - Annotates each variant with its source caller via CNV_SOURCE tag
    - Optionally preserves or clears FILTER fields
    - Ensures unique variant IDs across all files if requested
    - Produces a sorted and indexed output VCF

    Parameters
    ----------
    vcf_files : list[tuple[str, str | None]]
        List of (vcf_path, source_name) tuples. If source_name is not None, it will be
        added to the CNV_SOURCE INFO field for all variants from that VCF. If source_name
        is None, existing CNV_SOURCE values from the input records are preserved without
        modification.
        Example: [("cnv.vcf.gz", "cn.mops"), ("sv.vcf.gz", "gridss"), ("merged.vcf.gz", None)]
    output_vcf : str
        Path to output combined VCF file (.vcf.gz)
    fasta_index : str
        Path to reference genome FASTA index (.fai). All VCF headers will be
        updated to match the contigs from this index to ensure consistency
        across samples and prevent downstream processing errors.
    preserve_filters : bool, optional
        If True, preserve FILTER field definitions in the combined header and
        maintain original FILTER values in all variant records.
        If False, exclude FILTER definitions from the combined header and reset
        all variant FILTER values to PASS. (default: False)
    make_ids_unique : bool, optional
        If True, ensure all variant IDs are unique across all input VCFs by
        appending numeric suffixes (e.g., ID_1, ID_2) when duplicates are found.
        (default: False)
    output_directory : str, optional
        Directory for storing temporary files during processing. If None, uses
        the directory of output_vcf. (default: None)

    Returns
    -------
    str
        Path to the final sorted and indexed combined VCF file

    Raises
    ------
    ValueError
        If vcf_files is empty
    FileNotFoundError
        If any input VCF or the FASTA index file does not exist
    RuntimeError
        If VCF headers have incompatible INFO/FORMAT field definitions

    Examples
    --------
    Combine SV calls with CNV calls that keep their own CNV_SOURCE, as merge_cnv_sv does (preserve filters):

    >>> combine_cnv_or_sv_vcf_files(
    ...     vcf_files=[("sv.vcf.gz", "gridss"), ("cnv.vcf.gz", None)],
    ...     output_vcf="combined.vcf.gz",
    ...     fasta_index="genome.fa.fai",
    ...     preserve_filters=True,
    ... )

    Consolidate the calls of two CNV callers, each tagged with its source (clear filters to PASS):

    >>> combine_cnv_or_sv_vcf_files(
    ...     vcf_files=[
    ...         ("sample1_depth.vcf.gz", "depth"),
    ...         ("sample1_cnmops.vcf.gz", "cn.mops"),
    ...     ],
    ...     output_vcf="combined_cnv.vcf.gz",
    ...     fasta_index="genome.fa.fai",
    ...     preserve_filters=False,
    ...     make_ids_unique=True,
    ... )

    See Also
    --------
    combine_vcf_headers_for_cnv : Combines VCF headers with configurable filter handling
    write_vcf_records_with_source : Writes VCF records with source annotation
    update_vcf_contig : Updates VCF header to match FASTA index contigs
    """
    if not vcf_files:
        raise ValueError("vcf_files must contain at least one VCF")

    # Setup output directory
    if output_directory is None:
        output_directory = os.path.dirname(os.path.abspath(output_vcf))
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    vcf_utils = VcfUtils()
    temporary_files = []

    # Step 1: Update headers to match FASTA index contigs
    logger.info("Updating VCF headers to match FASTA index contigs")
    updated_vcf_files = []
    for idx, (vcf_path, source_name) in enumerate(vcf_files):
        updated_vcf = update_vcf_contig(vcf_utils, vcf_path, fasta_index, output_directory, index=idx)
        temporary_files.append(updated_vcf)
        updated_vcf_files.append((updated_vcf, source_name))
    vcf_files = updated_vcf_files

    # Step 2: Open VCF files and combine headers
    logger.info("Combining VCF headers")
    vcf_handles = [pysam.VariantFile(vcf_path) for vcf_path, _ in vcf_files]

    try:
        # Combine all headers
        combined_header = vcf_handles[0].header.copy()
        for vcf_handle in vcf_handles[1:]:
            combined_header = combine_vcf_headers_for_cnv(
                combined_header, vcf_handle.header, keep_filters=preserve_filters
            )

        # Add CNV_SOURCE INFO tag if not present
        if "CNV_SOURCE" not in combined_header.info:
            combined_header.info.add(*INFO_TAG_REGISTRY["CNV_SOURCE"][:-1])

        # Step 3: Write records from all VCFs with source annotations
        logger.info(f"Writing records from {len(vcf_handles)} VCF files")
        temp_combined_vcf = pjoin(output_directory, "temp_combined.vcf.gz")
        temporary_files.append(temp_combined_vcf)

        with pysam.VariantFile(temp_combined_vcf, "w", header=combined_header) as vcf_out:
            seen_ids = set()
            for vcf_handle, (_, source_name) in zip(vcf_handles, vcf_files, strict=False):
                seen_ids = write_vcf_records_with_source(
                    vcf_handle,
                    vcf_out,
                    combined_header,
                    source_name,
                    make_ids_unique=make_ids_unique,
                    seen_ids=seen_ids,
                    clear_filters=not preserve_filters,  # Inverse of preserve
                )
    finally:
        for vcf_handle in vcf_handles:
            vcf_handle.close()

    # Step 4: Sort and index
    logger.info("Sorting and indexing combined VCF")
    vcf_utils.sort_vcf(temp_combined_vcf, output_vcf)
    vcf_utils.index_vcf(output_vcf)

    # Cleanup temporary files
    mu.cleanup_temp_files(temporary_files)

    logger.info(f"Successfully created combined VCF: {output_vcf}")
    return output_vcf


# ---------------------------------------------------------------- collapsing the combined calls
def _flatten_cnv_sources(sources_series):
    """
    Flatten and deduplicate CNV_SOURCE values from multiple variants.

    CNV_SOURCE can be a string, list, or tuple. This function:
    1. Converts all to lists
    2. Flattens into a single list
    3. Removes duplicates and sorts

    Example: ["depth", ["split_read", "gridss"]] -> ["depth", "gridss", "split_read"]
    """
    all_sources = []
    for source in sources_series:
        if not source:  # Skip None/empty
            continue
        if isinstance(source, list | tuple):
            all_sources.extend(source)
        else:
            all_sources.append(source)
    return sorted(set(all_sources))


def _collect_variant_ids(ids_series):
    """
    Collect non-empty variant IDs.

    Filters out None, empty strings, and "." (VCF's missing value indicator).
    """
    return [variant_id for variant_id in ids_series if variant_id and variant_id != "."]


def _aggregate_removed_variants(removed_vcf: str) -> dict[float, dict]:
    """
    Load removed variants and aggregate their metadata by CollapseId.

    When truvari collapses overlapping variants:
    - Kept variants get a CollapseId field
    - Removed variants get a MatchId field pointing to the CollapseId

    This function groups removed variants by MatchId and aggregates:
    - CNV_SOURCE: All source tools (depth, split_read, pair, cn.mops, gridss)
    - IDs: All variant IDs that were removed

    Parameters
    ----------
    removed_vcf : str
        Path to VCF file containing removed/merged variants

    Returns
    -------
    dict[float, dict]
        Dictionary mapping CollapseId -> {cnv_source: [...], id: [...]}
        Example: {1.0: {"cnv_source": ["depth", "gridss"], "id": ["CNV_1", "CNV_2"]}}
    """
    # Load removed variants with CNV_SOURCE and MatchId fields
    removed_df = vcftools.get_vcf_df(removed_vcf, custom_info_fields=["CNV_SOURCE", "MatchId"])
    logger.info(f"Read {len(removed_df)} removed variants")

    if removed_df.empty or "matchid" not in removed_df.columns:
        return {}

    # Extract MatchId value (stored as tuple in VCF) and convert to float
    removed_df["matchid"] = removed_df["matchid"].apply(lambda x: x[0] if x else None).astype(float)

    # Group by MatchId (=CollapseId) and aggregate CNV sources and IDs
    aggregated = removed_df.groupby("matchid").agg(
        {
            "cnv_source": _flatten_cnv_sources,
            "id": _collect_variant_ids,
        }
    )

    # Convert to dict for O(1) lookup: {CollapseId: {cnv_source: [...], id: [...]}}
    return aggregated.to_dict(orient="index")


def _merge_cnv_metadata(record: pysam.VariantRecord, agg_data: dict) -> None:
    """
    Merge CNV_SOURCE and CNV_ID from aggregated data into a record.

    Parameters
    ----------
    record : pysam.VariantRecord
        The record to update
    agg_data : dict
        Aggregated data containing cnv_source and id lists
    """
    # Merge CNV_SOURCE from removed variants
    all_sources = set()
    current_sources = record.info.get("CNV_SOURCE", [])
    if isinstance(current_sources, str):
        all_sources.add(current_sources)
    else:
        all_sources.update(current_sources)

    if agg_data.get("cnv_source"):
        all_sources.update(agg_data["cnv_source"])

    record.info["CNV_SOURCE"] = sorted(all_sources)

    # Add CNV_ID from removed variants
    removed_ids = agg_data.get("id", [])
    if removed_ids:
        record.info["CNV_ID"] = removed_ids


def _postprocess_collapsed_vcf(
    collapsed_vcf: str,
    removed_vcf: str,
    output_vcf: str,
) -> None:
    """Merge CNV source metadata from removed variants into collapsed variants."""
    aggregated = _aggregate_removed_variants(removed_vcf)

    collapsed_count = 0
    updated_count = 0

    with pysam.VariantFile(collapsed_vcf) as vcf_in:
        # Add CNV_ID to input header so records can use it
        if "CNV_ID" not in vcf_in.header.info:
            vcf_in.header.info.add(*INFO_TAG_REGISTRY["CNV_ID"][:-1])

        with pysam.VariantFile(output_vcf + ".tmp.vcf.gz", "w", header=vcf_in.header) as vcf_out:
            for record in vcf_in:
                collapsed_count += 1

                # Check if this record has CollapseId (indicating variants were merged)
                if "CollapseId" in record.info and (agg_data := aggregated.get(float(record.info["CollapseId"]))):
                    updated_count += 1
                    _merge_cnv_metadata(record, agg_data)

                # Remove temporary collapse metadata tags
                for tag in ["NumCollapsed", "NumConsolidated", "CollapseId"]:
                    if tag in record.info:
                        del record.info[tag]

                vcf_out.write(record)
    VcfUtils().sort_vcf(output_vcf + ".tmp.vcf.gz", output_vcf)
    os.unlink(output_vcf + ".tmp.vcf.gz")
    logger.info(
        f"Post-processing complete: {collapsed_count} total variants, " f"{updated_count} updated with CNV metadata"
    )


def _filter_large_svs_without_cnv_overlap(
    input_vcf: str,
    output_vcf: str,
    max_sv_length: int,
) -> tuple[int, int]:
    """
    Filter out large SVs that don't have CNV overlap.

    SVs with abs(SVLEN) > max_sv_length are only kept if they have CNV_SOURCE
    indicating overlap with CNV calls (i.e., CNV_SOURCE contains sources other than "gridss").

    Parameters
    ----------
    input_vcf : str
        Input VCF file path
    output_vcf : str
        Output VCF file path
    max_sv_length : int
        Maximum SV length threshold

    Returns
    -------
    tuple[int, int]
        (total_variants, filtered_variants)
    """
    total = 0
    filtered = 0

    with pysam.VariantFile(input_vcf) as vcf_in:
        header = vcf_in.header.copy()

        with pysam.VariantFile(output_vcf, "w", header=header) as vcf_out:
            for record in vcf_in:
                total += 1
                # SVLEN is a tuple in VCF INFO fields
                svlen_raw = record.info.get("SVLEN", (0,))
                svlen = abs(svlen_raw[0] if isinstance(svlen_raw, tuple) else svlen_raw)

                # Keep all variants <= max_sv_length
                if svlen <= max_sv_length:
                    vcf_out.write(record)
                    continue

                # For large variants, check CNV overlap via CNV_SOURCE
                cnv_sources = record.info.get("CNV_SOURCE", [])
                if isinstance(cnv_sources, str):
                    cnv_sources = [cnv_sources]

                # Keep if merged with CNV (has non-gridss sources)
                has_cnv_source = any(src != "gridss" for src in cnv_sources)
                if has_cnv_source:
                    vcf_out.write(record)
                else:
                    filtered += 1

    return total, filtered


def merge_cnv_sv_vcfs(
    cnv_vcf: str,
    sv_vcf: str,
    output_vcf: str,
    fasta_index: str,
    min_sv_length: int = 500,
    max_sv_length: int | None = 5000000,
    min_sv_qual: float = 0,
    distance: int = 0,
    pctsize: float = 0.5,
) -> str:
    """
    Merge CNV and SV VCF files, preferring higher-quality SV calls on overlap.

    Only PASS CNVs and PASS DEL/DUP SVs above thresholds participate in the merge/collapse operation.
    All other variants are preserved in the final output with their original FILTER values:

    SVs excluded from merge:
    - Non-DEL/DUP types (INS, CTX, BND, ...)
    - Non-PASS DEL/DUP (failed filters in original VCF)
    - PASS DEL/DUP below size threshold (too small for merge)

    CNVs excluded from merge:
    - Non-PASS CNVs (failed filters in original VCF)

    Large SVs (>max_sv_length) are only retained if they overlap with CNV calls
    by at least pctsize; otherwise they are filtered out.

    Parameters
    ----------
    cnv_vcf : str
        Path to input CNV VCF file
    sv_vcf : str
        Path to input SV VCF file (e.g., from GRIDSS)
    output_vcf : str
        Path to output merged VCF file
    fasta_index : str
        Path to reference genome FASTA index (.fai). VCF headers will be updated
        to match contigs from this index to ensure consistency.
    min_sv_length : int, optional
        Minimum absolute SVLEN for SV DEL / DUP calls to merge with the CNV calls (default: 500)
    max_sv_length : int | None, optional
        Maximum absolute SVLEN for SV calls. SVs larger than this are only kept if
        they overlap CNVs by at least pctsize (default: 5000000)
    min_sv_qual : float, optional
        Minimum QUAL score for SV calls (default: 0, no minimum)
    distance : int, optional
        Distance threshold for collapsing overlapping variants (default: 0, exact overlaps)
    pctsize : float, optional
        Minimum overlap fraction for collapsing variants (default: 0.5, 50%)

    Returns
    -------
    str
        Path to the output merged VCF file
    """

    output_directory = os.path.dirname(os.path.abspath(output_vcf))

    logger.info(f"Starting CNV+SV merge: CNV={cnv_vcf}, SV={sv_vcf}")
    max_len_msg = (
        f" (large SVs >{max_sv_length}bp kept only if at least {pctsize} overlapping CNV)" if max_sv_length else ""
    )
    qual_msg = f", Min QUAL: {min_sv_qual}" if min_sv_qual > 0 else ""
    length_range = f">= {min_sv_length}bp"

    vcf_utils = VcfUtils()
    temporary_files = []

    logger.info("Stage 1: Filtering SV VCF for PASS DEL/DUP calls")
    filtered_sv_vcf = pjoin(output_directory, "filtered_sv.vcf.gz")
    filter_expr = f'(SVTYPE="DEL" | SVTYPE="DUP") & abs(SVLEN) >= {min_sv_length} & QUAL >= {min_sv_qual}'
    vcf_utils.view_vcf(
        input_vcf=sv_vcf,
        output_vcf=filtered_sv_vcf,
        extra_args=f"-f PASS -i '{filter_expr}'",
    )
    vcf_utils.index_vcf(filtered_sv_vcf)
    temporary_files.extend([filtered_sv_vcf, filtered_sv_vcf + ".tbi"])

    logger.info(
        f"Filtered SV VCF contains {vcf_utils.get_vcf_count(filtered_sv_vcf)} \
            PASS DEL/DUP calls {length_range}{qual_msg}{max_len_msg}"
    )

    # Stage 1b: Extract excluded SVs to preserve in final output
    logger.info("Stage 1b: Extracting excluded SVs (non-DEL/DUP, non-PASS DEL/DUP, small PASS DEL/DUP)")

    excluded_sv_vcf = pjoin(output_directory, "excluded_sv.vcf.gz")

    # Combine all exclusion criteria into single bcftools expression
    excluded_sv_expr = (
        f'(SVTYPE!="DEL" & SVTYPE!="DUP") | '
        f'((SVTYPE="DEL" | SVTYPE="DUP") & FILTER!="PASS") | '
        f'((SVTYPE="DEL" | SVTYPE="DUP") & FILTER="PASS" & abs(SVLEN) < {min_sv_length})'
    )

    vcf_utils.view_vcf(
        input_vcf=sv_vcf,
        output_vcf=excluded_sv_vcf,
        extra_args=f"-i '{excluded_sv_expr}'",
    )
    vcf_utils.index_vcf(excluded_sv_vcf)
    temporary_files.extend([excluded_sv_vcf, excluded_sv_vcf + ".tbi"])

    logger.info(f"Extracted {vcf_utils.get_vcf_count(excluded_sv_vcf)} excluded SVs")

    # Stage 2: Combine CNV and SV VCFs with source annotations
    logger.info("Stage 2: Combining CNV and SV VCFs")
    temp_sorted_vcf = pjoin(output_directory, "temp_sorted.vcf.gz")
    combine_cnv_or_sv_vcf_files(
        vcf_files=[(filtered_sv_vcf, "gridss"), (cnv_vcf, None)],
        output_vcf=temp_sorted_vcf,
        fasta_index=fasta_index,
        preserve_filters=True,
        output_directory=output_directory,
    )
    temporary_files.extend([temp_sorted_vcf, temp_sorted_vcf + ".tbi"])

    logger.info("Stage 3: Collapsing overlapping variants (SV replaces CNV via QUAL)")

    collapsed_vcf_tmp = pjoin(output_directory, "collapsed_tmp.vcf.gz")
    removed_vcf = vcf_utils.collapse_vcf(
        vcf=temp_sorted_vcf,
        output_vcf=collapsed_vcf_tmp,
        refdist=distance,
        pctseq=0.0,
        pctsize=pctsize,
        maxsize=-1,
        ignore_filter=False,
        ignore_sv_type=False,
        pick_best=True,
        erase_removed=False,
    )
    temporary_files.extend([collapsed_vcf_tmp, removed_vcf])

    logger.info("Post-processing: merging CNV_SOURCE and adding CNV_ID from removed variants")
    postprocessed_vcf = pjoin(output_directory, "postprocessed.vcf.gz")
    _postprocess_collapsed_vcf(collapsed_vcf_tmp, str(removed_vcf), postprocessed_vcf)
    temporary_files.append(postprocessed_vcf)

    # Apply max_sv_length filtering to large SVs without CNV overlap
    if max_sv_length is not None:
        logger.info(f"Stage 4: Filtering large SVs (>{max_sv_length}bp) without CNV overlap")
        filtered_large_sv_vcf = pjoin(output_directory, "filtered_large_sv.vcf.gz")
        total, filtered = _filter_large_svs_without_cnv_overlap(
            postprocessed_vcf,
            filtered_large_sv_vcf,
            max_sv_length,
        )
        logger.info(f"Filtered {filtered}/{total} large SVs without CNV overlap (kept large SVs with CNV overlap)")
        vcf_utils.index_vcf(filtered_large_sv_vcf)
        temporary_files.extend([filtered_large_sv_vcf, filtered_large_sv_vcf + ".tbi"])
        merged_vcf = filtered_large_sv_vcf
    else:
        merged_vcf = postprocessed_vcf

    # Stage 5: Add excluded SVs to final output
    # (Non-PASS CNVs already in merged_vcf via pass-through)
    logger.info("Stage 5: Adding excluded SVs to final output")
    # Concatenate merged results with excluded SVs
    vcf_utils.concat_vcf([merged_vcf, excluded_sv_vcf], output_vcf)
    logger.info(f"Final VCF contains {vcf_utils.get_vcf_count(output_vcf)} total variants")

    mu.cleanup_temp_files(temporary_files)
    logger.info(f"Successfully created merged CNV+SV VCF: {output_vcf}")

    return output_vcf


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="merge_cnv_sv", description="Merge CNV and SV calls, preferring higher-quality SV calls on overlap"
    )
    parser.add_argument("--cnv_vcf", help="Input CNV VCF file", required=True, type=str)
    parser.add_argument("--sv_vcf", help="Input SV VCF file (e.g., from GRIDSS)", required=True, type=str)
    parser.add_argument("--output_vcf", help="Output merged VCF file", required=True, type=str)
    parser.add_argument("--fasta_index", help="Reference genome FASTA index (.fai) file", required=True, type=str)
    parser.add_argument(
        "--min_sv_length",
        help="Minimum absolute SVLEN for SV DEL / DUP calls to merge with the CNV calls (default: 500, the minimum CNV "
        "call length); shorter SV calls are kept unmerged",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--max_sv_length",
        help="Maximum absolute SVLEN for SV calls to include (default: 5000000, 5Mb)",
        type=int,
        default=5000000,
    )
    parser.add_argument(
        "--min_sv_qual",
        help="Minimum QUAL score for SV calls to include (default: 0, no minimum)",
        type=float,
        default=0,
    )
    parser.add_argument(
        "--distance",
        help="Distance threshold for collapsing overlapping variants (default: 0, exact overlaps only)",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--pctsize",
        help="Minimum size similarity (0.0-1.0) for collapsing (default: 0.5, require 50%% match)",
        type=float,
        default=0.5,
    )
    return parser.parse_args(argv)


def run(argv: list[str]):
    args = parse_args(argv[1:])
    merge_cnv_sv_vcfs(
        cnv_vcf=args.cnv_vcf,
        sv_vcf=args.sv_vcf,
        output_vcf=args.output_vcf,
        fasta_index=args.fasta_index,
        min_sv_length=args.min_sv_length,
        max_sv_length=args.max_sv_length,
        min_sv_qual=args.min_sv_qual,
        distance=args.distance,
        pctsize=args.pctsize,
    )


def main():
    run(sys.argv)


if __name__ == "__main__":
    main()
