# utilities for combining VCFs
import os
from os.path import join as pjoin
from pathlib import Path

import pysam
from ugbio_cnv.cnv_vcf_consts import INFO_TAG_REGISTRY
from ugbio_core import misc_utils as mu
from ugbio_core.logger import logger
from ugbio_core.vcf_utils import VcfUtils


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
