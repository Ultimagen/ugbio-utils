# utilities for combining VCFs
from os.path import join as pjoin

import numpy as np
import pysam
from ugbio_core import misc_utils as mu
from ugbio_core.logger import logger
from ugbio_core.vcf_utils import VcfUtils
from ugbio_core.vcfbed import vcftools


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


def update_vcf_contigs(
    vcf_utils: VcfUtils,
    cnmops_vcf: str,
    cnvpytor_vcf: str,
    fasta_index: str,
    output_directory: str,
) -> tuple[str, str]:
    """
    Update VCF headers with contigs from FASTA index.

    Parameters
    ----------
    vcf_utils : VcfUtils
        VcfUtils instance for VCF operations
    cnmops_vcf : str
        Path to cn.mops VCF
    cnvpytor_vcf : str
        Path to CNVpytor VCF
    fasta_index : str
        Path to FASTA index file
    output_directory : str
        Output directory for temporary files

    Returns
    -------
    tuple[str, str]
        Paths to updated cn.mops and CNVpytor VCF files
    """
    logger.info("Updating VCF headers with contigs from FASTA index")
    cnmops_vcf_updated = pjoin(output_directory, "cnmops.updated_contigs.vcf.gz")
    cnvpytor_vcf_updated = pjoin(output_directory, "cnvpytor.updated_contigs.vcf.gz")

    vcf_utils.update_vcf_contigs_from_fai(cnmops_vcf, cnmops_vcf_updated, fasta_index)
    vcf_utils.update_vcf_contigs_from_fai(cnvpytor_vcf, cnvpytor_vcf_updated, fasta_index)

    return cnmops_vcf_updated, cnvpytor_vcf_updated


def write_vcf_records_with_source(
    vcf_in: pysam.VariantFile,
    vcf_out: pysam.VariantFile,
    combined_header: pysam.VariantHeader,
    source_name: str,
) -> None:
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
    """
    logger.info(f"Writing records from {source_name} VCF")
    for record in vcf_in:
        # Clear filters - we remove filters imposed by the previous pipelines
        record.filter.clear()
        record.filter.add("PASS")
        # Create new record with combined header
        new_record = VcfUtils.copy_vcf_record(record, combined_header)
        # Add source tag if not already present
        if "CNV_SOURCE" not in new_record.info:
            new_record.info["CNV_SOURCE"] = (source_name,)
        vcf_out.write(new_record)


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

    # Add samples from both headers
    for sample in header1.samples:
        if sample not in combined_header.samples:
            combined_header.add_sample(sample)
    for sample in header2.samples:
        if sample not in combined_header.samples:
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


def merge_cnvs_in_vcf(input_vcf: str, output_vcf: str, distance: int = 1000) -> None:
    """
    Merge CNV variants in a VCF file that are within a specified distance.

    Parameters
    ----------
    input_vcf : str
        Path to the input VCF file containing CNV variants.
    output_vcf : str
        Path to the output VCF file where merged variants will be written.
    distance : int, optional
        Maximum distance between CNV variants to consider them for merging, by default 1000.

    Returns
    -------
    None
        Writes the merged VCF to output_vcf and creates an index.
    """
    output_vcf_collapse = output_vcf + ".collapse.tmp.vcf.gz"
    temporary_files = [output_vcf_collapse]
    vu = VcfUtils()
    removed_vcf = vu.collapse_vcf(
        vcf=input_vcf,
        output_vcf=output_vcf_collapse,
        refdist=distance,
        pctseq=0.0,
        pctsize=0.0,
        ignore_filter=True,
        ignore_sv_type=False,
        erase_removed=False,
    )
    temporary_files.append(str(removed_vcf))

    action_dictionary = {
        "weighted_avg": [
            "CNMOPS_SAMPLE_MEAN",
            "CNMOPS_SAMPLE_STDEV",
            "CNMOPS_COHORT_MEAN",
            "CNMOPS_COHORT_STDEV",
            "CopyNumber",
        ]
    }
    all_fields = sum(action_dictionary.values(), [])
    update_df = vcftools.get_vcf_df(str(removed_vcf), custom_info_fields=all_fields + ["SVLEN", "MatchId"]).sort_index()
    update_df["matchid"] = update_df["matchid"].apply(lambda x: x[0]).astype(float)
    update_df["end"] = update_df["pos"] + update_df["svlen"].apply(lambda x: x[0]) - 1
    with pysam.VariantFile(output_vcf_collapse) as vcf_in:
        hdr = vcf_in.header
        with pysam.VariantFile(output_vcf, "w", header=hdr) as vcf_out:
            for record in vcf_in:
                if "CollapseId" in record.info:
                    cid = float(record.info["CollapseId"])
                    update_records = update_df[update_df["matchid"] == cid]
                    record.stop = update_records["end"].max()
                    for action, fields in action_dictionary.items():
                        for field in fields:
                            values = np.array(list(update_records[field.lower()]) + [record.info[field]])
                            if action == "weighted_avg":
                                lengths = np.array(
                                    list(update_records["svlen"].apply(lambda x: x[0])) + [record.info["SVLEN"][0]]
                                )
                                weighted_avg = np.sum(values * lengths) / np.sum(lengths)
                                record.info[field] = round(weighted_avg, 3)

                    record.info["SVLEN"] = (record.stop - record.start,)
                vcf_out.write(record)
    mu.cleanup_temp_files(temporary_files)
