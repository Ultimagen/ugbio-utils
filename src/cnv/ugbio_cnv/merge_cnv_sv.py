"""Utilities for merging CNV and SV VCF files."""

import os
from os.path import join as pjoin

import pysam
from ugbio_cnv.cnv_vcf_consts import INFO_TAG_REGISTRY
from ugbio_cnv.combine_cnv_vcf_utils import combine_cnv_or_sv_vcf_files
from ugbio_core import misc_utils as mu
from ugbio_core.logger import logger
from ugbio_core.vcf_utils import VcfUtils
from ugbio_core.vcfbed import vcftools


def _flatten_cnv_sources(sources_series):
    """
    Flatten and deduplicate CNV_SOURCE values from multiple variants.

    CNV_SOURCE can be a string, list, or tuple. This function:
    1. Converts all to lists
    2. Flattens into a single list
    3. Removes duplicates and sorts

    Example: ["cn.mops", ["cnvpytor", "gridss"]] -> ["cn.mops", "cnvpytor", "gridss"]
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
    - CNV_SOURCE: All source tools (cn.mops, cnvpytor, gridss, etc.)
    - IDs: All variant IDs that were removed

    Parameters
    ----------
    removed_vcf : str
        Path to VCF file containing removed/merged variants

    Returns
    -------
    dict[float, dict]
        Dictionary mapping CollapseId -> {cnv_source: [...], id: [...]}
        Example: {1.0: {"cnv_source": ["cn.mops", "gridss"], "id": ["CNV_1", "CNV_2"]}}
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
    min_sv_length: int = 1000,
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
    - Non-DEL/DUP types (INV, INS, but excluding BND)
    - Non-PASS DEL/DUP (failed filters in original VCF)
    - PASS DEL/DUP below size threshold (too small for merge)

    CNVs excluded from merge:
    - Non-PASS CNVs (failed filters in original VCF)

    BND variants are excluded completely from output.

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
        Minimum absolute SVLEN for SV calls to include (default: 1000)
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
        f'(SVTYPE!="DEL" & SVTYPE!="DUP" & SVTYPE!="BND") | '
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
