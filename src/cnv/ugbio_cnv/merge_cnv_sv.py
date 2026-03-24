"""Utilities for merging CNV and SV VCF files."""

import os
from os.path import join as pjoin

import pysam
from ugbio_cnv.cnv_vcf_consts import INFO_TAG_REGISTRY
from ugbio_cnv.combine_cnv_vcf_utils import combine_vcf_headers_for_cnv, write_vcf_records_with_source
from ugbio_core import misc_utils as mu
from ugbio_core.logger import logger
from ugbio_core.vcf_utils import VcfUtils


def _postprocess_collapsed_vcf(  # noqa: C901, PLR0912, PLR0915
    collapsed_vcf: str,
    removed_vcf: str,
    output_vcf: str,
) -> None:
    """Merge CNV source metadata from removed variants into collapsed variants."""
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

    collapsed_count = 0
    num_collapsed_count = 0
    updated_count = 0

    records_to_process = []
    with pysam.VariantFile(collapsed_vcf) as vcf_in:
        header = vcf_in.header.copy()
        if "CNV_ID" not in header.info:
            header.info.add(*INFO_TAG_REGISTRY["CNV_ID"][:-1])

        for record in vcf_in:
            records_to_process.append(record)

    with pysam.VariantFile(output_vcf, "w", header=header) as vcf_out:
        for old_record in records_to_process:
            collapsed_count += 1

            record = vcf_out.new_record()
            record.chrom = old_record.chrom
            record.pos = old_record.pos
            record.id = old_record.id
            record.ref = old_record.ref
            record.alts = old_record.alts
            record.qual = old_record.qual
            if old_record.filter:
                record.filter.add(*old_record.filter)
            for key in old_record.info:
                record.info[key] = old_record.info[key]
            record.stop = old_record.stop
            for sample in old_record.samples:
                for key in old_record.samples[sample]:
                    record.samples[sample][key] = old_record.samples[sample][key]

            num_collapsed = record.info.get("NumCollapsed", 0)
            if num_collapsed > 0:
                num_collapsed_count += 1

                overlapping = []
                for removed in removed_variants:
                    if (
                        removed["chrom"] == record.chrom
                        and removed["svtype"] == record.info.get("SVTYPE", "")
                        and removed["start"] < record.stop
                        and removed["stop"] > record.start
                    ):
                        overlapping.append(removed)

                if overlapping:
                    updated_count += 1
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

                    record.info["CNV_SOURCE"] = sorted(all_sources)

                    removed_ids = [variant["id"] for variant in overlapping if variant["id"]]
                    if removed_ids:
                        record.info["CNV_ID"] = removed_ids

            for tag in ["NumCollapsed", "NumConsolidated", "CollapseId"]:
                if tag in record.info:
                    del record.info[tag]

            vcf_out.write(record)

    logger.info(
        f"Post-processing complete: {collapsed_count} total variants, "
        f"{num_collapsed_count} with NumCollapsed>0, "
        f"{updated_count} updated with CNV_ID"
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


def merge_cnv_sv_vcfs(  # noqa: PLR0915, PLR0912, C901
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
    """
    if output_directory is None:
        output_directory = os.path.dirname(os.path.abspath(output_vcf))
        os.makedirs(output_directory, exist_ok=True)
    elif not os.path.isdir(output_directory):
        raise ValueError(f"Output directory does not exist: {output_directory}")

    logger.info(f"Starting CNV+SV merge: CNV={cnv_vcf}, SV={sv_vcf}")
    max_len_msg = f" (large SVs >{max_sv_length}bp kept only if overlapping CNV)" if max_sv_length else ""
    qual_msg = f", Min QUAL: {min_sv_qual}" if min_sv_qual > 0 else ""
    logger.info(f"Length threshold: >={min_sv_length}bp{max_len_msg}{qual_msg}, Distance: {distance}bp")

    vcf_utils = VcfUtils()
    temporary_files = []

    logger.info("Stage 1: Filtering SV VCF for PASS DEL/DUP calls")
    filtered_sv_vcf = pjoin(output_directory, "filtered_sv.vcf.gz")
    filter_expr = f'(SVTYPE="DEL" | SVTYPE="DUP") & abs(SVLEN) >= {min_sv_length}'
    # Note: max_sv_length filtering happens post-collapse (large SVs kept only if overlapping CNV)
    if min_sv_qual > 0:
        filter_expr += f" & QUAL >= {min_sv_qual}"
    vcf_utils.view_vcf(
        input_vcf=sv_vcf,
        output_vcf=filtered_sv_vcf,
        extra_args=f"-f PASS -i '{filter_expr}'",
    )
    vcf_utils.index_vcf(filtered_sv_vcf)
    temporary_files.append(filtered_sv_vcf)

    with pysam.VariantFile(filtered_sv_vcf) as vcf_in:
        sv_count = sum(1 for _ in vcf_in)
    length_range = f">= {min_sv_length}bp"
    qual_suffix = f" with QUAL >= {min_sv_qual}" if min_sv_qual > 0 else ""
    max_len_note = f" (large SVs >{max_sv_length}bp will be filtered unless they overlap CNVs)" if max_sv_length else ""
    logger.info(f"Filtered SV VCF contains {sv_count} PASS DEL/DUP calls {length_range}{qual_suffix}{max_len_note}")

    # Stage 1a: Filter CNV VCF for PASS calls only
    logger.info("Stage 1a: Filtering CNV VCF for PASS calls")
    filtered_cnv_vcf = pjoin(output_directory, "filtered_cnv.vcf.gz")
    vcf_utils.view_vcf(
        input_vcf=cnv_vcf,
        output_vcf=filtered_cnv_vcf,
        extra_args="-f PASS",
    )
    vcf_utils.index_vcf(filtered_cnv_vcf)
    temporary_files.append(filtered_cnv_vcf)

    with pysam.VariantFile(filtered_cnv_vcf) as vcf_in:
        cnv_pass_count = sum(1 for _ in vcf_in)
    logger.info(f"Filtered CNV VCF contains {cnv_pass_count} PASS calls")

    # Stage 1b: Extract excluded SVs to preserve in final output
    logger.info("Stage 1b: Extracting excluded SVs (non-DEL/DUP, non-PASS DEL/DUP, small PASS DEL/DUP)")

    # Extract non-DEL/DUP SVs (INV, INS, etc., but excluding BND)
    non_deldip_vcf = pjoin(output_directory, "non_deldip_sv.vcf.gz")
    vcf_utils.view_vcf(
        input_vcf=sv_vcf,
        output_vcf=non_deldip_vcf,
        extra_args='-i \'SVTYPE!="DEL" & SVTYPE!="DUP" & SVTYPE!="BND"\' -O z',
    )
    vcf_utils.index_vcf(non_deldip_vcf)
    temporary_files.append(non_deldip_vcf)

    # Extract non-PASS DEL/DUP SVs (failed filters in original VCF)
    filtered_deldip_expr = '(SVTYPE="DEL" | SVTYPE="DUP") & FILTER!="PASS"'

    filtered_deldip_vcf = pjoin(output_directory, "filtered_deldip_sv.vcf.gz")
    vcf_utils.view_vcf(
        input_vcf=sv_vcf,
        output_vcf=filtered_deldip_vcf,
        extra_args=f"-i '{filtered_deldip_expr}' -O z",
    )
    vcf_utils.index_vcf(filtered_deldip_vcf)
    temporary_files.append(filtered_deldip_vcf)

    # Extract PASS DEL/DUP below size threshold (too small for merge)
    small_deldip_expr = f'(SVTYPE="DEL" | SVTYPE="DUP") & FILTER="PASS" & abs(SVLEN) < {min_sv_length}'

    small_deldip_vcf = pjoin(output_directory, "small_deldip_sv.vcf.gz")
    vcf_utils.view_vcf(
        input_vcf=sv_vcf,
        output_vcf=small_deldip_vcf,
        extra_args=f"-i '{small_deldip_expr}' -O z",
    )
    vcf_utils.index_vcf(small_deldip_vcf)
    temporary_files.append(small_deldip_vcf)

    # Check counts before concatenating
    with pysam.VariantFile(non_deldip_vcf) as vcf_in:
        non_deldip_count = sum(1 for _ in vcf_in)
    with pysam.VariantFile(filtered_deldip_vcf) as vcf_in:
        filtered_deldip_count = sum(1 for _ in vcf_in)
    with pysam.VariantFile(small_deldip_vcf) as vcf_in:
        small_deldip_count = sum(1 for _ in vcf_in)

    # Concatenate non-empty VCFs into single excluded VCF
    excluded_sv_vcf = pjoin(output_directory, "excluded_sv.vcf.gz")
    vcfs_to_concat = []
    if non_deldip_count > 0:
        vcfs_to_concat.append(non_deldip_vcf)
    if filtered_deldip_count > 0:
        vcfs_to_concat.append(filtered_deldip_vcf)
    if small_deldip_count > 0:
        vcfs_to_concat.append(small_deldip_vcf)

    if len(vcfs_to_concat) > 1:
        vcf_utils.concat_vcf(vcfs_to_concat, excluded_sv_vcf)
    elif len(vcfs_to_concat) == 1:
        # Only one VCF, just copy it
        import shutil

        shutil.copy(vcfs_to_concat[0], excluded_sv_vcf)
        shutil.copy(f"{vcfs_to_concat[0]}.tbi", f"{excluded_sv_vcf}.tbi")
    else:
        # No excluded SVs, create empty VCF
        with pysam.VariantFile(sv_vcf) as vcf_in:
            header = vcf_in.header.copy()
            with pysam.VariantFile(excluded_sv_vcf, "w", header=header) as vcf_out:
                pass  # Write empty VCF with header
        vcf_utils.index_vcf(excluded_sv_vcf)

    temporary_files.append(excluded_sv_vcf)

    # Log count of excluded SVs
    excluded_count = non_deldip_count + filtered_deldip_count + small_deldip_count
    logger.info(
        f"Extracted {excluded_count} excluded SVs "
        f"(non-DEL/DUP={non_deldip_count}, non-PASS DEL/DUP={filtered_deldip_count}, "
        f"small PASS DEL/DUP={small_deldip_count})"
    )

    # Stage 1c: Extract non-PASS CNVs to preserve in final output
    logger.info("Stage 1c: Extracting non-PASS CNVs")
    excluded_cnv_tmp_vcf = pjoin(output_directory, "excluded_cnv_tmp.vcf.gz")
    vcf_utils.view_vcf(
        input_vcf=cnv_vcf,
        output_vcf=excluded_cnv_tmp_vcf,
        extra_args="-e 'FILTER=\"PASS\"'",  # Exclude PASS (i.e., get non-PASS)
    )
    temporary_files.append(excluded_cnv_tmp_vcf)

    # Remove CNV_SOURCE and CNV_ID from excluded CNVs (they didn't participate in merge)
    excluded_cnv_vcf = pjoin(output_directory, "excluded_cnv.vcf.gz")
    with pysam.VariantFile(excluded_cnv_tmp_vcf) as vcf_in:
        header = vcf_in.header.copy()
        with pysam.VariantFile(excluded_cnv_vcf, "w", header=header) as vcf_out:
            excluded_cnv_count = 0
            for record in vcf_in:
                # Remove CNV_SOURCE and CNV_ID if present
                if "CNV_SOURCE" in record.info:
                    del record.info["CNV_SOURCE"]
                if "CNV_ID" in record.info:
                    del record.info["CNV_ID"]
                vcf_out.write(record)
                excluded_cnv_count += 1

    vcf_utils.index_vcf(excluded_cnv_vcf)
    temporary_files.append(excluded_cnv_vcf)
    logger.info(f"Extracted {excluded_cnv_count} non-PASS CNVs (CNV_SOURCE removed)")

    logger.info("Stage 2: Combining VCF headers")
    vcf_cnv = pysam.VariantFile(filtered_cnv_vcf)  # Use filtered CNV VCF instead of original
    vcf_sv = pysam.VariantFile(filtered_sv_vcf)
    combined_header = combine_vcf_headers_for_cnv(vcf_cnv.header, vcf_sv.header, keep_filters=True)

    if "CNV_SOURCE" not in combined_header.info:
        combined_header.info.add(*INFO_TAG_REGISTRY["CNV_SOURCE"][:-1])

    logger.info("Writing combined VCF with source annotations")
    temp_combined_vcf = pjoin(output_directory, "temp_combined.vcf.gz")
    with pysam.VariantFile(temp_combined_vcf, "w", header=combined_header) as vcf_out:
        seen_ids = write_vcf_records_with_source(
            vcf_sv,
            vcf_out,
            combined_header,
            "gridss",
            make_ids_unique=True,
            seen_ids=set(),
            clear_filters=False,  # SVs are already filtered to PASS, preserve PASS
        )

        vcf_cnv.close()
        vcf_cnv = pysam.VariantFile(filtered_cnv_vcf)  # Use filtered CNV VCF
        cnv_source = "unknown"
        for record in vcf_cnv:
            if "CNV_SOURCE" in record.info:
                cnv_source = record.info["CNV_SOURCE"][0]
            break

        vcf_cnv.close()
        vcf_cnv = pysam.VariantFile(filtered_cnv_vcf)  # Use filtered CNV VCF
        write_vcf_records_with_source(
            vcf_cnv,
            vcf_out,
            combined_header,
            cnv_source,
            make_ids_unique=True,
            seen_ids=seen_ids,
            clear_filters=False,  # CNVs are already filtered to PASS, preserve PASS
        )

    vcf_cnv.close()
    vcf_sv.close()
    temporary_files.append(temp_combined_vcf)

    logger.info("Stage 3: Sorting combined VCF")
    temp_sorted_vcf = pjoin(output_directory, "temp_sorted.vcf.gz")
    vcf_utils.sort_vcf(temp_combined_vcf, temp_sorted_vcf)
    vcf_utils.index_vcf(temp_sorted_vcf)
    temporary_files.append(temp_sorted_vcf)

    logger.info("Stage 4: Collapsing overlapping variants (SV replaces CNV via QUAL)")
    logger.info(f"Collapse parameters: refdist={distance}bp, pctsize={pctsize * 100}%")

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
    _postprocess_collapsed_vcf(collapsed_vcf_tmp, removed_vcf, postprocessed_vcf)
    temporary_files.append(postprocessed_vcf)

    # Apply max_sv_length filtering to large SVs without CNV overlap
    if max_sv_length is not None:
        logger.info(f"Stage 5: Filtering large SVs (>{max_sv_length}bp) without CNV overlap")
        filtered_large_sv_vcf = pjoin(output_directory, "filtered_large_sv.vcf.gz")
        total, filtered = _filter_large_svs_without_cnv_overlap(
            postprocessed_vcf,
            filtered_large_sv_vcf,
            max_sv_length,
        )
        filtered_large_sorted_sv_vcf = pjoin(output_directory, "filtered_large_sorted_sv.vcf.gz")
        vcf_utils.sort_vcf(filtered_large_sv_vcf, filtered_large_sorted_sv_vcf)
        logger.info(f"Filtered {filtered}/{total} large SVs without CNV overlap " f"(kept large SVs with CNV overlap)")
        vcf_utils.index_vcf(filtered_large_sorted_sv_vcf)
        temporary_files.append(filtered_large_sorted_sv_vcf)
        temporary_files.append(filtered_large_sorted_sv_vcf + ".tbi")
        temporary_files.append(filtered_large_sv_vcf + ".tbi")
        merged_vcf = filtered_large_sorted_sv_vcf
    else:
        merged_vcf = postprocessed_vcf

    # Stage 6: Concatenate excluded SVs and CNVs with merged output
    logger.info("Stage 6: Adding excluded SVs and CNVs to final output")

    # Check if there are any excluded SVs or CNVs
    with pysam.VariantFile(excluded_sv_vcf) as vcf_in:
        excluded_sv_count = sum(1 for _ in vcf_in)

    vcfs_to_concat = [merged_vcf]
    if excluded_sv_count > 0:
        vcfs_to_concat.append(excluded_sv_vcf)
    if excluded_cnv_count > 0:
        vcfs_to_concat.append(excluded_cnv_vcf)

    if len(vcfs_to_concat) > 1:
        # Concatenate merged results with excluded variants
        temp_concat_vcf = pjoin(output_directory, "temp_with_excluded.vcf.gz")
        vcf_utils.concat_vcf(vcfs_to_concat, temp_concat_vcf)
        temporary_files.append(temp_concat_vcf)

        # Sort and index final output
        vcf_utils.sort_vcf(temp_concat_vcf, output_vcf)
        logger.info(f"Added {excluded_sv_count} excluded SVs and {excluded_cnv_count} excluded CNVs to final output")
    else:
        # No excluded variants, just sort the merged output
        vcf_utils.sort_vcf(merged_vcf, output_vcf)
        logger.info("No excluded variants to add (all SVs and CNVs passed filters)")

    vcf_utils.index_vcf(output_vcf)

    with pysam.VariantFile(output_vcf) as vcf_in:
        final_count = sum(1 for _ in vcf_in)
    logger.info(f"Final VCF contains {final_count} total variants")

    mu.cleanup_temp_files(temporary_files)
    logger.info(f"Successfully created merged CNV+SV VCF: {output_vcf}")

    return output_vcf
