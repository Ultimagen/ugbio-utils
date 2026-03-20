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
    """Merge CNV and SV VCF files, preferring higher-quality SV calls on overlap."""
    if output_directory is None:
        output_directory = os.path.dirname(os.path.abspath(output_vcf))
        os.makedirs(output_directory, exist_ok=True)
    elif not os.path.isdir(output_directory):
        raise ValueError(f"Output directory does not exist: {output_directory}")

    logger.info(f"Starting CNV+SV merge: CNV={cnv_vcf}, SV={sv_vcf}")
    max_len_msg = f" to {max_sv_length}bp" if max_sv_length else ""
    qual_msg = f", Min QUAL: {min_sv_qual}" if min_sv_qual > 0 else ""
    logger.info(f"Length threshold: {min_sv_length}bp{max_len_msg}{qual_msg}, Distance: {distance}bp")

    vcf_utils = VcfUtils()
    temporary_files = []

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

    with pysam.VariantFile(filtered_sv_vcf) as vcf_in:
        sv_count = sum(1 for _ in vcf_in)
    length_range = f">= {min_sv_length}bp"
    if max_sv_length is not None:
        length_range = f"between {min_sv_length}bp and {max_sv_length}bp"
    qual_suffix = f" with QUAL >= {min_sv_qual}" if min_sv_qual > 0 else ""
    logger.info(f"Filtered SV VCF contains {sv_count} PASS DEL/DUP calls {length_range}{qual_suffix}")

    logger.info("Stage 2: Combining VCF headers")
    vcf_cnv = pysam.VariantFile(cnv_vcf)
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
        )

        vcf_cnv.close()
        vcf_cnv = pysam.VariantFile(cnv_vcf)
        cnv_source = "unknown"
        for record in vcf_cnv:
            if "CNV_SOURCE" in record.info:
                cnv_source = record.info["CNV_SOURCE"][0]
            break

        vcf_cnv.close()
        vcf_cnv = pysam.VariantFile(cnv_vcf)
        write_vcf_records_with_source(
            vcf_cnv,
            vcf_out,
            combined_header,
            cnv_source,
            make_ids_unique=True,
            seen_ids=seen_ids,
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

    vcf_utils.sort_vcf(postprocessed_vcf, output_vcf)
    vcf_utils.index_vcf(output_vcf)

    mu.cleanup_temp_files(temporary_files)
    logger.info(f"Successfully created merged CNV+SV VCF: {output_vcf}")

    return output_vcf
