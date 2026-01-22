import os
import tempfile
from os.path import join as pjoin

import pysam
from simppl.simple_pipeline import SimplePipeline
from ugbio_core.logger import logger
from ugbio_core.vcf_utils import VcfUtils


def _create_variant_bed(merged_vcf: str, bed_file: str) -> None:
    """Create a BED file from VCF variants."""
    sp = SimplePipeline(0, 1)
    cmd_bcftools = f"bcftools query -f '%CHROM\t%POS0\t%END\n' {merged_vcf}"
    sp.print_and_run(cmd_bcftools, out=bed_file)


def _find_closest_tandem_repeats(bed1, bed2, genome_file, output_file):
    """Find closest tandem repeats for each variant."""
    sp = SimplePipeline(0, 1)
    cmd = f"bedtools closest -D ref -g {genome_file} -a {bed1} -b {bed2} | cut -f1,3,5-10"
    sp.print_and_run(cmd, out=output_file)


def _prepare_annotation_files(tmpdir, tr_tsv_file):
    """Prepare files for VCF annotation."""
    sp = SimplePipeline(0, 3)
    sorted_tsv = pjoin(tmpdir, "merged_vcf.tmp.TRdata.sorted.tsv")
    cmd = f"sort -k1,1 -k2,2n {tr_tsv_file}"
    sp.print_and_run(cmd, out=sorted_tsv)

    gz_tsv = sorted_tsv + ".gz"
    cmd = f"bgzip -c {sorted_tsv}"
    sp.print_and_run(cmd, out=gz_tsv)

    cmd = f"tabix -s 1 -b 2 -e 2 {gz_tsv}"
    sp.print_and_run(cmd)

    header = pysam.VariantHeader()
    info_fields = [
        ("TR_START", "1", "Integer", "Closest tandem Repeat Start"),
        ("TR_END", "1", "Integer", "Closest Tandem Repeat End"),
        ("TR_SEQ", "1", "String", "Closest Tandem Repeat Sequence"),
        ("TR_LENGTH", "1", "Integer", "Closest Tandem Repeat total length"),
        ("TR_SEQ_UNIT_LENGTH", "1", "Integer", "Closest Tandem Repeat unit length"),
        ("TR_DISTANCE", "1", "Integer", "Closest Tandem Repeat Distance"),
    ]
    for hdr_id, hdr_number, hdr_type, hdr_description in info_fields:
        header.add_meta(
            "INFO", items=[("ID", hdr_id), ("Number", hdr_number), ("Type", hdr_type), ("Description", hdr_description)]
        )

    hdr_file = pjoin(tmpdir, "tr_hdr.txt")
    # Write the header into a plain text file
    with open(hdr_file, "w") as f:
        lines = str(header).splitlines()
        for line in lines[1:-1]:
            f.write(line + "\n")

    return gz_tsv, hdr_file


def integrate_tandem_repeat_features(merged_vcf: str, ref_tr_file: str, genome_file: str, out_dir: str) -> str:
    """
    Integrate tandem repeat features into a VCF file.

    This function annotates variants in a VCF file with information about nearby tandem repeats,
    including their start/end positions, sequences, distances, and lengths.

    Parameters
    ----------
    merged_vcf : str
        Path to the input VCF file (gzipped) containing variants to be annotated.
    ref_tr_file : str
        Path to the reference tandem repeat BED file containing tandem repeat regions.
    genome_file : str
        Path to the reference genome FASTA index file (.fai).
    out_dir : str
        Output directory where the annotated VCF file and temporary files will be written.

    Returns
    -------
    str
        Path to the output VCF file with tandem repeat annotations. The output file will be
        gzipped and indexed, with '.tr_info.vcf.gz' suffix replacing the original '.vcf.gz'.

    Notes
    -----
    The function adds the following INFO fields to the VCF:
    - TR_START: Start position of the closest tandem repeat
    - TR_END: End position of the closest tandem repeat
    - TR_SEQ: Sequence of the tandem repeat unit
    - TR_LENGTH: Total length of the tandem repeat region
    - TR_SEQ_UNIT_LENGTH: Length of the tandem repeat unit
    - TR_DISTANCE: Distance from variant to the closest tandem repeat
    """
    with tempfile.TemporaryDirectory(dir=out_dir) as tmpdir:
        # Create variant BED file
        bed1 = pjoin(tmpdir, "merged_vcf.tmp.bed")
        _create_variant_bed(merged_vcf, bed1)

        # Find closest tandem repeats
        bed1_with_closest_tr_tsv = pjoin(tmpdir, "merged_vcf.tmp.TRdata.tsv")
        _find_closest_tandem_repeats(bed1, ref_tr_file, genome_file, bed1_with_closest_tr_tsv)

        # Prepare annotation files
        gz_tsv, hdr_file = _prepare_annotation_files(tmpdir, bed1_with_closest_tr_tsv)

        # Annotate VCF
        merged_vcf_with_tr_info = pjoin(out_dir, os.path.basename(merged_vcf).replace(".vcf.gz", ".tr_info.vcf.gz"))

        # Use VcfUtils to annotate the VCF
        vcf_utils = VcfUtils()
        vcf_utils.annotate_vcf(
            input_vcf=merged_vcf,
            output_vcf=merged_vcf_with_tr_info,
            annotation_file=gz_tsv,
            header_file=hdr_file,
            columns="CHROM,POS,INFO/TR_START,INFO/TR_END,INFO/TR_SEQ,INFO/TR_LENGTH,INFO/TR_SEQ_UNIT_LENGTH,INFO/TR_DISTANCE",
        )

    return merged_vcf_with_tr_info


def filter_and_annotate_tr(
    input_vcf: str,
    ref_tr_file: str,
    genome_file: str,
    out_dir: str,
    filter_string: str | None = "PASS",
) -> str:
    """
    Filter VCF and annotate with tandem repeat features in a single pass.

    This unified preprocessing function:
    1. Filters the VCF to keep only specified variants (e.g., PASS)
    2. Annotates the filtered variants with tandem repeat information

    By filtering first, we reduce the number of variants processed by TR annotation,
    improving performance on large files.

    Parameters
    ----------
    input_vcf : str
        Path to the input VCF file (gzipped).
    ref_tr_file : str
        Path to the reference tandem repeat BED file.
    genome_file : str
        Path to the reference genome FASTA index file (.fai).
    out_dir : str
        Output directory for the processed VCF file.
    filter_string : str, optional
        FILTER value to keep (e.g., "PASS"). If None, no filtering is applied.
        Defaults to "PASS".

    Returns
    -------
    str
        Path to the output VCF file with FILTER applied and TR annotations added.
        The output file will have '.filtered.tr_info.vcf.gz' suffix.
    """
    vcf_utils = VcfUtils()

    with tempfile.TemporaryDirectory(dir=out_dir) as tmpdir:
        # Step 1: Filter VCF (if filter_string is provided)
        if filter_string:
            logger.debug(f"Filtering VCF with filter: {filter_string}")
            filtered_vcf = pjoin(tmpdir, os.path.basename(input_vcf).replace(".vcf.gz", ".filtered.vcf.gz"))
            extra_args = f"-f {filter_string}"
            vcf_utils.view_vcf(input_vcf, filtered_vcf, n_threads=os.cpu_count() or 1, extra_args=extra_args)
            vcf_utils.index_vcf(filtered_vcf)
            vcf_to_annotate = filtered_vcf
        else:
            vcf_to_annotate = input_vcf

        # Step 2: Create variant BED file from filtered VCF
        logger.debug("Creating variant BED file for TR annotation")
        bed_file = pjoin(tmpdir, "variants.bed")
        _create_variant_bed(vcf_to_annotate, bed_file)

        # Step 3: Find closest tandem repeats
        logger.debug("Finding closest tandem repeats")
        tr_tsv = pjoin(tmpdir, "variants.tr_data.tsv")
        _find_closest_tandem_repeats(bed_file, ref_tr_file, genome_file, tr_tsv)

        # Step 4: Prepare annotation files (sort, bgzip, tabix)
        logger.debug("Preparing TR annotation files")
        gz_tsv, hdr_file = _prepare_annotation_files(tmpdir, tr_tsv)

        # Step 5: Annotate VCF with TR fields
        logger.debug("Annotating VCF with TR fields")
        output_suffix = ".filtered.tr_info.vcf.gz" if filter_string else ".tr_info.vcf.gz"
        output_vcf = pjoin(out_dir, os.path.basename(input_vcf).replace(".vcf.gz", output_suffix))
        vcf_utils.annotate_vcf(
            input_vcf=vcf_to_annotate,
            output_vcf=output_vcf,
            annotation_file=gz_tsv,
            header_file=hdr_file,
            columns="CHROM,POS,INFO/TR_START,INFO/TR_END,INFO/TR_SEQ,INFO/TR_LENGTH,INFO/TR_SEQ_UNIT_LENGTH,INFO/TR_DISTANCE",
        )

    logger.info(f"Filtered and TR-annotated VCF written to: {output_vcf}")
    return output_vcf
