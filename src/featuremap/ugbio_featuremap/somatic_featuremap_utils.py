import os
import tempfile
from os.path import join as pjoin

import pysam
from simppl.simple_pipeline import SimplePipeline
from ugbio_core.vcf_utils import VcfUtils


def _create_variant_bed(merged_vcf, bed_file):
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


def integrate_tandem_repeat_features(merged_vcf, ref_tr_file, genome_file, out_dir):
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
