import os
import subprocess
import tempfile
from os.path import join as pjoin

import pandas as pd
import pysam
from ugbio_core.logger import logger


def _create_variant_bed(merged_vcf, bed_file):
    """Create a BED file from VCF variants."""
    cmd_bcftools = ["bcftools", "query", "-f", "%CHROM\t%POS\t%END\n", merged_vcf]
    cmd_awk = ["awk", 'BEGIN{OFS="\t"} {print $1, $2, $3+1}']
    with open(bed_file, "w") as out_file:
        p1 = subprocess.Popen(cmd_bcftools, stdout=subprocess.PIPE)
        p2 = subprocess.Popen(cmd_awk, stdin=p1.stdout, stdout=out_file)
        p1.stdout.close()
        p2.communicate()
        p1.wait()


def _find_closest_tandem_repeats(bed1, bed2, output_file):
    """Find closest tandem repeats for each variant."""
    cmd_bedtools = ["bedtools", "closest", "-D", "ref", "-a", bed1, "-b", bed2]
    cmd_cut = ["cut", "-f1-2,5-8"]
    with open(output_file, "w") as out_file:
        p1 = subprocess.Popen(cmd_bedtools, stdout=subprocess.PIPE)
        p2 = subprocess.Popen(cmd_cut, stdin=p1.stdout, stdout=out_file)
        p1.stdout.close()
        p2.communicate()
        p1.wait()


def _process_tandem_repeat_data(tr_tsv_file):
    """Process tandem repeat data and calculate metrics."""
    df_tr_info = pd.read_csv(tr_tsv_file, sep="\t", header=None)
    df_tr_info.columns = ["chrom", "pos", "TR_START", "TR_END", "TR_SEQ", "TR_DISTANCE"]
    df_tr_info["TR_LENGTH"] = df_tr_info["TR_END"] - df_tr_info["TR_START"]

    extracted = df_tr_info["TR_SEQ"].str.extract(r"\((\w+)\)")
    df_tr_info["TR_SEQ_UNIT_LENGTH"] = extracted[0].str.len()

    if df_tr_info["TR_SEQ_UNIT_LENGTH"].isna().any():
        logger.warning(
            "Some TR_SEQ values did not match the expected pattern '(unit)'. "
            "Setting TR_SEQ_UNIT_LENGTH to 0 for these rows."
        )
        df_tr_info["TR_SEQ_UNIT_LENGTH"] = df_tr_info["TR_SEQ_UNIT_LENGTH"].fillna(0).astype(int)

    df_tr_info.to_csv(tr_tsv_file, sep="\t", header=None, index=False)


def _prepare_annotation_files(tmpdir, tr_tsv_file):
    """Prepare files for VCF annotation."""
    sorted_tsv = pjoin(tmpdir, "merged_vcf.tmp.TRdata.sorted.tsv")
    cmd = ["sort", "-k1,1", "-k2,2n", tr_tsv_file]
    with open(sorted_tsv, "w") as out_file:
        subprocess.check_call(cmd, stdout=out_file)

    gz_tsv = sorted_tsv + ".gz"
    cmd = ["bgzip", "-c", sorted_tsv]
    with open(gz_tsv, "wb") as out_file:
        subprocess.check_call(cmd, stdout=out_file)

    cmd = ["tabix", "-s", "1", "-b", "2", "-e", "2", gz_tsv]
    subprocess.check_call(cmd)

    hdr_txt = [
        '##INFO=<ID=TR_START,Number=1,Type=String,Description="Closest tandem Repeat Start">',
        '##INFO=<ID=TR_END,Number=1,Type=String,Description="Closest Tandem Repeat End">',
        '##INFO=<ID=TR_SEQ,Number=1,Type=String,Description="Closest Tandem Repeat Sequence">',
        '##INFO=<ID=TR_DISTANCE,Number=1,Type=String,Description="Closest Tandem Repeat Distance">',
        '##INFO=<ID=TR_LENGTH,Number=1,Type=String,Description="Closest Tandem Repeat total length">',
        '##INFO=<ID=TR_SEQ_UNIT_LENGTH,Number=1,Type=String,Description="Closest Tandem Repeat unit length">',
    ]
    hdr_file = pjoin(tmpdir, "tr_hdr.txt")
    with open(hdr_file, "w") as f:
        f.writelines(line + "\n" for line in hdr_txt)

    return gz_tsv, hdr_file


def integrate_tandem_repeat_features(merged_vcf, ref_tr_file, out_dir):
    with tempfile.TemporaryDirectory(dir=out_dir) as tmpdir:
        # Create variant BED file
        bed1 = pjoin(tmpdir, "merged_vcf.tmp.bed")
        _create_variant_bed(merged_vcf, bed1)

        # Sort reference tandem repeat file
        ref_tr_file_sorted = pjoin(tmpdir, "ref_tr_file.sorted.bed")
        cmd = ["bedtools", "sort", "-i", ref_tr_file]
        with open(ref_tr_file_sorted, "w") as sorted_file:
            subprocess.check_call(cmd, stdout=sorted_file)

        # Find closest tandem repeats
        bed1_with_closest_tr_tsv = pjoin(tmpdir, "merged_vcf.tmp.TRdata.tsv")
        _find_closest_tandem_repeats(bed1, ref_tr_file_sorted, bed1_with_closest_tr_tsv)

        # Process tandem repeat data
        _process_tandem_repeat_data(bed1_with_closest_tr_tsv)

        # Prepare annotation files
        gz_tsv, hdr_file = _prepare_annotation_files(tmpdir, bed1_with_closest_tr_tsv)

        # Annotate VCF
        merged_vcf_with_tr_info = pjoin(out_dir, os.path.basename(merged_vcf).replace(".vcf.gz", ".tr_info.vcf.gz"))
        cmd = [
            "bcftools",
            "annotate",
            "-Oz",
            "-o",
            merged_vcf_with_tr_info,
            "-a",
            gz_tsv,
            "-h",
            hdr_file,
            "-c",
            "CHROM,POS,INFO/TR_START,INFO/TR_END,INFO/TR_SEQ,INFO/TR_DISTANCE,INFO/TR_LENGTH,INFO/TR_SEQ_UNIT_LENGTH",
            merged_vcf,
        ]
        subprocess.check_call(cmd)

    pysam.tabix_index(merged_vcf_with_tr_info, preset="vcf", min_shift=0, force=True)
    return merged_vcf_with_tr_info
