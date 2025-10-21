import os
import subprocess
import tempfile
from os.path import join as pjoin

import pandas as pd
import pysam
from ugbio_core.logger import logger


def integrate_tandem_repeat_features(merged_vcf, ref_tr_file, out_dir):
    # Use a temporary directory for all intermediate files
    with tempfile.TemporaryDirectory(dir=out_dir) as tmpdir:
        # generate tandem repeat info
        bed1 = pjoin(tmpdir, "merged_vcf.tmp.bed")
        cmd = (
            f"bcftools query -f '%CHROM\t%POS\t%END\n' {merged_vcf} | "
            f"awk 'BEGIN{{OFS=\"\t\"}} {{print $1, $2, $3+1}}' > {bed1}"
        )
        subprocess.check_call(cmd, shell=True)  # noqa: S602

        # sort the reference tandem repeat file
        ref_tr_file_sorted = pjoin(tmpdir, "ref_tr_file.sorted.bed")
        cmd = ["bedtools", "sort", "-i", ref_tr_file]
        with open(ref_tr_file_sorted, "w") as sorted_file:
            subprocess.check_call(cmd, stdout=sorted_file)
        # find closest tandem-repeat for each variant
        bed2 = ref_tr_file_sorted
        bed1_with_closest_tr_tsv = pjoin(tmpdir, "merged_vcf.tmp.TRdata.tsv")
        cmd_bedtools = ["bedtools", "closest", "-D", "ref", "-a", bed1, "-b", bed2]
        cmd_cut = ["cut", "-f1-2,5-8"]
        with open(bed1_with_closest_tr_tsv, "w") as out_file:
            p1 = subprocess.Popen(cmd_bedtools, stdout=subprocess.PIPE)
            p2 = subprocess.Popen(cmd_cut, stdin=p1.stdout, stdout=out_file)
            p1.stdout.close()
            p2.communicate()
            p1.wait()

        df_tr_info = pd.read_csv(bed1_with_closest_tr_tsv, sep="\t", header=None)
        df_tr_info.columns = ["chrom", "pos", "TR_START", "TR_END", "TR_SEQ", "TR_DISTANCE"]
        df_tr_info["TR_LENGTH"] = df_tr_info["TR_END"] - df_tr_info["TR_START"]
        # Extract repeat unit length, handle cases where pattern does not match
        extracted = df_tr_info["TR_SEQ"].str.extract(r"\((\w+)\)")
        df_tr_info["TR_SEQ_UNIT_LENGTH"] = extracted[0].str.len()
        # Fill NaN values with 0 and log a warning if any were found
        if df_tr_info["TR_SEQ_UNIT_LENGTH"].isna().any():
            logger.warning(
                "Some TR_SEQ values did not match the expected pattern '(unit)'. "
                "Setting TR_SEQ_UNIT_LENGTH to 0 for these rows."
            )
            df_tr_info["TR_SEQ_UNIT_LENGTH"] = df_tr_info["TR_SEQ_UNIT_LENGTH"].fillna(0).astype(int)
        df_tr_info.to_csv(bed1_with_closest_tr_tsv, sep="\t", header=None, index=False)

        sorted_tsv = pjoin(tmpdir, "merged_vcf.tmp.TRdata.sorted.tsv")
        cmd = ["sort", "-k1,1", "-k2,2n", bed1_with_closest_tr_tsv]
        with open(sorted_tsv, "w") as out_file:
            subprocess.check_call(cmd, stdout=out_file)
        gz_tsv = sorted_tsv + ".gz"
        cmd = ["bgzip", "-c", sorted_tsv]
        with open(gz_tsv, "wb") as out_file:
            subprocess.check_call(cmd, stdout=out_file)
        cmd = ["tabix", "-s", "1", "-b", "2", "-e", "2", gz_tsv]
        subprocess.check_call(cmd)

        # integrate tandem repeat info into the merged vcf file
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
