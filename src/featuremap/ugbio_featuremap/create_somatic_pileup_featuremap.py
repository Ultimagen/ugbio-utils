import argparse
import logging
import os
import subprocess
import sys
import tempfile
from os.path import join as pjoin

import pandas as pd
from ugbio_core.logger import logger
from ugbio_core.vcf_utils import VcfUtils
from ugbio_core.vcfbed import vcftools

vu = VcfUtils()


def integrate_tandem_repeat_features(merged_vcf, ref_tr_file, out_dir):
    # Use a temporary directory for all intermediate files
    with tempfile.TemporaryDirectory(dir=out_dir) as tmpdir:
        # generate tandem repeat info
        df_merged_vcf = vcftools.get_vcf_df(merged_vcf)
        df_merged_vcf.insert(
            2, "end", df_merged_vcf["pos"] + 1
        )  # TBD: get the actual end coordinate when the variant is not SNV (Insertion).
        bed1 = pjoin(tmpdir, "merged_vcf.tmp.bed")
        df_merged_vcf[["chrom", "pos", "end"]].to_csv(bed1, sep="\t", header=None, index=False)
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
        df_tr_info.columns = ["chrom", "pos", "TR_start", "TR_end", "TR_seq", "TR_distance"]
        df_tr_info["TR_length"] = df_tr_info["TR_end"] - df_tr_info["TR_start"]
        # Extract repeat unit length, handle cases where pattern does not match
        extracted = df_tr_info["TR_seq"].str.extract(r"\((\w+)\)")
        df_tr_info["TR_seq_unit_length"] = extracted[0].str.len()
        # Fill NaN values with 0 and log a warning if any were found
        if df_tr_info["TR_seq_unit_length"].isna().any():
            logger.warning(
                "Some TR_seq values did not match the expected pattern '(unit)'. "
                "Setting TR_seq_unit_length to 0 for these rows."
            )
            df_tr_info["TR_seq_unit_length"] = df_tr_info["TR_seq_unit_length"].fillna(0).astype(int)
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
            '##INFO=<ID=TR_start,Number=1,Type=String,Description="Closest tandem Repeat Start">',
            '##INFO=<ID=TR_end,Number=1,Type=String,Description="Closest Tandem Repeat End">',
            '##INFO=<ID=TR_seq,Number=1,Type=String,Description="Closest Tandem Repeat Sequence">',
            '##INFO=<ID=TR_distance,Number=1,Type=String,Description="Closest Tandem Repeat Distance">',
            '##INFO=<ID=TR_length,Number=1,Type=String,Description="Closest Tandem Repeat total length">',
            '##INFO=<ID=TR_seq_unit_length,Number=1,Type=String,Description="Closest Tandem Repeat unit length">',
        ]
        hdr_file = pjoin(tmpdir, "tr_hdr.txt")
        with open(hdr_file, "w") as f:
            f.writelines(line + "\n" for line in hdr_txt)
        merged_vcf_with_tr_info = merged_vcf.replace(".vcf.gz", ".tr_info.vcf.gz")
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
            "CHROM,POS,INFO/TR_start,INFO/TR_end,INFO/TR_seq,INFO/TR_distance,INFO/TR_length,INFO/TR_seq_unit_length",
            merged_vcf,
        ]
        subprocess.check_call(cmd)
    return merged_vcf_with_tr_info


def merge_vcf_files(tumor_vcf, normal_vcf, out_merged_vcf, n_cpu: int | None = None):
    """
    Merge tumor and normal VCF files into a single VCF file.

    Parameters
    ----------
    tumor_vcf : str
        Path to the tumor VCF file with INFO fields moved to FORMAT.
    normal_vcf : str
        Path to the normal VCF file with INFO fields moved to FORMAT.
    out_merged_vcf : str
        Path to the output merged VCF file.
    n_cpu: int
        Number of CPU to use in merge and view

    Returns
    -------
    str
        Path to the output merged VCF file with tumor-PASS variants only.
    """
    if n_cpu is None:
        n_cpu = os.cpu_count()
    # merging T-N VCF files - this results with records from both tumor and normal VCF files
    cmd_merge = [
        "bcftools",
        "merge",
        "--threads",
        str(n_cpu),
        "-m",
        "none",
        "--force-samples",
        "-Oz",
        "-o",
        out_merged_vcf,
        tumor_vcf,
        normal_vcf,
    ]
    logger.debug(" ".join(cmd_merge))
    subprocess.check_call(cmd_merge)
    vu.index_vcf(out_merged_vcf)

    vu.view_vcf(
        out_merged_vcf, out_merged_vcf.replace(".vcf.gz", ".tumor_PASS.vcf.gz"), extra_args="-f PASS", n_threads=n_cpu
    )
    # Index the filtered VCF
    vu.index_vcf(out_merged_vcf.replace(".vcf.gz", ".tumor_PASS.vcf.gz"))

    return out_merged_vcf.replace(".vcf.gz", ".tumor_PASS.vcf.gz")


def __parse_args(argv: list[str]) -> argparse.Namespace:
    """
    Parse command line arguments.

    Parameters
    ----------
    argv : list of str
        Command line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        prog="create_somatic_pileup_featuremap.py",
        description=run.__doc__,
    )
    parser.add_argument("--tumor_vcf", help="tumor vcf file", required=True, type=str)
    parser.add_argument("--normal_vcf", help="normal vcf file", required=True, type=str)
    parser.add_argument("--sample_name", help="sample_name", required=True, type=str)
    parser.add_argument("--cpu", help="number of CPU to use", required=False, type=int, default=8)
    parser.add_argument(
        "--out_directory",
        help="out directory where intermediate and output files will be saved."
        " if not supplied all files will be written to current directory",
        required=False,
        type=str,
        default=".",
    )
    parser.add_argument(
        "--filter_for_tumor_pass_variants",
        help="If set, the output VCF will only contain tumor-PASS variants.",
        action="store_true",
        default=False,
    )
    return parser.parse_args(argv[1:])


def run(argv):
    """
    Merge two VCF files (tumor and normal) into a single VCF file.

    The output VCF file will have all tumor records merged with corresponding normal records,
    with INFO fields moved to FORMAT.

    Parameters
    ----------
    argv : list of str
        Command line arguments.

    Returns
    -------
    None
    """
    args = __parse_args(argv)
    logger.setLevel(logging.DEBUG)
    for handler in logger.handlers:
        handler.setLevel(logging.DEBUG)

    logger.info(f"Output directory: {args.out_directory}")

    # Create output directory if it doesn't exist
    if not os.path.exists(args.out_directory):
        os.makedirs(args.out_directory)
        logger.info(f"Created output directory: {args.out_directory}")
    created_files = []
    # Set up the output VCF file path
    out_merged_vcf = pjoin(args.out_directory, f"{args.sample_name}.tumor_normal.merged.vcf.gz")
    logger.info(f"Output merged VCF file: {out_merged_vcf}")

    # Merge the tumor and normal VCF files into a single VCF file
    if args.filter_for_tumor_pass_variants:
        logger.info("Adding SingleRead filter to the tumor file")
        out_add_filter_vcf = pjoin(
            args.out_directory, os.path.basename(args.tumor_vcf).replace(".vcf.gz", ".with_sr_filter.vcf.gz")
        )
        vu.filter_vcf(
            input_vcf=args.tumor_vcf,
            output_vcf=out_add_filter_vcf,
            filter_name="SingleRead",
            exclude_expression="sum(FMT/FILT)<=1",
            n_threads=args.cpu,
        )
        logger.info("Adding SingleRead filter to the tumor file: done")
        created_files.append(out_add_filter_vcf)
        logger.info("Filtering for tumor-PASS variants only")
        tumor_vcf = out_add_filter_vcf.replace(".vcf.gz", ".tumor_PASS.vcf.gz")
        vu.view_vcf(
            input_vcf=out_add_filter_vcf,
            output_vcf=tumor_vcf,
            n_threads=args.cpu,
            extra_args="-f PASS",
        )
        vu.index_vcf(tumor_vcf)
        logger.info("Filtering for tumor-PASS variants only: done")
        created_files.append(tumor_vcf)
        created_files.append(tumor_vcf + ".tbi")
    else:
        tumor_vcf = args.tumor_vcf
        logger.info("No filtering for tumor-PASS variants. Merging all records from both VCF files")
    unfiltered_normal_vcf = pjoin(
        args.out_directory, os.path.basename(args.normal_vcf).replace(".vcf.gz", ".unfiltered.vcf.gz")
    )
    vu.remove_filter_annotations(args.normal_vcf, unfiltered_normal_vcf, args.cpu)
    vu.index_vcf(unfiltered_normal_vcf)
    created_files.append(unfiltered_normal_vcf)
    created_files.append(unfiltered_normal_vcf + ".tbi")
    out_merged_vcf_tumor_pass = merge_vcf_files(tumor_vcf, unfiltered_normal_vcf, out_merged_vcf)
    logger.info(f"Merged VCF file created: {out_merged_vcf}")
    logger.info(f"Merged VCF tumor-PASS file created: {out_merged_vcf_tumor_pass}")

    for f in created_files:
        os.remove(f)


def main():
    run(sys.argv)


if __name__ == "__main__":
    main()
