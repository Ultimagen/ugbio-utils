import argparse
import logging
import os
import subprocess
import sys
from os.path import join as pjoin

from ugbio_core.logger import logger
from ugbio_core.vcf_utils import VcfUtils

vu = VcfUtils()


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
        out_add_filter_vcf = args.tumor_vcf.replace(".vcf.gz", ".with_sr_filter.vcf.gz")
        vu.filter_vcf(
            input_vcf=args.tumor_vcf,
            output_vcf=out_add_filter_vcf,
            filter_name="SingleRead",
            exclude_expression="sum(FMT/FILT)<=1",
            n_threads=args.cpu,
        )
        logger.info("Adding SingleRead filter to the tumor file: done")
        created_files.append(out_add_filter_vcf)
        logger.info("Filtering for tumor-PASS variants only.")
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
        logger.info("No filtering for tumor-PASS variants. Merging all records from both VCF files.")
    unfiltered_normal_vcf = args.normal_vcf.replace(".vcf.gz", ".unfiltered.vcf.gz")
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
