import argparse
import logging
import os
import subprocess
import sys
from os.path import join as pjoin

import pysam
from ugbio_core.logger import logger
from ugbio_core.vcf_utils import VcfUtils

vu = VcfUtils()
created_files = []


def add_single_read_filter(input_vcf: str, output_directory: str, n_threads: int = 1) -> str:
    """
    Add SingleRead filter to a VCF file.

    Parameters
    ----------
    input_vcf : str
        Path to the input VCF file.
    output_directory : str
        Output directory for the filtered VCF file.
    n_threads : int, optional
        Number of threads to use (default is 1).

    Returns
    -------
    str
        Path to the output VCF file with SingleRead filter applied.
    """
    logger.info("Adding SingleRead filter to the tumor file")
    out_add_filter_vcf = pjoin(
        output_directory, os.path.basename(input_vcf).replace(".vcf.gz", "") + ".with_sr_filter.vcf.gz"
    )
    vu.filter_vcf(
        input_vcf=input_vcf,
        output_vcf=out_add_filter_vcf,
        filter_name="SingleRead",
        exclude_expression="sum(FMT/FILT)<2",
        n_threads=n_threads,
    )
    vu.index_vcf(out_add_filter_vcf)
    logger.info("Adding SingleRead filter to the tumor file: done")
    created_files.append(out_add_filter_vcf)
    created_files.append(f"{out_add_filter_vcf}.tbi")
    return out_add_filter_vcf


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

    # Check if tumor and normal VCFs have the same sample name
    with pysam.VariantFile(tumor_vcf) as tumor_vcf_in, pysam.VariantFile(normal_vcf) as normal_vcf_in:
        tumor_sample_name = list(tumor_vcf_in.header.samples)[0]
        normal_sample_name = list(normal_vcf_in.header.samples)[0]

    # merging T-N VCF files - this results with records from both tumor and normal VCF files
    out_merged_full_vcf = out_merged_vcf.replace(".vcf.gz", "") + ".full.vcf.gz"
    cmd_merge = [
        "bcftools",
        "merge",
        "--threads",
        str(n_cpu),
        "-m",
        "none",
        "-Oz",
        "-o",
        out_merged_full_vcf,
        tumor_vcf,
        normal_vcf,
    ]
    if tumor_sample_name == normal_sample_name:
        logger.warning(
            f"Tumor and normal VCFs have the same sample name ({tumor_sample_name}). "
            "Using --force-samples to allow merging."
        )
        cmd_merge.insert(2, "--force-samples")
    logger.debug(" ".join(cmd_merge))
    subprocess.check_call(cmd_merge)
    vu.index_vcf(out_merged_full_vcf)

    created_files.append(out_merged_full_vcf)
    created_files.append(out_merged_full_vcf + ".tbi")

    # Keep only records that are present in the tumor VCF file
    vu.view_vcf(
        input_vcf=out_merged_full_vcf,
        output_vcf=out_merged_vcf,
        n_threads=n_cpu,
        extra_args="-i 'COUNT(FORMAT/RL[0:0]) > 0'",
    )
    vu.index_vcf(out_merged_vcf)


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
        "--keep-non-pass-tumor-candidates",
        help="If set, the output VCF will also contain non-PASS variants.",
        action="store_true",
        default=False,
    )
    return parser.parse_args(argv[1:])


def run(argv):
    """
    Merge two VCF files (tumor and normal) into a single VCF file.

    The output VCF file will have tumor records (filtered for SingleRead variants)
    merged with corresponding normal records.
    If the `--keep-non-pass-tumor-candidates` flag is set, non-PASS variants will also be included in the output.

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

    # Add SingleRead filter to the tumor VCF file
    out_add_filter_vcf = add_single_read_filter(args.tumor_vcf, args.out_directory, args.cpu)

    # Merge the tumor and normal VCF files into a single VCF file
    if args.keep_non_pass_tumor_candidates:
        logger.info("including non-PASS tumor variants.")
        tumor_vcf = out_add_filter_vcf.replace(".with_sr_filter.vcf.gz", ".no_sr.vcf.gz")
        vu.view_vcf(
            input_vcf=out_add_filter_vcf,
            output_vcf=tumor_vcf,
            n_threads=args.cpu,
            extra_args="-i 'FILTER!~\"SingleRead\"'",
        )
        vu.index_vcf(tumor_vcf)
        logger.info("No filtering for tumor-PASS variants. Filtering only for SingleRead variants.")
        created_files.append(tumor_vcf)
        created_files.append(tumor_vcf + ".tbi")
    else:
        logger.info("Filtering for tumor-PASS variants only.")
        tumor_vcf = out_add_filter_vcf.replace(".vcf.gz", ".tumor_PASS.vcf.gz")
        vu.view_vcf(
            input_vcf=out_add_filter_vcf,
            output_vcf=tumor_vcf,
            n_threads=args.cpu,
            extra_args="-f PASS -i 'FILTER!~\"SingleRead\"'",
        )
        vu.index_vcf(tumor_vcf)
        logger.info("Filtering for tumor-PASS variants only: done")
        created_files.append(tumor_vcf)
        created_files.append(tumor_vcf + ".tbi")

    unfiltered_normal_vcf = pjoin(
        args.out_directory, os.path.basename(args.normal_vcf).replace(".vcf.gz", ".unfiltered.vcf.gz")
    )
    vu.remove_filter_annotations(args.normal_vcf, unfiltered_normal_vcf, args.cpu)
    vu.index_vcf(unfiltered_normal_vcf)
    created_files.append(unfiltered_normal_vcf)
    created_files.append(unfiltered_normal_vcf + ".tbi")

    # Set up the output VCF file path
    out_merged_tmp_vcf = pjoin(args.out_directory, f"{args.sample_name}.tumor_normal.merged.tmp.vcf.gz")
    merge_vcf_files(tumor_vcf, unfiltered_normal_vcf, out_merged_tmp_vcf)
    created_files.append(out_merged_tmp_vcf)
    created_files.append(out_merged_tmp_vcf + ".tbi")

    # Add tumor sample name to merged VCF header
    with pysam.VariantFile(out_merged_tmp_vcf) as vcf_in:
        tumor_sample_name = list(vcf_in.header.samples)[0]
    # Add header line using pysam
    out_merged_vcf = out_merged_tmp_vcf.replace(".tmp.vcf.gz", ".vcf.gz")
    with pysam.VariantFile(out_merged_tmp_vcf) as vcf_in:
        header = vcf_in.header.copy()
        header.add_line(f"##tumor_sample={tumor_sample_name}")
        with pysam.VariantFile(out_merged_vcf, "wz", header=header) as vcf_out:
            for rec in vcf_in:
                vcf_out.write(rec)
    vu.index_vcf(out_merged_vcf)
    logger.info(f"Added tumor sample header: ##tumor_sample={tumor_sample_name}")
    logger.info(f"Output merged VCF file: {out_merged_vcf}")

    # Clean up intermediate files
    for f in created_files:
        try:
            os.remove(f)
            logger.debug(f"Removed temporary file: {f}")
        except Exception as e:
            logger.warning(f"Could not remove temporary file {f}: {e}")


def main():
    run(sys.argv)


if __name__ == "__main__":
    main()
