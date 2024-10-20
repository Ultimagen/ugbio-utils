# noqa: N999
import argparse
import logging
import os
import re
import sys
import warnings
from os.path import join as pjoin

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from ugbio_core.logger import logger
from ugbio_core.pileuptofreq import create_frequncies_from_pileup

warnings.filterwarnings("ignore")

sns.set_context("talk")
sns.set_style("white")


def mpileup_to_freq(pileup_file, outfile):
    """
    Convert mpileup file to frequencies counts
    """
    # mpileup to frequencies counts
    df_freq = create_frequncies_from_pileup(pileup_file)
    print(df_freq.head(10))
    # consider SNPs only
    pattern = "|".join([re.escape(sub) for sub in ["Deletion", "+", "-"]])
    df_freq = df_freq[~df_freq["Base"].str.contains(pattern, case=False)]
    df_freq = df_freq[df_freq["Base"] != df_freq["Ref"]]
    # add Allele Frequency column
    df_freq["AF"] = df_freq["Count"].astype(int) / df_freq["Depth"].astype(int)
    # convert to bed
    df_freq[["Chrom", "Pos", "Pos", "AF"]].to_csv(outfile, sep="\t", header=None, index=None)


def get_neutral_af(snp_bed_file, cnvs_file, outfile) -> pd.DataFrame:
    cmd = f"bedtools intersect -v -a {snp_bed_file} -b {cnvs_file} > {outfile}"
    os.system(cmd)  # noqa: S605
    df_neutral_af = pd.read_csv(outfile, header=None, sep="\t")
    df_neutral_af.columns = ["Chrom", "Start", "End", "AF"]
    return df_neutral_af


def plot_neutral_af(df_neutral_af, sample_name, outfile):
    plt.figure()
    df_neutral_af["AF"].hist(bins=50)
    xlabels = np.arange(0, 1.1, 0.1)
    plt.xticks(xlabels)
    plt.xlabel("Allele Frequency")
    plt.ylabel("Count (logscale)")
    plt.title(f"{sample_name} : Allele Frequency distribution in neutral regions")
    plt.yscale("log")
    plt.savefig(outfile, dpi=300, bbox_inches="tight")


def run(argv):
    """
    Runs the plot_FREEC_neutral_AF.py script to generate histogram of AF of neutral (non-CNV)
    locations across the sample.
    input arguments:
    --mpileup: input mpileup file.
    --cnvs_file: input bed file holding the called CNVs.
    --out_directory: output directory
    --sample_name: sample name
    output files:
    AF histogram: <out_directory>/<mpileup basename>.neutralAF.hist.jpeg.jpeg
        shows histogram of AFof neutral (non-CNV) locations across the sample.
    AF bed file: <out_directory>/<mpileup basename>.freq.SNP.neutral.bed
    """
    parser = argparse.ArgumentParser(
        prog="plot_FREEC_neutral_AF.py",
        description="generate histogram of AF of neutral (non-CNV) locations across the sample",
    )

    parser.add_argument(
        "--mpileup", help="input mpileup file as generated from controlFREEC pipeline.", required=True, type=str
    )
    parser.add_argument("--cnvs_file", help="input bed file holding the sample's called CNVs", required=True, type=str)
    parser.add_argument("--out_directory", help="output directory", required=True, type=str)
    parser.add_argument("--sample_name", help="sample name", required=True, type=str)
    parser.add_argument("--verbosity", help="Verbosity: ERROR, WARNING, INFO, DEBUG", required=False, default="INFO")

    args = parser.parse_args(argv[1:])
    logger.setLevel(getattr(logging, args.verbosity))

    pileup_file = args.mpileup
    basename = os.path.basename(pileup_file)
    outdir = args.out_directory
    logger.info(f"file will be written to {outdir}")

    # convert mpileup to SNP bed file with AF
    snp_bed_file = pjoin(outdir, f"{basename}.freq.SNP.bed")
    mpileup_to_freq(pileup_file, snp_bed_file)

    # get neutral (non-CNV) AF
    neutral_snp_bed_file = pjoin(outdir, f"{basename}.freq.SNP.neutral.bed")
    df_neutral_af = get_neutral_af(snp_bed_file, args.cnvs_file, neutral_snp_bed_file)

    # plot neutral AF histogram
    neutral_snp_hist_file = pjoin(outdir, f"{basename}.freq.SNP.neutral.hist.jpeg")
    plot_neutral_af(df_neutral_af, args.sample_name, neutral_snp_hist_file)
    logger.info(f"out hist file : {neutral_snp_hist_file}")


def main():
    run(sys.argv)


if __name__ == "__main__":
    main()
