# noqa: N999
import argparse
import logging
import sys
import warnings
from os.path import join as pjoin

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from ugbio_core.logger import logger

warnings.filterwarnings("ignore")

sns.set_context("talk")
sns.set_style("white")


def read_ratio_file(file):
    df_freec_ratio = pd.read_csv(file, sep="\t")
    df_freec_ratio = df_freec_ratio.sort_values(by=["Chromosome", "Start"])
    df_freec_ratio["End"] = df_freec_ratio["Start"] + 999
    return df_freec_ratio


def plot_ratio_values(df, sample_name, outdir):
    plt.figure(figsize=(20, 4))
    plt.scatter(np.arange(1, len(df["Ratio"].to_list()[0::50]) + 1), df["Ratio"].to_list()[0::50], alpha=0.05)
    plt.axhline(y=1, color="grey", linestyle="-")
    # plt.ylim([-1,3])
    plt.xlabel("location along the genome")
    plt.ylabel("fold-change")
    plt.title(f"{sample_name} FREEC Fold-change along the genome")
    out_file = pjoin(outdir, f"{sample_name}.fold_change.jpeg")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    return out_file


def run(argv):
    """
    Runs the plot_FREEC_fold_change.py script to generate fold-change (as outputted from ControlFREEC) of tumor/normal
    along the genome.
    input arguments:
    --ratio_file: input ratio.txt file as outputted from controlFREEC pipeline.
    --out_directory: output directory
    --sample_name: sample name
    output files:
    Fold-Change plot: <out_directory>/<samplename>.fold_change.jpeg
        shows fold-change (as outputted from ControlFREEC) of tumor/normal along the genome.
    """
    parser = argparse.ArgumentParser(
        prog="plot_FREEC_fold_change.py",
        description="generate fold-change (as outputted from ControlFREEC) of tumor/normal along the genome.",
    )

    parser.add_argument(
        "--ratio_file", help="ratio.txt file as outputted from controlFREEC pipeline", required=True, type=str
    )
    parser.add_argument("--out_directory", help="output directory", required=True, type=str)
    parser.add_argument("--sample_name", help="sample name", required=True, type=str)
    parser.add_argument("--verbosity", help="Verbosity: ERROR, WARNING, INFO, DEBUG", required=False, default="INFO")

    args = parser.parse_args(argv[1:])
    logger.setLevel(getattr(logging, args.verbosity))

    ratio_file = args.ratio_file
    sample_name = args.sample_name
    outdir = args.out_directory
    logger.info(f"file will be written to {outdir}")

    df_freec_ratio = read_ratio_file(ratio_file)
    outfile = plot_ratio_values(df_freec_ratio, sample_name, outdir)
    logger.info(f"output file: {outfile}")


def main():
    run(sys.argv)


if __name__ == "__main__":
    main()
