import argparse
import logging
import os
import re
import subprocess
import sys
from os.path import join as pjoin

import pandas as pd
import ugbio_cnv.convert_combined_cnv_results_to_vcf
from pyfaidx import Fasta
from ugbio_core.logger import logger

bedmap = "bedmap"


def run_cmd(cmd):
    logger.info(cmd)
    subprocess.run(cmd, shell=True, check=True)  # noqa: S602


def __parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="cnv_results_to_vcf.py", description="converts CNV calls in bed format to vcf."
    )

    parser.add_argument("--cnmops_cnv_calls", help="input bed file holding cn.mops CNV calls", required=True, type=str)
    parser.add_argument(
        "--cnvpytor_cnv_calls", help="input bed file holding cnvpytor CNV calls", required=True, type=str
    )
    parser.add_argument(
        "--del_jalign_merged_results", help="jaign results for Deletions in tsv format", required=True, type=str
    )
    parser.add_argument(
        "--deletions_length_cutoff", help="deletions length cutoff", required=False, type=int, default=3000
    )
    parser.add_argument(
        "--jalign_written_cutoff",
        help="minimal number of supporting jaligned reads for DEL",
        required=False,
        type=int,
        default=1,
    )
    parser.add_argument(
        "--distance_threshold",
        help="distance threshold for merging CNV segments",
        required=False,
        type=int,
        default=1500,
    )
    parser.add_argument(
        "--duplication_length_cutoff_for_cnmops_filter",
        help="duplication_length_cutoff_for_cnmops_filter",
        required=False,
        type=int,
        default=10000,
    )
    parser.add_argument("--ug_cnv_lcr", help="UG-CNV-LCR bed file", required=False, type=str)
    parser.add_argument("--ref_fasta", help="reference genome fasta file", required=True, type=str)
    parser.add_argument("--fasta_index", help="fasta.fai file", required=True, type=str)
    parser.add_argument("--out_directory", help="output directory", required=False, type=str)
    parser.add_argument("--sample_name", help="sample name", required=True, type=str)
    parser.add_argument("--verbosity", help="Verbosity: ERROR, WARNING, INFO, DEBUG", required=False, default="INFO")

    return parser.parse_args(argv[1:])


def calculate_gaps_count_per_cnv(df_cnmops_calls: pd.DataFrame, ref_fasta: str) -> pd.DataFrame:
    """
    Calculate the number of 'N' bases in each CNV call region from the reference genome FASTA file.

     Parameters
     ----------
     df_cnmops_calls : pandas.DataFrame
         DataFrame containing CNV calls with columns ['chrom', 'start', 'end'].
     ref_fasta : str
         Path to the hg38 reference genome FASTA file.

     Returns
     -------
     pandas.DataFrame
         Updated DataFrame with additional columns: 'N_count', 'len', and 'pN'.
    """
    if not os.path.exists(ref_fasta):
        raise FileNotFoundError(f"Fasta file {ref_fasta} does not exist.")

    genome = Fasta(ref_fasta, build_index=False, rebuild=False)

    n_count = []
    for index, row in df_cnmops_calls.iterrows():  # noqa: B007
        chrom = row["chrom"]
        start = row["start"]
        end = row["end"]
        # pyfaidx uses 0-based start, end-exclusive indexing
        seq = genome[chrom][start - 1 : end].seq  # Convert to 0-based
        cnv_n_count = seq.upper().count("N")
        n_count.append(cnv_n_count)

    df_cnmops_calls["N_count"] = n_count
    df_cnmops_calls["len"] = df_cnmops_calls["end"] - df_cnmops_calls["start"] + 1
    df_cnmops_calls["pN"] = df_cnmops_calls["N_count"] / df_cnmops_calls["len"]

    return df_cnmops_calls


def parse_cnmops_cnv_calls(cnmops_cnv_calls: str, out_directory: str, ref_fasta: str, pN: float = 0) -> str:  # noqa: N803
    """
    Parses cn.mops CNV calls from an input BED file.

    Parameters
    ----------
    cnmops_cnv_calls : str
        Path to the cn.mops CNV calls BED file.
    out_directory : str
        Output directory to store results.
    pN : float
        Threshold for filtering CNV calls based on the fraction of reference genome
        gaps (Ns) in the call region.

    Returns
    -------
    out_cnmops_cnvs : str
        Path to the output BED file with parsed CNV calls.

    """
    cnmops_cnv_calls_tmp_file = f"{pjoin(out_directory,os.path.basename(cnmops_cnv_calls))}.tmp"

    # remove all tags from cnmops cnv calls file:
    run_cmd(
        f"cat {cnmops_cnv_calls} | sed 's/UG-CNV-LCR//g' | sed 's/LEN//g' | sed 's/|//g' \
            > {cnmops_cnv_calls_tmp_file}"
    )

    df_cnmops_cnvs = pd.read_csv(cnmops_cnv_calls_tmp_file, sep="\t", header=None)
    df_cnmops_cnvs.columns = ["chrom", "start", "end", "CN"]
    df_cnmops_cnvs = calculate_gaps_count_per_cnv(df_cnmops_cnvs, ref_fasta)
    # Filter by pN value
    df_cnmops_cnvs = df_cnmops_cnvs[df_cnmops_cnvs["pN"] <= pN]

    return df_cnmops_cnvs


def get_dup_cnmops_cnv_calls(
    df_cnmops: pd.DataFrame, sample_name: str, out_directory: str, distance_threshold: int
) -> str:
    """
    Parameters
    ----------
    df_cnmops : pandas.DataFrame
        DataFrame holding cn.mops CNV calls.
    sample_name : str
        Sample name.
    out_directory : str
        Output folder to store results.
    distance_threshold : int
        Distance threshold for merging CNV segments.

    Returns
    -------
    str
        Path to the duplications called by cn.mops bed file.
    """
    # get duplications from cn.mops calls
    cnmops_cnvs_dup = pjoin(out_directory, f"{sample_name}.cnmops_cnvs.DUP.bed")
    # df_cnmops = pd.read_csv(cnmops_cnv_calls, sep="\t", header=None)
    # df_cnmops.columns = ["chrom", "start", "end", "CN"]
    df_cnmops["cn_numbers"] = [re.search(r"CN([\d\.]+)", item).group(1) for item in df_cnmops["CN"]]
    out_cnmops_cnvs_dup_calls = pjoin(out_directory, f"{sample_name}.cnmops_cnvs.DUP.calls.bed")
    neutral_cn = 2
    df_cnmops[df_cnmops["cn_numbers"].astype(float) > neutral_cn][["chrom", "start", "end", "CN"]].to_csv(
        out_cnmops_cnvs_dup_calls, sep="\t", header=None, index=False
    )

    if os.path.getsize(out_cnmops_cnvs_dup_calls) > 0:
        run_cmd(
            f"cat {out_cnmops_cnvs_dup_calls} | \
                bedtools merge -d {distance_threshold} -c 4 -o distinct -i - | \
                awk '$3-$2>=10000' | \
                sed 's/$/\\tDUP\\tcn.mops/' | \
                cut -f1,2,3,5,6,4 > {cnmops_cnvs_dup}"
        )

        df_cnmops_cnvs_dup = pd.read_csv(cnmops_cnvs_dup, sep="\t", header=None)
        df_cnmops_cnvs_dup.columns = ["chrom", "start", "end", "CN", "CNV_type", "source"]
        df_cnmops_cnvs_dup["copy_number"] = df_cnmops_cnvs_dup["CN"].apply(lambda x: x.replace("CN", ""))

        out_cnmops_cnvs_dup = pjoin(out_directory, f"{sample_name}.cnmops_cnvs.DUP.all_fields.bed")
        df_cnmops_cnvs_dup[["chrom", "start", "end", "CNV_type", "source", "copy_number"]].to_csv(
            out_cnmops_cnvs_dup, sep="\t", header=None, index=False
        )

        return out_cnmops_cnvs_dup
    else:
        logger.info("No duplications found in cn.mops CNV calls.")
        return ""


def parse_cnvpytor_cnv_calls(cnvpytor_cnv_calls: str, pN: float = 0) -> pd.DataFrame:  # noqa: N803
    """
    Parses cnvpytor CNV calls from a tsv file.

    Parameters
    ----------
    cnvpytor_cnv_calls : str
        Path to the cnvpytor CNV calls bed file.
    pN : float
        Threshold for filtering CNV calls based on the fraction of reference genome
        gaps (Ns) in call region.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing parsed CNV calls.
    """
    # Result is stored in tab separated files with following columns:
    # CNV type: "deletion" or "duplication",
    # CNV region (chr:start-end),
    # CNV size,
    # CNV level - read depth normalized to 1,
    # e-val1 -- e-value (p-value multiplied by genome size divided by bin size) calculated
    #           using t-test statistics between RD statistics in the region and global,
    # e-val2 -- e-value (p-value multiplied by genome size divided by bin size) from the probability of RD values within
    #           the region to be in the tails of a gaussian distribution of binned RD,
    # e-val3 -- same as e-val1 but for the middle of CNV,
    # e-val4 -- same as e-val2 but for the middle of CNV,
    # q0 -- fraction of reads mapped with q0 quality in call region,
    # pN -- fraction of reference genome gaps (Ns) in call region,
    # dG -- distance from closest large (>100bp) gap in reference genome.

    df_cnvpytor_cnvs = pd.read_csv(cnvpytor_cnv_calls, sep="\t", header=None)
    df_cnvpytor_cnvs.columns = [
        "cnv_type",
        "cnv_region",
        "len",
        "cnv_level",
        "e-val1",
        "e-val2",
        "e-val3",
        "e-val4",
        "q0",
        "pN",
        "dG",
    ]

    # Split cnv_region into 'chr', 'start', 'end'
    df_cnvpytor_cnvs[["chrom", "pos"]] = df_cnvpytor_cnvs["cnv_region"].str.split(":", expand=True)
    df_cnvpytor_cnvs[["start", "end"]] = df_cnvpytor_cnvs["pos"].str.split("-", expand=True)
    df_cnvpytor_cnvs = df_cnvpytor_cnvs.drop(columns="pos")
    df_cnvpytor_cnvs["start"] = df_cnvpytor_cnvs["start"].astype(int)
    df_cnvpytor_cnvs["end"] = df_cnvpytor_cnvs["end"].astype(int)

    # Filter by pN value
    df_cnvpytor_cnvs = df_cnvpytor_cnvs[df_cnvpytor_cnvs["pN"] <= 0]

    return df_cnvpytor_cnvs


def get_dup_cnvpytor_cnv_calls(df_cnvpytor_cnv_calls: pd.DataFrame, sample_name: str, out_directory: str) -> str:  # noqa: N803
    """
    Parameters
    ----------
    df_cnvpytor_cnv_calls : pandas.DataFrame
        DataFrame holding cnvpytor CNV calls.
    sample_name : str
        Sample name.
    out_directory : str
        Output folder to store results.

    Returns
    -------
    str
        Path to the duplications called by cnvpytor bed file.
    """
    cnvpytor_cnvs_dup = pjoin(out_directory, f"{sample_name}.cnvpytor_cnvs.DUP.bed")
    df_cnvpytor_cnv_calls_duplications = df_cnvpytor_cnv_calls[
        df_cnvpytor_cnv_calls["cnv_type"] == "duplication"
    ].copy()
    df_cnvpytor_cnv_calls_duplications["cnv_type"] = "DUP"
    df_cnvpytor_cnv_calls_duplications["copy_number"] = "DUP"
    df_cnvpytor_cnv_calls_duplications["source"] = "cnvpytor"

    if len(df_cnvpytor_cnv_calls_duplications) > 0:
        df_cnvpytor_cnv_calls_duplications[["chrom", "start", "end", "cnv_type", "source", "copy_number"]].to_csv(
            cnvpytor_cnvs_dup, sep="\t", header=None, index=False
        )
        return cnvpytor_cnvs_dup
    else:
        logger.info("No duplications found in cnvpytor CNV calls.")
        return ""


def process_del_jalign_results(
    del_jalign_results: str,
    sample_name: str,
    out_directory: str,
    ref_fasta: str,
    pN: float = 0,  # noqa: N803
    deletions_length_cutoff: int = 3000,
    jalign_written_cutoff: int = 1,
) -> str:
    """
    Processes jalign results for deletions and filters them.

    Parameters
    ----------
    del_jalign_results : str
        Jalign results for Deletions in tsv format.
    sample_name : str
        Sample name.
    out_directory : str
        Output folder to store results.
    ref_fasta : str
        Reference genome fasta file.
    pN : float, optional
        Threshold for filtering CNV calls based on the fraction of reference genome gaps (Ns) in the call region.
    deletions_length_cutoff : int, optional
        Deletions length cutoff.
    jalign_written_cutoff : int, optional
        Minimal number of supporting jaligned reads for DEL.

    Returns
    -------
    str
        Path to deletions called by cn.mops and cnvpytor bed file.
    """
    # reads jalign results
    df_cnmops_cnvpytor_del = pd.read_csv(del_jalign_results, sep="\t", header=None)
    df_cnmops_cnvpytor_del.columns = [
        "chrom",
        "start",
        "end",
        "CN",
        "jalign_written",
        "6",
        "7",
        "jdelsize_min",
        "jdelsize_max",
        "jdelsize_avg",
        "jumpland_min",
        "jumpland_max",
        "jumpland_avg",
    ]
    df_cnmops_cnvpytor_del["len"] = df_cnmops_cnvpytor_del["end"] - df_cnmops_cnvpytor_del["start"]
    df_cnmops_cnvpytor_del["source"] = df_cnmops_cnvpytor_del["CN"].apply(
        lambda x: "cn.mops" if pd.Series(x).str.contains("CN").any() else "cnvpytor"
    )
    df_cnmops_cnvpytor_del["CNV_type"] = "DEL"
    df_cnmops_cnvpytor_del["copy_number"] = df_cnmops_cnvpytor_del["CN"].apply(lambda x: x.replace("CN", ""))
    df_cnmops_cnvpytor_del["copy_number"] = df_cnmops_cnvpytor_del["copy_number"].apply(
        lambda x: "DEL" if pd.Series(x).str.contains("deletion").any() else x
    )
    df_cnmops_cnvpytor_del_filtered = df_cnmops_cnvpytor_del[
        (df_cnmops_cnvpytor_del["jalign_written"] >= jalign_written_cutoff)
        | (df_cnmops_cnvpytor_del["len"] > deletions_length_cutoff)
    ]

    df_cnmops_cnvpytor_del_filtered = calculate_gaps_count_per_cnv(df_cnmops_cnvpytor_del_filtered, ref_fasta)
    df_cnmops_cnvpytor_del_filtered = df_cnmops_cnvpytor_del_filtered[df_cnmops_cnvpytor_del_filtered["pN"] <= pN]

    out_del_jalign = pjoin(
        out_directory,
        f"{sample_name}.cnmops_cnvpytor.DEL.jalign_lt{str(jalign_written_cutoff)}_or_len_lt{str(deletions_length_cutoff)}.bed",
    )
    df_cnmops_cnvpytor_del_filtered[["chrom", "start", "end", "CNV_type", "source", "copy_number"]].to_csv(
        out_del_jalign, sep="\t", header=None, index=False
    )

    out_del_jalign_merged = pjoin(
        out_directory,
        f"{sample_name}.cnmops_cnvpytor.DEL.jalign_lt{str(jalign_written_cutoff)}_or_len_lt{str(deletions_length_cutoff)}.merged.bed",
    )
    run_cmd(
        f"cat {out_del_jalign} | bedtools sort -i - | \
            bedtools merge -c 4,5,6 -o distinct  -i -  > {out_del_jalign_merged}"
    )

    return out_del_jalign_merged


def get_cnmops_cnvpytor_common_del(del_candidates: str, sample_name: str, out_directory: str) -> str:
    """
    Get deletions called by both cn.mops and cnvpytor, regardless of jalign results.

    Parameters
    ----------
    del_candidates : str
        All deletions candidates (jalign results for Deletions in tsv format).
    sample_name : str
        Sample name.
    out_directory : str
        Output folder to store results.

    Returns
    -------
    str
        Path to deletions called by both cn.mops and cnvpytor bed file.
    """
    del_candidates_called_by_both_cnmops_cnvpytor = pjoin(
        out_directory, f"{sample_name}.del_candidates_called_by_both_cnmops_cnvpytor.bed"
    )
    run_cmd(
        f'cat {del_candidates} | cut -f1-4  | bedtools merge -c 4 -o distinct -i - | \
        grep -E "CN.*deletion|deletion.*CN" > {del_candidates_called_by_both_cnmops_cnvpytor} \
            || touch {del_candidates_called_by_both_cnmops_cnvpytor}'
    )

    if os.path.getsize(del_candidates_called_by_both_cnmops_cnvpytor) > 0:
        df_del_candidates_called_by_both_cnmops_cnvpytor = pd.read_csv(
            del_candidates_called_by_both_cnmops_cnvpytor, sep="\t", header=None
        )
        df_del_candidates_called_by_both_cnmops_cnvpytor.columns = ["chrom", "start", "end", "CN"]
        df_del_candidates_called_by_both_cnmops_cnvpytor["CNV_type"] = "DEL"
        df_del_candidates_called_by_both_cnmops_cnvpytor["source"] = "cn.mops,cnvpytor"

        copy_number_list = []
        for __index, row in df_del_candidates_called_by_both_cnmops_cnvpytor.iterrows():
            cn = row["CN"]
            cn_list = cn.split(",")
            copy_number_value = ""
            for val in cn_list:
                if "CN" in val:
                    copy_number_value = copy_number_value + f"{val.split('CN')[1]},"
            copy_number_value = copy_number_value + "DEL"
            copy_number_list.append(copy_number_value)
        df_del_candidates_called_by_both_cnmops_cnvpytor["copy_number"] = copy_number_list

        out_del_candidates_called_by_both_cnmops_cnvpytor = pjoin(
            out_directory, f"{sample_name}.del_candidates_called_by_both_cnmops_cnvpytor.all_fields.bed"
        )
        df_del_candidates_called_by_both_cnmops_cnvpytor[
            ["chrom", "start", "end", "CNV_type", "source", "copy_number"]
        ].to_csv(out_del_candidates_called_by_both_cnmops_cnvpytor, sep="\t", header=None, index=False)

        return out_del_candidates_called_by_both_cnmops_cnvpytor
    else:
        logger.info("No deletions found by both cn.mops and cnvpytor.")
        return ""


def run(argv):
    """
    Combine CNVs from cn.mops and cnvpytor using jalign results and convert them to VCF.

    Parameters
    ----------
    argv : list of str
        Command-line arguments.

    Input Arguments
    ---------------
    --cnmops_cnv_calls : str
        Input BED file holding cn.mops CNV calls.
    --cnvpytor_cnv_calls : str
        Input BED file holding cnvpytor CNV calls.
    --del_jalign_merged_results : str
        Jalign results for deletions in TSV format.

    Output Files
    ------------
    <sample_name>.cnmops_cnvpytor.cnvs.combined.bed : str
        Combined CNV calls called by cn.mops and cnvpytor.
    <sample_name>.cnmops_cnvpytor.cnvs.combined.UG-CNV-LCR_annotate.bed : str
        Combined CNV calls with UG-CNV-LCR annotation.
    """
    args = __parse_args(argv)
    logger.setLevel(getattr(logging, args.verbosity))

    out_directory = args.out_directory
    sample_name = args.sample_name
    # format cnvpytor cnv calls :
    df_cnmops_cnv_calls = parse_cnmops_cnv_calls(args.cnmops_cnv_calls, out_directory, args.ref_fasta)
    df_cnvpytor_cnv_calls = parse_cnvpytor_cnv_calls(args.cnvpytor_cnv_calls)

    ############################
    ### process DUPlications ###
    ############################
    out_cnmops_cnvs_dup = get_dup_cnmops_cnv_calls(
        df_cnmops_cnv_calls, sample_name, out_directory, args.distance_threshold
    )
    out_cnvpytor_cnvs_dup = get_dup_cnvpytor_cnv_calls(df_cnvpytor_cnv_calls, sample_name, out_directory)
    # merge duplications
    if not out_cnmops_cnvs_dup and not out_cnvpytor_cnvs_dup:
        logger.info("No duplications found in cn.mops and cnvpytor CNV calls.")
        cnmops_cnvpytor_merged_dup = ""
    else:
        cnmops_cnvpytor_merged_dup = pjoin(out_directory, f"{sample_name}.cnmops_cnvpytor.DUP.merged.bed")
        run_cmd(
            f"cat {out_cnmops_cnvs_dup} {out_cnvpytor_cnvs_dup} | bedtools sort -i - | \
            bedtools merge -c 4,5,6 -o distinct -i - > {cnmops_cnvpytor_merged_dup}"
        )

    ############################
    ###  process DELetions   ###
    ############################

    out_del_jalign_merged = process_del_jalign_results(
        args.del_jalign_merged_results,
        sample_name,
        out_directory,
        ref_fasta=args.ref_fasta,
        pN=0,
        deletions_length_cutoff=args.deletions_length_cutoff,
        jalign_written_cutoff=args.jalign_written_cutoff,
    )
    out_del_candidates_called_by_both_cnmops_cnvpytor = get_cnmops_cnvpytor_common_del(
        args.del_jalign_merged_results, sample_name, out_directory
    )
    # merge deletions
    out_del_calls = pjoin(
        out_directory,
        f"{sample_name}.cnmops_cnvpytor.DEL.jalign_lt{str(args.jalign_written_cutoff)}_or_len_lt{str(args.deletions_length_cutoff)}.called_by_both_cnmops_cnvpytor.bedtools_merge.bed",
    )
    run_cmd(
        f"cat {out_del_jalign_merged} {out_del_candidates_called_by_both_cnmops_cnvpytor} | \
            bedtools sort -i - | bedtools merge -c 4,5,6 -o distinct  -i - > {out_del_calls}"
    )

    # combine results
    out_cnvs_combined = pjoin(out_directory, f"{sample_name}.cnmops_cnvpytor.cnvs.combined.bed")
    run_cmd(f"cat {cnmops_cnvpytor_merged_dup} {out_del_calls} | bedtools sort -i - > {out_cnvs_combined}")
    logger.info(f"out_cnvs_combined: {out_cnvs_combined}")

    if args.ug_cnv_lcr:
        # annotate with ug-cnv-lcr if provided
        # result file should be in the following format:
        # ["chr", "start", "end", "CNV_type", "CNV_calls_source", "copy_number", "UG-CNV-LCR"]
        out_cnvs_combined_annotated = f"{out_cnvs_combined}.annotate.bed"
        run_cmd(
            f"bedmap --echo --echo-map-id-uniq --delim '\\t' --bases-uniq-f \
            {out_cnvs_combined} {args.ug_cnv_lcr} > {out_cnvs_combined_annotated}"
        )
        logger.info(f"out_cnvs_combined_annotated: {out_cnvs_combined_annotated}")

        overlap_filtration_cutoff = 0.5  # 50% overlap with LCR regions
        df_annotate_calls = pd.read_csv(out_cnvs_combined_annotated, sep="\t", header=None)
        df_annotate_calls.columns = [
            "chr",
            "start",
            "end",
            "CNV_type",
            "CNV_calls_source",
            "copy_number",
            "UG-CNV-LCR",
            "pUG-CNV-LCR_overlap",
        ]
        df_annotate_calls["LCR_label_value"] = df_annotate_calls.apply(
            lambda row: row["UG-CNV-LCR"] if row["pUG-CNV-LCR_overlap"] >= overlap_filtration_cutoff else ".", axis=1
        )

        df_annotate_calls[
            ["chr", "start", "end", "CNV_type", "CNV_calls_source", "copy_number", "LCR_label_value"]
        ].to_csv(out_cnvs_combined_annotated, sep="\t", header=None, index=False)
        logger.info(f"out_cnvs_combined_annotated: {out_cnvs_combined_annotated}")

    else:
        out_cnvs_combined_annotated = out_cnvs_combined

    # convert to vcf
    vcf_args = [
        "convert_combined_cnv_results_to_vcf",
        "--cnv_annotated_bed_file",
        out_cnvs_combined_annotated,
        "--fasta_index_file",
        args.fasta_index,
        "--out_directory",
        out_directory,
        "--sample_name",
        sample_name,
    ]
    print(vcf_args)
    out_cnvs_combined_annotated_vcf = ugbio_cnv.convert_combined_cnv_results_to_vcf.run(vcf_args)
    logger.info(f"out_cnvs_combined_annotated_vcf: {out_cnvs_combined_annotated_vcf}")


def main():
    run(sys.argv)


if __name__ == "__main__":
    main()
