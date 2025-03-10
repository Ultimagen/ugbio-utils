import argparse
import logging
import subprocess
import sys
from os.path import join as pjoin

import pandas as pd
import ugbio_cnv.convert_combined_cnv_results_to_vcf
from ugbio_core.logger import logger


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
        "--DEL_jalign_merged_results", help="jaign results for Deletions in tsv format", required=True, type=str
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
    parser.add_argument("--ug_cnv_lcr", help="UG-CNV-LCR bed file", required=True, type=str)
    parser.add_argument("--fasta_index", help="fasta.fai file", required=True, type=str)

    parser.add_argument("--out_directory", help="output directory", required=False, type=str)
    parser.add_argument("--sample_name", help="sample name", required=True, type=str)
    parser.add_argument("--verbosity", help="Verbosity: ERROR, WARNING, INFO, DEBUG", required=False, default="INFO")

    return parser.parse_args(argv[1:])


def get_cnmops_cnv_calls(cnmops_cnv_calls, sample_name, out_directory):
    cnmops_cnvs_DUP = pjoin(out_directory, f"{sample_name}.cnmops_cnvs.DUP.bed")  # noqa: N806
    run_cmd(
        f"cat {cnmops_cnv_calls} | awk -F \"N\" '$NF>2' | \
            bedtools merge -d 1500 -c 4 -o distinct -i - | \
            awk '$3-$2>=10000' | \
            sed 's/$/\\tDUP\\tcn.mops/' | \
            cut -f1,2,3,5,6,4 > {cnmops_cnvs_DUP}"
    )

    df_cnmops_cnvs_DUP = pd.read_csv(cnmops_cnvs_DUP, sep="\t", header=None)  # noqa: N806
    df_cnmops_cnvs_DUP.columns = ["chrom", "start", "end", "CN", "CNV_type", "source"]
    df_cnmops_cnvs_DUP["copy_number"] = df_cnmops_cnvs_DUP["CN"].apply(lambda x: x.replace("CN", ""))

    out_cnmops_cnvs_DUP = pjoin(out_directory, f"{sample_name}.cnmops_cnvs.DUP.all_fields.bed")  # noqa: N806
    df_cnmops_cnvs_DUP[["chrom", "start", "end", "CNV_type", "source", "copy_number"]].to_csv(
        out_cnmops_cnvs_DUP, sep="\t", header=None, index=False
    )

    return out_cnmops_cnvs_DUP


def get_cnvpytor_cnv_calls(cnvpytor_cnv_calls, sample_name, out_directory):
    cnvpytor_cnvs_DUP = pjoin(out_directory, f"{sample_name}.cnvpytor_cnvs.DUP.bed")  # noqa: N806
    run_cmd(f"cat {cnvpytor_cnv_calls} |  grep \"duplication\" | sed 's/$/\\tDUP\\tcnvpytor/'  > {cnvpytor_cnvs_DUP}")

    df_cnvpytor_cnvs_DUP = pd.read_csv(cnvpytor_cnvs_DUP, sep="\t", header=None)  # noqa: N806
    df_cnvpytor_cnvs_DUP.columns = ["chrom", "start", "end", "CN", "CNV_type", "source"]
    df_cnvpytor_cnvs_DUP["copy_number"] = "DUP"
    out_cnvpytor_cnvs_DUP = pjoin(out_directory, f"{sample_name}.cnvpytor_cnvs.DUP.all_fields.bed")  # noqa: N806
    df_cnvpytor_cnvs_DUP[["chrom", "start", "end", "CNV_type", "source", "copy_number"]].to_csv(
        out_cnvpytor_cnvs_DUP, sep="\t", header=None, index=False
    )

    return out_cnvpytor_cnvs_DUP


def process_DEL_jalign_results(  # noqa: N802
    DEL_jalign_results,  # noqa: N803
    sample_name,
    out_directory,
    deletions_length_cutoff=3000,
    jalign_written_cutoff=1,
):
    # reads jalign results
    df_cnmops_cnvpytor_DEL = pd.read_csv(DEL_jalign_results, sep="\t", header=None)  # noqa: N806
    df_cnmops_cnvpytor_DEL.columns = [
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
    df_cnmops_cnvpytor_DEL["len"] = df_cnmops_cnvpytor_DEL["end"] - df_cnmops_cnvpytor_DEL["start"]
    df_cnmops_cnvpytor_DEL["source"] = df_cnmops_cnvpytor_DEL["CN"].apply(
        lambda x: "cn.mops" if pd.Series(x).str.contains("CN").any() else "cnvpytor"
    )
    df_cnmops_cnvpytor_DEL["CNV_type"] = "DEL"
    df_cnmops_cnvpytor_DEL["copy_number"] = df_cnmops_cnvpytor_DEL["CN"].apply(lambda x: x.replace("CN", ""))
    df_cnmops_cnvpytor_DEL["copy_number"] = df_cnmops_cnvpytor_DEL["copy_number"].apply(
        lambda x: "DEL" if pd.Series(x).str.contains("deletion").any() else x
    )
    df_cnmops_cnvpytor_DEL_filtered = df_cnmops_cnvpytor_DEL[  # noqa: N806
        (df_cnmops_cnvpytor_DEL["jalign_written"] >= jalign_written_cutoff)
        | (df_cnmops_cnvpytor_DEL["len"] > deletions_length_cutoff)
    ]

    out_DEL_jalign = pjoin(  # noqa: N806
        out_directory,
        f"{sample_name}.cnmops_cnvpytor.DEL.jalign_lt{str(jalign_written_cutoff)}_or_len_lt{str(deletions_length_cutoff)}.bed",
    )
    df_cnmops_cnvpytor_DEL_filtered[["chrom", "start", "end", "CNV_type", "source", "copy_number"]].to_csv(
        out_DEL_jalign, sep="\t", header=None, index=False
    )

    out_DEL_jalign_merged = pjoin(  # noqa: N806
        out_directory,
        f"{sample_name}.cnmops_cnvpytor.DEL.jalign_lt{str(jalign_written_cutoff)}_or_len_lt{str(deletions_length_cutoff)}.merged.bed",
    )
    run_cmd(
        f"cat {out_DEL_jalign} | bedtools sort -i - | \
            bedtools merge -d 1500 -c 4,5,6 -o distinct  -i -  > {out_DEL_jalign_merged}"
    )

    return out_DEL_jalign_merged


def get_cnmops_cnvpytor_common_DEL(del_candidates, sample_name, out_directory):  # noqa: N802
    del_candidates_called_by_both_cnmops_cnvpytor = pjoin(
        out_directory, "{sample_name}.del_candidates_called_by_both_cnmops_cnvpytor.bed"
    )
    run_cmd(
        f'cut -f1-4 {del_candidates} | bedtools merge -c 4 -o distinct -i - |\
            grep "CN" | grep "deletion" > {del_candidates_called_by_both_cnmops_cnvpytor}'
    )

    df_del_candidates_called_by_both_cnmops_cnvpytor = pd.read_csv(
        del_candidates_called_by_both_cnmops_cnvpytor, sep="\t", header=None
    )
    df_del_candidates_called_by_both_cnmops_cnvpytor.columns = ["chrom", "start", "end", "CN"]
    df_del_candidates_called_by_both_cnmops_cnvpytor["CNV_type"] = "DEL"
    df_del_candidates_called_by_both_cnmops_cnvpytor["source"] = "cn.mops,cnvpytor"

    copy_number_list = []
    for __index, row in df_del_candidates_called_by_both_cnmops_cnvpytor.iterrows():
        CN = row["CN"]  # noqa: N806
        CN_list = CN.split(",")  # noqa: N806
        copy_number_value = ""
        for val in CN_list:
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


def run(argv):
    """
    combines cnvs from cnmops and cnvpytor using jalign results and converts them to vcf.
    input arguments:
    --cnmops_cnv_calls: input bed file holding cn.mops CNV calls
    --cnvpytor_cnv_calls: input bed file holding cnvpytor CNV calls
    --DEL_jalign_merged_results: jalign results for Deletions in tsv format.

    output files:
    bed file: <sample_name>.cnmops_cnvpytor.cnvs.combined.bed
        shows combined CNV calls called by cn.mops and cnvpytor.
    annotated bed file: <sample_name>.cnmops_cnvpytor.cnvs.combined.UG-CNV-LCR_annotate.bed
        shows combined CNV calls with UG-CNV-LCR annotation.
    """
    args = __parse_args(argv)
    logger.setLevel(getattr(logging, args.verbosity))

    out_directory = args.out_directory
    sample_name = args.sample_name
    ############################
    ### process DUPlications ###
    ############################
    out_cnmops_cnvs_DUP = get_cnmops_cnv_calls(args.cnmops_cnv_calls, sample_name, out_directory)  # noqa: N806
    out_cnvpytor_cnvs_DUP = get_cnvpytor_cnv_calls(args.cnvpytor_cnv_calls, sample_name, out_directory)  # noqa: N806
    # merge duplications
    cnmops_cnvpytor_merged_DUP = pjoin(out_directory, f"{sample_name}.cnmops_cnvpytor.DUP.merged.bed")  # noqa: N806
    run_cmd(
        f"cat {out_cnmops_cnvs_DUP} {out_cnvpytor_cnvs_DUP} | bedtools sort -i - | \
        bedtools merge -d 1500 -c 4,5,6 -o distinct -i - > {cnmops_cnvpytor_merged_DUP}"
    )

    ############################
    ###  process DELetions   ###
    ############################

    out_DEL_jalign_merged = process_DEL_jalign_results(  # noqa: N806
        args.DEL_jalign_merged_results,  # noqa: N806
        sample_name,
        out_directory,
        args.deletions_length_cutoff,
        args.jalign_written_cutoff,
    )
    out_del_candidates_called_by_both_cnmops_cnvpytor = get_cnmops_cnvpytor_common_DEL(
        args.DEL_jalign_merged_results, sample_name, out_directory
    )
    # merge deletions
    out_del_calls = pjoin(
        out_directory,
        f"{sample_name}.cnmops_cnvpytor.DEL.jalign_lt{str(args.jalign_written_cutoff)}_or_len_lt{str(args.deletions_length_cutoff)}.called_by_both_cnmops_cnvpytor.bedtools_merge.bed",
    )
    run_cmd(
        f"cat {out_DEL_jalign_merged} {out_del_candidates_called_by_both_cnmops_cnvpytor} | \
            bedtools sort -i - | bedtools merge -d 1500 -c 4,5,6 -o distinct  -i - > {out_del_calls}"
    )

    # combine results
    out_cnvs_combined = pjoin(out_directory, f"{sample_name}.cnmops_cnvpytor.cnvs.combined.bed")
    run_cmd(f"cat {cnmops_cnvpytor_merged_DUP} {out_del_calls} | bedtools sort -i - > {out_cnvs_combined}")
    logger.info(f"out_cnvs_combined: {out_cnvs_combined}")

    # annotate with ug-cnv-lcr
    out_cnvs_combined_annotated = pjoin(
        out_directory, f"{sample_name}.cnmops_cnvpytor.cnvs.combined.UG-CNV-LCR_annotate.bed"
    )
    run_cmd(
        f"bedtools intersect -f 0.5 -loj -wa -wb -a {out_cnvs_combined} -b {args.ug_cnv_lcr} | \
            cut -f 1-6,10 > {out_cnvs_combined_annotated}"
    )
    logger.info(f"out_cnvs_combined_annotated: {out_cnvs_combined_annotated}")

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
