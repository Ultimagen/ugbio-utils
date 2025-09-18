import argparse
import logging
import os
import statistics
import subprocess
import sys
import tempfile
import warnings
from concurrent.futures import ThreadPoolExecutor
from os.path import basename, dirname
from os.path import join as pjoin

import pysam
from ugbio_core.logger import logger
from ugbio_core.vcfbed import vcftools

from ugbio_featuremap.featuremap_utils import FeatureMapFields

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# to make it more robust , maybe we can do the following:
# given a vcf,
# get all info/format fields
# go over the records - if fields is integer/float mark the field name for aggregation
columns_for_mean_aggregation = {
    [FeatureMapFields.BCSQ.value, "Base calling error likelihood in calling the SNV, in Phred scale"],
    [FeatureMapFields.BCSQCSS.value, "Cycle Skip Size when computing base calling error likelihood"],
    [FeatureMapFields.RL.value, "Read length (post adapter trimming)"][
        FeatureMapFields.INDEX.value,
        "Position in the read of the SNV relative to read start, in the synthesis direction",
    ],
    [FeatureMapFields.DUP.value, "Is the read a duplicate, interpreted from CRAM flag"],
    [FeatureMapFields.REV.value, "Is the read mapped to the reverse strand, interpreted from CRAM flag"],
    [
        FeatureMapFields.SCST.value,
        "Softclip length in the start of the read (synthesis direction), interpreted from the CIGAR string",
    ],
    [
        FeatureMapFields.SCED.value,
        "Softclip length in the end of the read (synthesis direction), interpreted from the CIGAR string",
    ],
    [FeatureMapFields.MAPQ.value, "Read mapping quality"],
    [FeatureMapFields.EDIST.value, "Read Levenshtein edit distance from reference"],
    [
        FeatureMapFields.HAMDIST.value,
        "Hamming distance: number of M bases different on read from references, disregarding indels",
    ],
    [
        FeatureMapFields.HAMDIST_FILT.value,
        "Filtered Hamming distance: number of m<=M bases passing the adjacent base filter different on read from "
        "references, disregarding indels",
    ],
    [FeatureMapFields.MQUAL.value, "SingleReadSNV model inferred raw Phred scaled quality"],
    [FeatureMapFields.SNVQ.value, "SingleReadSNV model inferred Phred scaled quality, recalibrated to the SNVQ value"],
    [
        FeatureMapFields.FILT.value,
        "Pre-filter status for SNVs reads, 1 means an SNVs read passed all the filters defined in the SRSNV model,"
        "0 means it failed at least one filter",
    ],
    [FeatureMapFields.SMQ_BEFORE.value, "Mean quality of 20 bases before the locus in the synthesis direction"],
    [FeatureMapFields.SMQ_AFTER.value, "Mean quality of 20 bases after the locus in the synthesis direction"],
    [FeatureMapFields.ADJ_REF_DIFF.value, "The 3 adjacent bases to the locus do not fully match the reference genome"],
}
columns_for_min_aggregation = [
    FeatureMapFields.SNVQ.value.lower(),
    FeatureMapFields.X_INDEX.value.lower(),
]
columns_for_max_aggregation = [
    FeatureMapFields.SNVQ.value.lower(),
    FeatureMapFields.X_INDEX.value.lower(),
]
columns_for_fillna = [
    FeatureMapFields.IS_CYCLE_SKIP.value
]  # check if IS_CYCLE_SKIP is really removed from Featuremap v1.23

added_agg_features = {
    "alt_reads": ["number of supporting reads for the alternative allele", "Integer"],
    "ref_allele": ["reference allele", "String"],
    "alt_allele": ["alternative allele", "String"],
    "x_qual_mean": ["mean value of X_QUAL", "Float"],
    "x_score_mean": ["mean value of X_SCORE", "Float"],
    "x_edist_mean": ["mean value of X_EDIST", "Float"],
    "x_length_mean": ["mean value of X_LENGTH", "Float"],
    "x_mapq_mean": ["mean value of X_MAPQ", "Float"],
    "x_fc1_mean": ["mean value of X_FC1-Number of M bases different on read from references", "Float"],
    "x_fc2_mean": ["mean value of X_FC2-Number of features before score threshold filter", "Float"],
    "max_softclip_length_mean": ["mean value of MAX_SOFTCLIP_LENGTH", "Float"],
    "x_flags_mean": ["mean value of X_FLAGS", "Float"],
    "ml_qual_mean": ["mean value of ML_QUAL", "Float"],
    "x_qual_max": ["max value of X_QUAL", "Float"],
    "x_index_max": ["max value of X_INDEX", "Integer"],
    "x_qual_min": ["min value of X_QUAL", "Float"],
    "x_index_min": ["min value of X_INDEX", "Integer"],
    "count_forward": ["number of forward reads", "Integer"],
    "count_reverse": ["number of reverse reads", "Integer"],
    "count_duplicate": ["number of duplicate reads", "Integer"],
    "count_non_duplicate": ["number of non-duplicate reads", "Integer"],
}


def df_vcf_manual_aggregation(df_variants):  # noqa: C901
    def aggregate_mean(df, colname):
        values = []
        for tup in df[colname]:
            cleaned_list = list(tup)
            values.append(statistics.mean(cleaned_list))
        return values

    def aggregate_min(df, colname):
        values = []
        for tup in df[colname]:
            cleaned_list = list(tup)
            values.append(min(cleaned_list))
        return values

    def aggregate_max(df, colname):
        values = []
        for tup in df[colname]:
            cleaned_list = list(tup)
            values.append(max(cleaned_list))
        return values

    def parse_st_et_fields(df, colname):
        tags = ["MINUS", "MIXED", "PLUS", "UNDETERMINED"]
        for tag in tags:
            df[colname + "_" + tag.lower()] = df[colname].apply(lambda x, tag=tag: x.split("|").count(tag))
        return df

    def calculate_num_mixed(df):
        num_mixed = []
        for _index, row in df.iterrows():
            st, et = row["st"], row["et"]
            st_list = st.split("|")
            et_list = et.split("|")
            sum_value = sum(1 for i, j in zip(st_list, et_list, strict=False) if (i == j) & (i == "MIXED"))
            num_mixed.append(sum_value)
        df["num_mixed_reads"] = num_mixed
        return df

    def parse_is_forward(df):
        df["count_forward"] = df["is_forward"].apply(lambda x: x.split("|").count("T"))
        df["count_reverse"] = df["is_forward"].apply(lambda x: x.split("|").count("F"))
        return df

    def parse_is_duplicate(df):
        df["count_duplicate"] = df["is_duplicate"].apply(lambda x: x.split("|").count("T"))
        df["count_non_duplicate"] = df["is_duplicate"].apply(lambda x: x.split("|").count("F"))
        return df

    df_variants["alt_reads"] = [tup[1] for tup in df_variants["ad"]]
    df_variants["ref_allele"] = [tup[0] for tup in df_variants["alleles"]]
    df_variants["alt_allele"] = [tup[1] for tup in df_variants["alleles"]]
    for colname in columns_for_mean_aggregation:
        agg_colname = colname.lower() + "_mean"
        df_variants[agg_colname] = aggregate_mean(df_variants, colname.lower())
    for colname in columns_for_max_aggregation:
        agg_colname = colname.lower() + "_max"
        df_variants[agg_colname] = aggregate_max(df_variants, colname.lower())
    for colname in columns_for_min_aggregation:
        agg_colname = colname.lower() + "_min"
        df_variants[agg_colname] = aggregate_min(df_variants, colname.lower())
    df_variants = parse_is_forward(df_variants)
    df_variants = parse_is_duplicate(df_variants)
    df_variants["is_cycle_skip"] = df_variants["is_cycle_skip"].fillna(value=False)

    df_variants = calculate_num_mixed(df_variants)

    return df_variants


def sort_and_filter_vcf(featuremap_vcf_file, temp_dir, filter_string, interval_srting):
    sorted_featuremap = pjoin(temp_dir, basename(featuremap_vcf_file).replace(".vcf.gz", ".sorted.vcf.gz"))
    sorted_filtered_featuremap = pjoin(
        temp_dir, basename(featuremap_vcf_file).replace(".vcf.gz", ".sorted.filtered.vcf.gz")
    )
    sort_cmd = f"bcftools view {featuremap_vcf_file} {interval_srting} |\
                bcftools sort - -Oz -o {sorted_featuremap} && \
                bcftools index -t {sorted_featuremap}"
    logger.debug(sort_cmd)
    subprocess.check_call(sort_cmd, shell=True)  # noqa: S602
    if filter_string != "":
        sort_cmd = f"bcftools view {filter_string} {featuremap_vcf_file} {interval_srting} |\
                    bcftools sort - -Oz -o {sorted_filtered_featuremap} && \
                    bcftools index -t {sorted_filtered_featuremap}"
        logger.debug(sort_cmd)
        subprocess.check_call(sort_cmd, shell=True)  # noqa: S602
    else:
        sorted_filtered_featuremap = sorted_featuremap
    return sorted_featuremap, sorted_filtered_featuremap


def add_agg_fields_to_header(hdr):
    for field in added_agg_features:
        field_type = added_agg_features[field][1]
        field_description = added_agg_features[field][0]
        hdr.info.add(field, 1, field_type, field_description)


def process_vcf_row(row, df_variants, hdr, vcfout, write_agg_params):
    pos = row.pos
    chrom = row.chrom
    alt_allele = row.alleles[1]
    df_record = df_variants[
        (df_variants["chrom"] == chrom) & (df_variants["pos"] == pos) & (df_variants["alt_allele"] == alt_allele)
    ]

    if len(df_record) > 0:
        if write_agg_params:
            for key in added_agg_features:
                row.info[key] = df_record[key].to_list()[0]
    vcfout.write(row)


def featuremap_fields_aggregation(  # noqa: C901
    featuremap_vcf_file: str,
    output_vcf: str,
    filter_tags=None,
    genomic_interval: str = None,
    write_agg_params: bool = True,  # noqa: FBT001, FBT002
    verbose: bool = True,  # noqa: FBT001, FBT002
) -> str:
    """
    Write the vcf file with the aggregated fields and the xgb probability
    """
    if verbose:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)

    filter_string = f"-f {filter_tags}" if filter_tags else ""
    interval_srting = genomic_interval if genomic_interval else ""
    with tempfile.TemporaryDirectory(dir=dirname(output_vcf)) as temp_dir:
        sorted_featuremap, sorted_filtered_featuremap = sort_and_filter_vcf(
            featuremap_vcf_file, temp_dir, filter_string, interval_srting
        )

        # read vcf block to dataframe
        info_fields, format_fields, custom_info_fields = vcftools.get_vcf_fields_name(sorted_filtered_featuremap)
        custom_info_fields = list(custom_info_fields)
        df_variants = vcftools.get_vcf_df(sorted_filtered_featuremap, custom_info_fields=custom_info_fields)
        df_variants = df_vcf_manual_aggregation(df_variants)

        with pysam.VariantFile(sorted_featuremap) as vcfin:
            hdr = vcfin.header
            add_agg_fields_to_header(hdr)
            with pysam.VariantFile(output_vcf, mode="w", header=hdr) as vcfout:
                for row in vcfin:
                    process_vcf_row(row, df_variants, hdr, vcfout, write_agg_params)
            vcfout.close()
            vcfin.close()
    pysam.tabix_index(output_vcf, preset="vcf", min_shift=0, force=True)
    return output_vcf


def featuremap_fields_aggregation_on_an_interval_list(
    featuremap_vcf_file: str,
    output_vcf: str,
    interval_list: str,
    filter_tags=None,
    verbose: bool = True,  # noqa: FBT001, FBT002
) -> None:
    """
    Apply featuremap fields aggregation on an interval list
    Inputs:
        featuremap (str): The input featuremap vcf file
        output_vcf (str): The output pileup vcf file
        interval_list (str): The interval list file
        verbose (bool): The verbosity level (default: True)
    Output:
        output_vcf (str): The output vcf file including the aggregated fields and the xgb probability
    """
    if not output_vcf.endswith(".vcf.gz"):
        logger.debug("adding .vcf.gz suffix to the output vcf file")
        output_vcf = output_vcf + ".vcf.gz"

    with tempfile.TemporaryDirectory(dir=dirname(output_vcf)):
        genomic_intervals = []
        with open(interval_list, encoding="utf-8") as f:
            for line in f:
                # ignore header lines
                if line.startswith("@"):
                    continue
                # read genomic ineterval
                genomic_interval = line.strip()
                genomic_interval_list = genomic_interval.split("\t")
                chrom = genomic_interval_list[0]
                start = genomic_interval_list[1]
                end = genomic_interval_list[2]
                genomic_interval = chrom + ":" + str(start) + "-" + str(end)
                genomic_intervals.append(genomic_interval)

        params = [
            (
                featuremap_vcf_file,
                f"{output_vcf}.{genomic_interval}.int_list.vcf.gz",
                filter_tags,
                genomic_interval,
                verbose,
            )
            for genomic_interval in genomic_intervals
        ]
        num_cpus = os.cpu_count()
        with ThreadPoolExecutor(max_workers=num_cpus) as executor:
            results = list(executor.map(lambda p: featuremap_fields_aggregation(*p), params))

        # Write each string to the file
        with open("interval_vcf_files.list", "w") as file:
            for interval_vcf_file in results:
                file.write(interval_vcf_file + "\n")

        cmd = (
            f"bcftools concat -f interval_vcf_files.list -a | "
            f"bcftools sort - -Oz -o {output_vcf} && "
            f"bcftools index -t {output_vcf}"
        )
        logger.debug(cmd)
        subprocess.check_call(cmd, shell=True)  # noqa: S602
    return output_vcf


def __parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="Featuremap fields aggregation",
        description=run.__doc__,
    )
    parser.add_argument(
        "-f",
        "--featuremap",
        type=str,
        required=True,
        help="""Featuremap pileup vcf file""",
    )
    parser.add_argument(
        "-filter_string",
        "--filter_string",
        type=str,
        required=False,
        default="PASS",
        help="""filter tags to apply on the featuremap pileup vcf file""",
    )
    parser.add_argument(
        "-o",
        "--output_vcf",
        type=str,
        required=True,
        help="""Output pileup vcf file""",
    )
    parser.add_argument(
        "-i",
        "--interval_list_file",
        type=str,
        required=True,
        help="""Interval list file""",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        type=bool,
        required=False,
        default=True,
        help="""Whether to print debug messages (default: True)""",
    )
    return parser.parse_args(argv[1:])


def run(argv):
    """Add aggregated parameters and xgb probability to the featuremap pileup vcf file"""
    args_in = __parse_args(argv)
    featuremap_fields_aggregation_on_an_interval_list(
        args_in.featuremap_pileup,
        args_in.output_vcf,
        args_in.interval_list_file,
        args_in.filter_string,
        args_in.verbose,
    )


def main():
    run(sys.argv)


if __name__ == "__main__":
    main()
