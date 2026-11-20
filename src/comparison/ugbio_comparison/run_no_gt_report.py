# Copyright 2022 Ultima Genomics Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
# DESCRIPTION
#    Collect the statistics of the report without ground truth (report_wo_gt.ipynb)
from __future__ import annotations

import argparse
import itertools
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import ugbio_core.bed_utils as bu
import ugbio_core.vcfbed.variant_annotation as annotation
from ugbio_core.dna_sequence_utils import revcomp
from ugbio_core.logger import logger
from ugbio_core.vcfbed import vcftools

from ugbio_comparison.sigprofiler_runner import run_somatic_signatures, save_signatures_to_h5
from ugbio_comparison.vcf_comparison_utils import annotate_concordance

MAX_HMER_LENGTH = 12
SECOND_ALT_INDEX = 2
GATK_TABLES = (
    "CompOverlap",
    "CountVariants",
    "TiTvVariantEvaluator",
    "IndelLengthHistogram",
    "IndelSummary",
    "MetricsCollection",
    "ValidationReport",
    "VariantSummary",
    "MultiallelicSummary",
)
FIELDS_TO_IGNORE = [
    "GNOMAD_AF",
    "X_IC",
    "X_IL",
    "X_HIL",
    "X_HIN",
    "X_LM",
    "X_RM",
    "X_GCC",
    "X_CSS",
    "PL",
    "DB",
    "AD",
]


def insertion_deletion_statistics(variants: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    statistics for indel deletions and insertions.
    We collect for each hmer-indel length the number of insertions and deletions per base (A or G)
    The statistics are collected for homozygous and heterozygous separately
    """

    def calc_hmer_stats(gt_variants: pd.DataFrame) -> pd.DataFrame:
        hmer_stat = {}
        for hlen in range(1, MAX_HMER_LENGTH + 1):
            variants_of_length = gt_variants[gt_variants["hmer_indel_length"] == hlen]
            length_col = []
            for ind_clas in ("ins", "del"):
                indels = variants_of_length[variants_of_length["indel_classify"] == ind_clas]
                for nuc in (("A", "T"), ("G", "C")):
                    length_col.append(indels["hmer_indel_nuc"].isin(nuc).sum())
            hmer_stat[hlen] = length_col
        return pd.DataFrame(data=hmer_stat, index=["ins A", "ins G", "del A", "del G"])

    is_hom = variants["gt"] == (1, 1)
    return {"homo": calc_hmer_stats(variants[is_hom]), "hete": calc_hmer_stats(variants[~is_hom])}


def allele_freq_hist(variants: pd.DataFrame, nbins: int = 100) -> pd.DataFrame:
    """
    statistics for allele frequency
    For each group return the AF distribution in bins
    """
    bins = np.linspace(0, 1, nbins + 1)
    if "vaf" in variants.columns:
        variants["af"] = variants["vaf"]
    result = {}
    for group in variants["variant_type"].unique():
        histogram_data, _ = np.histogram(
            variants[variants["variant_type"] == group]["af"].apply(lambda x: x[0] if isinstance(x, tuple) else x),
            bins,
        )
        result[group] = pd.Series(histogram_data)
    return pd.DataFrame(data=result)


def _get_by_index(values, index):
    return values[index] if ((values is not None) and (len(values) > index)) else np.nan


def snp_statistics(variants: pd.DataFrame, ref_fasta: str) -> pd.Series:
    """Count the SNPs per trinucleotide context (96 categories, both strands collapsed)"""
    variants = annotation.classify_indel(variants)
    variants = annotation.is_hmer_indel(variants, ref_fasta)
    variants = annotation.get_motif_around(variants, 5, ref_fasta)
    variants = variants.set_index(["chrom", "pos"])

    variants["af"] = variants["af"].apply(lambda x: x[0] if isinstance(x, tuple) else x)
    variants["alt_1"] = variants["alleles"].apply(lambda x: x[1])
    variants["alt_2"] = variants["alleles"].apply(lambda x: _get_by_index(x, SECOND_ALT_INDEX))
    for ind in range(1, 3):
        variants[f"gt_{ind}"] = variants["gt"].apply(lambda x, index=ind: _get_by_index(x, index))

    variants = variants.drop(columns=["alleles", "gt", "af"])

    motif_index_bothstrands = pd.MultiIndex.from_tuples(
        [
            x
            for x in itertools.product(
                ["".join(x) for x in itertools.product(["A", "C", "G", "T"], repeat=3)],
                ["A", "C", "G", "T"],
            )
            if x[0][1] != x[1]
        ],
        names=["ref_motif", "alt_1"],
    )
    motif_index = pd.MultiIndex.from_tuples(
        [
            x
            for x in itertools.product(
                ["".join(x) for x in itertools.product(["A", "C", "G", "T"], repeat=3)],
                ["A", "C", "G", "T"],
            )
            if x[0][1] != x[1] and x[0][1] in ("A", "C")
        ],
        names=["ref_motif", "alt_1"],
    )

    snps = variants[~variants["indel"]].drop(
        columns=[
            "indel",
            "indel_classify",
            "indel_length",
            "hmer_indel_length",
            "hmer_indel_nuc",
        ]
    )

    snps["ref_motif"] = snps["left_motif"].str.slice(-1) + snps["ref"] + snps["right_motif"].str.slice(0, 1)

    motifs_bothstrands = (
        snps.groupby(["ref_motif", "alt_1"])
        .size()
        .reindex(motif_index_bothstrands)
        .fillna(0)
        .astype(int)
        .rename("size")
    )

    index_vars = motifs_bothstrands.index.names
    x = motifs_bothstrands.reset_index()
    x_f = (
        x[x["ref_motif"].str.slice(1, 2).isin(["A", "C"])]
        .set_index(index_vars)
        .reindex(motif_index)
        .fillna(0)
        .astype(int)
    )
    x_r = x[x["ref_motif"].str.slice(1, 2).isin(["G", "T"])]
    x_r = (
        x_r.assign(**{c: x_r[c].apply(revcomp) for c in index_vars})
        .set_index(index_vars)
        .reindex(motif_index)
        .fillna(0)
        .astype(int)
    )
    return (x_f + x_r)["size"]


def variant_eval_statistics(
    vcf_input: str,
    reference: str,
    dbsnp: str,
    output_prefix: str,
    sample_id_or_name: str | int = 0,
    annotation_names: list | None = None,
    *,
    append_confident_calls: bool = False,
) -> dict[str, pd.DataFrame]:
    """Run GATK VariantEval and parse its tables"""
    if annotation_names is None:
        annotation_names = []
    annotation_names_str = [f"--selectNames {x}" for x in annotation_names]
    if append_confident_calls:
        annotation_names_str += ["--selectNames HIGHCONF"]
    annotation_conditions_str = [f'--select vc.hasAttribute("{x}")' for x in annotation_names]
    if append_confident_calls:
        if isinstance(sample_id_or_name, int):
            annotation_conditions_str += [f"--select vc.getGenotype({sample_id_or_name}).getGQ()>=20"]
        elif isinstance(sample_id_or_name, str):
            annotation_conditions_str += [f'--select vc.getGenotype("{sample_id_or_name}").getGQ()>=20']
    cmd = (
        [
            "gatk",
            "VariantEval",
            "--eval",
            f"{vcf_input}",
            "--reference",
            f"{reference}",
            "--dbsnp",
            f"{dbsnp}",
            "--output",
            f"{output_prefix}.txt",
        ]
        + annotation_names_str
        + annotation_conditions_str
    )
    logger.info(" ".join(cmd))
    subprocess.check_call(" ".join(cmd).split())

    with open(f"{output_prefix}.txt", encoding="latin-1") as output_file:
        return _parse_single_report(output_file)


def _parse_single_report(report) -> dict[str, pd.DataFrame]:
    def _parse_single_table(input_table) -> pd.DataFrame:
        headers = input_table.readline().split()
        table = []
        line = input_table.readline().strip("\n").split()
        while len(line) == len(headers):
            table.append(line)
            line = input_table.readline().strip("\n").split()

        return pd.DataFrame(table, columns=headers)

    data = {}
    for line in report:
        table_name = next((name for name in GATK_TABLES if f"#:GATKTable:{name}" in line), None)
        if table_name is not None:
            data[table_name] = _parse_single_table(report)
    return data


def run_eval_tables_only(arg_values: argparse.Namespace) -> None:
    if arg_values.sample_name is not None:
        sample_id = arg_values.sample_name
    elif arg_values.sample_id is not None:
        sample_id = arg_values.sample_id
    else:
        sample_id = 0
    eval_tables = variant_eval_statistics(
        arg_values.input_file,
        arg_values.reference,
        arg_values.dbsnp,
        arg_values.output_prefix,
        sample_id,
        arg_values.annotation_names,
        append_confident_calls=True,
    )
    for eval_table_name, eval_table in eval_tables.items():
        eval_table.to_hdf(f"{arg_values.output_prefix}.h5", key=f"eval_{eval_table_name}")


def run_full_analysis(arg_values: argparse.Namespace) -> None:
    if arg_values.callable_region is not None:
        callable_size = bu.BedUtils().count_bases_in_bed_file(arg_values.callable_region)
        callable_size_series = pd.Series(callable_size, index=["callable_size"])
        callable_size_series.to_hdf(f"{arg_values.output_prefix}.h5", key="callable_size")
    eval_tables = variant_eval_statistics(
        arg_values.input_file,
        arg_values.reference,
        arg_values.dbsnp,
        arg_values.output_prefix,
        annotation_names=[],
    )

    logger.info("Converting vcf to df")
    variants = vcftools.get_vcf_df(
        arg_values.input_file,
        sample_id=arg_values.sample_id,
        sample_name=arg_values.sample_name,
        ignore_fields=FIELDS_TO_IGNORE,
    )
    logger.info("Annotating vcf")
    annotated_variants, _ = annotate_concordance(variants, arg_values.reference)

    logger.info("insertion/ deletion statistics")
    ins_del_stats = insertion_deletion_statistics(variants)

    logger.info("Allele frequency histogram")
    af_hist = allele_freq_hist(annotated_variants)

    logger.info("snps motifs statistics")
    snp_motifs = snp_statistics(variants, arg_values.reference)

    logger.info("save all statistics in h5 file")
    ins_del_stats["hete"].to_hdf(f"{arg_values.output_prefix}.h5", key="ins_del_hete")
    ins_del_stats["homo"].to_hdf(f"{arg_values.output_prefix}.h5", key="ins_del_homo")
    af_hist.to_hdf(f"{arg_values.output_prefix}.h5", key="af_hist")
    snp_motifs.to_hdf(f"{arg_values.output_prefix}.h5", key="snp_motifs")
    for eval_table_name, eval_table in eval_tables.items():
        eval_table.to_hdf(f"{arg_values.output_prefix}.h5", key=f"eval_{eval_table_name}")


def run_somatic_analysis(arg_values: argparse.Namespace) -> None:
    """COSMIC signature analysis (SigProfiler).

    Writes the activities and sample profiles to ``<output_prefix>.h5`` (keys ``<context>_activity``
    and ``<context>_profile``) and the plots to ``<output_prefix>/final_results/``.
    """
    output_dir = Path(arg_values.output_prefix)
    sigprofiler_args = [
        "--input_vcf",
        arg_values.input_file,
        "--genome_archive",
        arg_values.genome_archive,
        "--genome",
        arg_values.reference_name,
        "--cosmic_version",
        str(arg_values.cosmic_version),
    ]
    if arg_values.reference_dir is not None:
        sigprofiler_args += ["--reference_dir", arg_values.reference_dir]
    summary = run_somatic_signatures(output_dir, sigprofiler_args)
    save_signatures_to_h5(output_dir, summary, f"{arg_values.output_prefix}.h5")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="run_no_gt_report",
        description="Collect metrics for runs without ground truth",
    )

    subparsers = ap.add_subparsers(required=True)
    full_analysis = subparsers.add_parser(name="full_analysis", description="Run the full analysis of no_gt_report")

    full_analysis.add_argument("--input_file", help="Input vcf file", required=True, type=str)
    full_analysis.add_argument("--dbsnp", help="dbsnp vcf file", required=True, type=str)
    full_analysis.add_argument("--reference", help="Reference genome", required=True, type=str)
    full_analysis.add_argument("--output_prefix", help="output file", required=True, type=str)
    full_analysis.add_argument(
        "--sample_id",
        help="sample id (useful in case there is more than one in the vcf)",
        required=False,
        type=int,
        default=0,
    )
    full_analysis.add_argument(
        "--sample_name", help="sample name (useful in case there is more than one in the vcf)", required=False, type=str
    )
    full_analysis.add_argument(
        "--callable_region", help="callable region bed file", required=False, type=str, default=None
    )

    full_analysis.set_defaults(func=run_full_analysis)

    # Run variant eval only - for JC report
    parser_eval_tables = subparsers.add_parser(name="variant_eval", description="Run variant eval only")
    parser_eval_tables.add_argument("--input_file", help="Input vcf file", required=True, type=str)
    parser_eval_tables.add_argument("--dbsnp", help="dbsnp vcf file", required=True, type=str)
    parser_eval_tables.add_argument("--reference", help="Reference genome", required=True, type=str)
    parser_eval_tables.add_argument("--output_prefix", help="output file", required=True, type=str)
    parser_eval_tables.add_argument(
        "--sample_name", help="Name of the sample to fetch from the VCF", required=False, type=str
    )
    parser_eval_tables.add_argument(
        "--sample_id", help="Number of the sample to fetch from the VCF", required=False, type=int
    )

    parser_eval_tables.add_argument(
        "--annotation_names",
        help="annotation name to filter on",
        required=False,
        nargs="*",
    )
    parser_eval_tables.set_defaults(func=run_eval_tables_only)

    # Run somatic analysis on somatic data
    somatic_analysis = subparsers.add_parser(
        name="somatic_analysis", description="Run mutation signatures and motif graphs"
    )

    somatic_analysis.add_argument("--input_file", help="Input vcf file (plain or bgzipped)", required=True, type=str)
    somatic_analysis.add_argument(
        "--reference_name", help="Reference genome name", required=False, type=str, default="GRCh38"
    )
    somatic_analysis.add_argument(
        "--genome_archive",
        help="SigProfiler archive of the reference genome (e.g. GRCh38.tar.gz), installed locally",
        required=True,
        type=str,
    )
    somatic_analysis.add_argument(
        "--reference_dir",
        help="Writable directory to install the genome into, by default <output_prefix>/sigprofiler_reference",
        required=False,
        type=str,
    )
    somatic_analysis.add_argument("--output_prefix", help="output file and directory", required=True, type=str)
    somatic_analysis.add_argument(
        "--cosmic_version", help="Signatures cosmic version", required=False, type=str, default="3.3"
    )
    somatic_analysis.set_defaults(func=run_somatic_analysis)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
