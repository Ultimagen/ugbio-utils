#!/env/python
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
#    Compare UG callset to ground truth (preferentially using VCFEVAL as an engine)
# CHANGELOG in reverse chronological order

from __future__ import annotations

import argparse
import logging
import os
import sys
from os.path import dirname
from shutil import copyfile

import pandas as pd
import pysam
from joblib import Parallel, delayed
from simppl.simple_pipeline import SimplePipeline
from tqdm import tqdm
from ugbio_core.bed_utils import BedUtils
from ugbio_core.consts import DEFAULT_FLOW_ORDER
from ugbio_core.h5_utils import read_hdf
from ugbio_core.logger import logger
from ugbio_core.vcf_utils import VcfUtils
from ugbio_core.vcfbed import vcftools
from ugbio_core.vcfbed.interval_file import IntervalFile

from ugbio_comparison import comparison_utils
from ugbio_comparison.comparison_pipeline import ComparisonPipeline
from ugbio_comparison.vcf_comparison_utils import VcfComparisonUtils

MIN_CONTIG_LENGTH = 100000


def _contig_concordance_annotate_reinterpretation(  # noqa: PLR0913
    # pylint: disable=too-many-arguments
    raw_calls_vcf,
    concordance_vcf,
    contig,
    reference,
    bw_high_quality,
    bw_all_quality,
    annotate_intervals,
    flow_order,
    base_name_outputfile,
    enable_reinterpretation,
    ignore_low_quality_fps,
    scoring_field,
):
    logger.info("Reading %s", contig)
    concordance = comparison_utils.vcf2concordance(
        raw_calls_vcf,
        concordance_vcf,
        contig,
        scoring_field=scoring_field,
    )

    annotated_concordance, _ = comparison_utils.annotate_concordance(
        concordance,
        reference,
        bw_high_quality,
        bw_all_quality,
        annotate_intervals,
        flow_order=flow_order,
    )

    if enable_reinterpretation:
        annotated_concordance = comparison_utils.reinterpret_variants(
            annotated_concordance, reference, ignore_low_quality_fps=ignore_low_quality_fps
        )
    logger.debug("%s: %s", contig, annotated_concordance.shape)
    annotated_concordance.to_hdf(f"{base_name_outputfile}{contig}.h5", key=contig)


def get_parser() -> argparse.ArgumentParser:
    ap_var = argparse.ArgumentParser(prog="run_comparison_pipeline.py", description="Compare VCF to ground truth")
    ap_var.add_argument("--input_prefix", help="Prefix of the input file", required=True, type=str)
    ap_var.add_argument("--output_file", help="Output h5 file", required=True, type=str)
    ap_var.add_argument(
        "--output_interval",
        help="Output bed file of intersected intervals",
        required=True,
        type=str,
    )
    ap_var.add_argument("--gtr_vcf", help="Ground truth VCF file", required=True, type=str)
    ap_var.add_argument(
        "--cmp_intervals",
        help="Ranges on which to perform comparison (bed/interval_list)",
        required=False,
        type=str,
        default=None,
    )
    ap_var.add_argument(
        "--highconf_intervals",
        help="High confidence intervals (bed/interval_list)",
        required=True,
        type=str,
    )
    ap_var.add_argument(
        "--annotate_intervals",
        help="interval files for annotation (multiple possible)",
        required=False,
        type=str,
        default=None,
        action="append",
    )
    ap_var.add_argument("--reference", help="Reference genome", required=True, type=str)
    ap_var.add_argument("--reference_dict", help="Reference genome dictionary", required=False, type=str)
    ap_var.add_argument(
        "--reference_sdf",
        help="VCFEVAL SDF index for the reference genome, in case difference from reference.sdf",
        required=False,
        type=str,
        default=None,
    )
    ap_var.add_argument(
        "--coverage_bw_high_quality",
        help="BigWig file with coverage only on high mapq reads",
        required=False,
        default=None,
        type=str,
        action="append",
    )
    ap_var.add_argument(
        "--coverage_bw_all_quality",
        help="BigWig file with coverage on all mapq reads",
        required=False,
        default=None,
        type=str,
        action="append",
    )
    ap_var.add_argument(
        "--call_sample_name",
        help="Name of the call sample",
        required=True,
        default="sm1",
    )
    ap_var.add_argument("--truth_sample_name", help="Name of the truth sample", required=True)
    ap_var.add_argument("--ignore_filter_status", help="Ignore variant filter status", action="store_true")
    ap_var.add_argument(
        "--revert_hom_ref",
        help="For DeepVariant callsets - revert filtered hom_ref to het_ref for max recall calculation",
        action="store_true",
    )
    ap_var.add_argument(
        "--scoring_field",
        help="The pipeline expects a TREE_SCORE column in order to score the variants. If another field is \
        provided via scoring_field then its values will be copied to the TREE_SCORE column",
        required=False,
        default=None,
        type=str,
    )
    ap_var.add_argument(
        "--flow_order",
        type=str,
        help="Sequencing flow order (4 cycle)",
        required=False,
        default=DEFAULT_FLOW_ORDER,
    )
    ap_var.add_argument(
        "--output_suffix",
        help="Add suffix to the output file",
        required=False,
        default="",
        type=str,
    )
    ap_var.add_argument(
        "--enable_reinterpretation",
        help="Enable variant reinterpretation in flow space (disabled by default)",
        action="store_true",
    )
    ap_var.add_argument(
        "--special_chromosome",
        help="The chromosome that would be used for the \
        'concordance' dataframe (whole genome mode only)",
        default="chr9",
    )
    ap_var.add_argument("--is_mutect", help="Are the VCFs output of Mutect (false)", action="store_true")
    ap_var.add_argument("--n_jobs", help="n_jobs of parallel on contigs", type=int, default=-1)
    ap_var.add_argument(
        "--use_tmpdir", help="Should temporary files be stored in temporary directory", action="store_true"
    )
    ap_var.add_argument(
        "--verbosity",
        help="Verbosity: ERROR, WARNING, INFO, DEBUG",
        required=False,
        default="INFO",
    )
    return ap_var


def _setup_pipeline_and_utilities(args) -> tuple[SimplePipeline, VcfComparisonUtils, VcfUtils]:
    """Setup pipeline and utility objects."""
    logger.setLevel(getattr(logging, args.verbosity))
    sp = SimplePipeline(args.fc, args.lc, debug=args.d, print_timing=True)
    vcu = VcfComparisonUtils(sp)
    vu = VcfUtils(sp)
    return sp, vcu, vu


def _setup_interval_files(sp, args) -> tuple[IntervalFile, IntervalFile]:
    """Setup interval file objects."""
    # use tmpdir consistent with comparison_pipeline
    if args.use_tmpdir:
        scratchdir = dirname(args.output_file)
    else:
        scratchdir = False  # do not use tmpdir

    cmp_intervals = IntervalFile(sp, args.cmp_intervals, args.reference, args.reference_dict, scratchdir=scratchdir)
    highconf_intervals = IntervalFile(
        sp, args.highconf_intervals, args.reference, args.reference_dict, scratchdir=scratchdir
    )

    return cmp_intervals, highconf_intervals


def _intersect_intervals_and_save_args(vu, cmp_intervals, highconf_intervals, args) -> None:
    """Intersect intervals and save arguments to output file."""
    # intersect intervals and output as a bed file
    if cmp_intervals.is_none():  # interval of highconf_intervals
        logger.info(f"copy {args.highconf_intervals} to {args.output_interval}")
        copyfile(str(highconf_intervals.as_bed_file()), args.output_interval)
    else:
        BedUtils().intersect_bed_files(
            str(cmp_intervals.as_bed_file()), str(highconf_intervals.as_bed_file()), args.output_interval
        )

    # Save input arguments
    args_dict = {k: str(vars(args)[k]) for k in vars(args)}
    pd.DataFrame(args_dict, index=[0]).to_hdf(args.output_file, key="input_args")


def _create_comparison_pipeline(vcu, vu, cmp_intervals, highconf_intervals, args) -> ComparisonPipeline:
    """Create and configure the comparison pipeline."""
    return ComparisonPipeline(
        vcu=vcu,
        vu=vu,
        input_prefix=args.input_prefix,
        truth_file=args.gtr_vcf,
        cmp_intervals=cmp_intervals,
        highconf_intervals=highconf_intervals,
        ref_genome=args.reference,
        sdf_index=args.reference_sdf,
        call_sample=args.call_sample_name,
        truth_sample=args.truth_sample_name,
        output_file_name=args.output_file,
        output_suffix=args.output_suffix,
        ignore_filter=args.ignore_filter_status,
        revert_hom_ref=args.revert_hom_ref,
    )


def _process_single_interval_concordance(raw_calls_vcf, concordance_vcf, args) -> None:
    """Process concordance for single interval file."""
    concordance_df = comparison_utils.vcf2concordance(
        raw_calls_vcf,
        concordance_vcf,
        scoring_field=args.scoring_field,
    )
    annotated_concordance_df, _ = comparison_utils.annotate_concordance(
        concordance_df,
        args.reference,
        args.coverage_bw_high_quality,
        args.coverage_bw_all_quality,
        args.annotate_intervals,
        flow_order=args.flow_order,
    )

    if args.enable_reinterpretation:
        annotated_concordance_df = comparison_utils.reinterpret_variants(
            annotated_concordance_df,
            args.reference,
            ignore_low_quality_fps=args.is_mutect,
        )

    annotated_concordance_df.to_hdf(args.output_file, key="concordance", mode="a")
    # hack until we totally remove chr9
    annotated_concordance_df.to_hdf(args.output_file, key="comparison_result", mode="a")
    vcftools.bed_files_output(
        annotated_concordance_df,
        args.output_file,
        mode="w",
        create_gt_diff=(not args.is_mutect),
    )


def _get_filtered_contigs(raw_calls_vcf) -> list[str]:
    """Get list of contigs filtered by minimum length."""
    with pysam.VariantFile(raw_calls_vcf) as variant_file:
        # we filter out short contigs to prevent huge files
        filtered_contigs = []
        for contig_name in variant_file.header.contigs:
            contig = variant_file.header.contigs[contig_name]
            if contig.length is not None and contig.length > MIN_CONTIG_LENGTH:
                filtered_contigs.append(str(contig_name))
        return filtered_contigs


def _process_whole_genome_concordance(raw_calls_vcf, concordance_vcf, args) -> None:
    """Process concordance for whole genome (per chromosome)."""
    contigs = _get_filtered_contigs(raw_calls_vcf)
    base_name_outputfile = os.path.splitext(args.output_file)[0]

    # Process each contig in parallel
    Parallel(n_jobs=args.n_jobs, max_nbytes=None)(
        delayed(_contig_concordance_annotate_reinterpretation)(
            raw_calls_vcf,
            concordance_vcf,
            contig,
            args.reference,
            args.coverage_bw_high_quality,
            args.coverage_bw_all_quality,
            args.annotate_intervals,
            args.flow_order,
            base_name_outputfile,
            args.enable_reinterpretation,
            args.is_mutect,
            args.scoring_field,
        )
        for contig in tqdm(contigs)
    )

    _merge_contig_results(contigs, base_name_outputfile, args)


def _merge_contig_results(contigs, base_name_outputfile, args) -> None:
    """Merge temporary H5 files and generate bed files."""
    # find columns and set the same header for empty dataframes
    df_columns = None
    for contig in contigs:
        h5_temp = read_hdf(f"{base_name_outputfile}{contig}.h5", key=str(contig))
        if h5_temp.shape == (0, 0):  # empty dataframes are dropped to save space
            continue
        df_columns = pd.DataFrame(columns=h5_temp.columns)
        break

    # Merge temp files into main output file
    for contig in contigs:
        h5_temp = read_hdf(f"{base_name_outputfile}{contig}.h5", key=str(contig))
        if h5_temp.shape == (0, 0):  # empty dataframes get default columns
            h5_temp = pd.concat((h5_temp, df_columns), axis=1)
        h5_temp.to_hdf(args.output_file, mode="a", key=str(contig))
        if contig == args.special_chromosome:
            h5_temp.to_hdf(args.output_file, mode="a", key="concordance")
        os.remove(f"{base_name_outputfile}{contig}.h5")

    # Generate bed files
    write_mode = "w"
    for contig in contigs:
        annotated_concordance_df = read_hdf(args.output_file, key=str(contig))
        vcftools.bed_files_output(
            annotated_concordance_df,
            args.output_file,
            mode=write_mode,
            create_gt_diff=(not args.is_mutect),
        )
        write_mode = "a"


def run(argv: list[str]):
    """Concordance between VCF and ground truth"""
    parser = get_parser()
    SimplePipeline.add_parse_args(parser)
    args = parser.parse_args(argv[1:])

    # Setup pipeline and utilities
    sp, vcu, vu = _setup_pipeline_and_utilities(args)

    # Setup interval files
    cmp_intervals, highconf_intervals = _setup_interval_files(sp, args)

    # Intersect intervals and save arguments
    _intersect_intervals_and_save_args(vu, cmp_intervals, highconf_intervals, args)

    # Create and run comparison pipeline
    comparison_pipeline = _create_comparison_pipeline(vcu, vu, cmp_intervals, highconf_intervals, args)
    raw_calls_vcf, concordance_vcf = comparison_pipeline.run()

    # Process concordance based on interval type
    if not cmp_intervals.is_none():
        # single interval-file concordance - will be saved in a single dataframe
        _process_single_interval_concordance(raw_calls_vcf, concordance_vcf, args)
    else:
        # whole-genome concordance - will be saved in dataframe per chromosome
        _process_whole_genome_concordance(raw_calls_vcf, concordance_vcf, args)


def main():
    run(sys.argv)


if __name__ == "__main__":
    main()
