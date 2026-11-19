#!/usr/bin/env python
# Copyright 2026 Ultima Genomics Inc.
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
#    Prepare training data for the depth CNV filtering model (VcfType.CNV)
"""
Prepare training data for the depth CNV filtering model.

Every call of at least min_len inside the benchmark BED is labelled with truvari bench against the truth: matched -> 1,
unmatched -> 0 (pctovl 0.5, pctsize 0.5, pctseq 0, refdist 100 kb, duplications matched to insertions). Writes
<prefix>.train.h5 and <prefix>.test.h5, split by chromosome, with the INFO columns the filter reads and "label".
Samples with known events (e.g. cell lines) are prepared the same way from their pieces (split_cnv_into_pieces writes
the pieces, their truth and benchmark BED). All tables are the input of

    train_models_pipeline --vcf_type cnv --gt_type approximate --train_dfs <prefix>.train.h5 ... --test_dfs ...
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile

import pandas as pd
import pysam
from ugbio_comparison.sv_comparison_pipeline import SVComparison
from ugbio_core.logger import logger
from ugbio_core.vcf_utils import VcfUtils
from ugbio_core.vcfbed import vcftools

from ugbio_filtering import transformers
from ugbio_filtering.tprep_constants import VcfType

# a call matches a truth event of its type with 50% reciprocal overlap and size within 2x
TRUVARI_OPTIONS = {
    "pctseq": 0.0,
    "maxsize": -1,
    "pctovl": 0.5,
    "pctsize": 0.5,
    "refdist": 100000,
    "chunksize": 100000,
    "dup_to_ins": True,  # T2T-Q100 writes duplications as insertions
}
KEY_COLUMNS = ["chrom", "pos", "id"]  # calls may share a start (e.g. two pair DUP candidates); IDs are unique


def read_calls(call_vcf: str) -> pd.DataFrame:
    """Calls with the INFO columns the depth CNV filter reads, plus chrom, pos, id and end."""
    calls = vcftools.get_vcf_df(call_vcf, custom_info_fields=transformers.CNV_INFO_FIELDS)
    with pysam.VariantFile(call_vcf) as vcf:
        ends = {(r.chrom, r.pos, r.id): r.stop for r in vcf}
    calls = calls.reset_index(drop=True)
    calls["end"] = [ends[k] for k in zip(calls["chrom"], calls["pos"], calls["id"], strict=True)]
    return calls


def label_calls(call_vcf: str, truth_vcf: str, benchmark_bed: str, workdir: str, min_len: int = 500) -> pd.DataFrame:
    """
    Label every call of at least min_len inside the benchmark BED against the truth with truvari bench.

    Parameters
    ----------
    call_vcf : str
        Depth CNV calls (any FILTER: all are scored).
    truth_vcf : str
        Truth SV VCF.
    benchmark_bed : str
        Benchmark regions; calls outside are not labelled.
    workdir : str
        Directory for the intermediate files.
    min_len : int
        Minimal call length.

    Returns
    -------
    pd.DataFrame
        chrom, pos and label (1 matched, 0 unmatched).
    """
    selected = os.path.join(workdir, "calls.selected.vcf.gz")
    allpass = os.path.join(workdir, "calls.allpass.vcf.gz")  # no FILTER, so truvari scores every call
    vcf_utils = VcfUtils()
    vcf_utils.view_vcf(call_vcf, selected, extra_args=f"-i 'abs(INFO/SVLEN)>={min_len}'")
    vcf_utils.remove_filter_annotations(selected, allpass)
    bench = os.path.join(workdir, "truvari")
    SVComparison().run_truvari(
        allpass, truth_vcf, bench, bed=benchmark_bed, ignore_filter=True, ignore_type=False, **TRUVARI_OPTIONS
    )
    rows = []
    for name, label in (("tp-comp", 1), ("fp", 0)):
        with pysam.VariantFile(os.path.join(bench, f"{name}.vcf.gz")) as vcf:
            rows += [(r.chrom, r.pos, r.id, label) for r in vcf]
    return pd.DataFrame(rows, columns=[*KEY_COLUMNS, "label"]).drop_duplicates(KEY_COLUMNS)


def training_prep_cnv(
    call_vcf: str,
    truth_vcf: str,
    benchmark_bed: str,
    train_chromosomes: list[str],
    output_prefix: str,
    min_len: int = 500,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Write <output_prefix>.train.h5 and <output_prefix>.test.h5 for train_models_pipeline (vcf_type cnv).

    Parameters
    ----------
    call_vcf : str
        Depth CNV calls with truth (e.g. HG002).
    truth_vcf : str
        Truth SV VCF.
    benchmark_bed : str
        Benchmark regions of the truth.
    train_chromosomes : list[str]
        Chromosomes of call_vcf used for training; the other labelled chromosomes go to the test set.
    output_prefix : str
        Output prefix.
    min_len : int
        Minimal call length.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        The training and test sets.
    """
    columns = [*transformers.get_needed_features(VcfType.CNV), "label", *KEY_COLUMNS]
    with tempfile.TemporaryDirectory(dir=os.path.dirname(os.path.abspath(output_prefix))) as workdir:
        labels = label_calls(call_vcf, truth_vcf, benchmark_bed, workdir, min_len)
    labelled = read_calls(call_vcf).merge(labels, on=KEY_COLUMNS, how="inner")
    is_train = labelled["chrom"].isin(train_chromosomes)
    train, test = labelled[is_train].reindex(columns=columns), labelled[~is_train].reindex(columns=columns)
    train.to_hdf(f"{output_prefix}.train.h5", key="data", mode="w")
    test.to_hdf(f"{output_prefix}.test.h5", key="data", mode="w")
    logger.info(
        f"training rows {len(train)} ({int(train['label'].sum())} true); "
        f"test rows {len(test)} ({int(test['label'].sum())} true)"
    )
    return train, test


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="training_prep_cnv_pipeline",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--call_vcf", required=True, help="Depth CNV calls (any FILTER), or pieces of known events")
    parser.add_argument("--truth_vcf", required=True, help="Truth SV VCF (for pieces: <prefix>.truth.vcf.gz)")
    parser.add_argument("--benchmark_bed", required=True, help="Benchmark regions of the truth")
    parser.add_argument("--train_chromosomes", nargs="+", required=True, help="Chromosomes used for training")
    parser.add_argument("--output_prefix", required=True, help="Output prefix (<prefix>.train.h5, <prefix>.test.h5)")
    parser.add_argument("--min_len", type=int, default=500, help="Minimal call length (default: 500)")
    parser.add_argument("--verbosity", default="INFO", help="Verbosity: ERROR, WARNING, INFO, DEBUG")
    return parser.parse_args(argv)


def run(argv: list[str]):
    args = parse_args(argv[1:])
    logger.setLevel(getattr(logging, args.verbosity))
    training_prep_cnv(
        args.call_vcf,
        args.truth_vcf,
        args.benchmark_bed,
        args.train_chromosomes,
        args.output_prefix,
        args.min_len,
    )


def main():
    run(sys.argv)


if __name__ == "__main__":
    main()
