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
#    Prepare training data for the depth CNV filtering model (VcfType.CNV_DEPTH)
"""
Prepare training data for the depth CNV filtering model.

Every call of at least min_len inside the benchmark BED is labelled with truvari bench against the truth: matched -> 1,
unmatched -> 0 (pctovl 0.5, pctsize 0.5, pctseq 0, refdist 100 kb, duplications matched to insertions). Calls of other
samples with known events (e.g. cell lines) can be added as positives: calls of the event's type with at least half of
their length inside an event. Writes <prefix>.train.h5 and <prefix>.test.h5, split by chromosome, with the INFO
columns the filter reads and "label": the input of

    train_models_pipeline --vcf_type cnv_depth --gt_type approximate --train_dfs <prefix>.train.h5 --test_dfs ...
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile

import pandas as pd
import pysam
from ugbio_core.exec_utils import print_and_execute
from ugbio_core.logger import logger
from ugbio_core.vcfbed import vcftools

from ugbio_filtering import transformers
from ugbio_filtering.tprep_constants import VcfType

TRUVARI_OPTIONS = [
    "--pctseq",
    "0",
    "--sizemax",
    "-1",
    "--pctovl",
    "0.5",
    "--pctsize",
    "0.5",
    "--refdist",
    "100000",
    "--chunksize",
    "100000",
    "--dup-to-ins",
]
KEY_COLUMNS = ["chrom", "pos", "id"]  # calls may share a start (e.g. two pair DUP candidates); IDs are unique


def read_calls(call_vcf: str) -> pd.DataFrame:
    """Calls with the INFO columns the depth CNV filter reads, plus chrom, pos, id and end."""
    calls = vcftools.get_vcf_df(call_vcf, custom_info_fields=transformers.DEPTH_CNV_INFO_FIELDS)
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
    allpass = os.path.join(workdir, "calls.allpass.vcf.gz")  # all calls PASS, so truvari scores every one
    print_and_execute(
        f"bcftools view -i 'abs(INFO/SVLEN)>={min_len}' -Oz -o {selected} {call_vcf}", module_name=__name__
    )
    print_and_execute(f"bcftools annotate -x FILTER -Oz -o {allpass} {selected}", module_name=__name__)
    pysam.tabix_index(allpass, preset="vcf", force=True)
    bench = os.path.join(workdir, "truvari")
    options = " ".join(TRUVARI_OPTIONS)
    print_and_execute(
        f"truvari bench -b {truth_vcf} -c {allpass} -o {bench} --includebed {benchmark_bed} {options}",
        module_name=__name__,
    )
    rows = []
    for name, label in (("tp-comp", 1), ("fp", 0)):
        with pysam.VariantFile(os.path.join(bench, f"{name}.vcf.gz")) as vcf:
            rows += [(r.chrom, r.pos, r.id, label) for r in vcf]
    return pd.DataFrame(rows, columns=[*KEY_COLUMNS, "label"]).drop_duplicates(KEY_COLUMNS)


def known_event_positives(call_vcf: str, events_bed: str, min_inside: float = 0.5, min_len: int = 500) -> pd.DataFrame:
    """
    Calls of a sample with known events, labelled 1 when of the event's type and at least min_inside inside an event.

    Parameters
    ----------
    call_vcf : str
        Depth CNV calls of the sample.
    events_bed : str
        Known events: chrom, start (0-based), end, svtype (DEL / DUP).
    min_inside : float
        Minimal fraction of the call inside the event.
    min_len : int
        Minimal call length, as for the truth-labelled calls.

    Returns
    -------
    pd.DataFrame
        The positive calls with their INFO columns and label 1.
    """
    events = pd.read_csv(
        events_bed, sep="\t", header=None, usecols=[0, 1, 2, 3], names=["chrom", "start", "end", "svtype"]
    )
    calls = read_calls(call_vcf)
    start0 = calls["pos"] - 1
    keep = pd.Series(data=False, index=calls.index)
    long_enough = (calls["end"] - start0) >= min_len
    for e in events.itertuples(index=False):
        same = (calls["chrom"] == e.chrom) & (calls["svtype"] == e.svtype)
        inside = (calls["end"].clip(upper=e.end) - start0.clip(lower=e.start)).clip(lower=0)
        keep |= long_enough & same & (inside >= min_inside * (calls["end"] - start0).clip(lower=1))
    return calls[keep].assign(label=1)


def training_prep_depth_cnv(  # noqa: PLR0913
    call_vcf: str,
    truth_vcf: str,
    benchmark_bed: str,
    train_chromosomes: list[str],
    output_prefix: str,
    known_events: list[tuple[str, str]] | None = None,
    min_len: int = 500,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Write <output_prefix>.train.h5 and <output_prefix>.test.h5 for train_models_pipeline (vcf_type cnv_depth).

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
    known_events : list[tuple[str, str]], optional
        (call VCF, events BED) of samples with known events; their positives are added to the training set.
    min_len : int
        Minimal call length.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        The training and test sets.
    """
    columns = [*transformers.get_needed_features(VcfType.CNV_DEPTH), "label", *KEY_COLUMNS]
    with tempfile.TemporaryDirectory(dir=os.path.dirname(os.path.abspath(output_prefix))) as workdir:
        labels = label_calls(call_vcf, truth_vcf, benchmark_bed, workdir, min_len)
    labelled = read_calls(call_vcf).merge(labels, on=KEY_COLUMNS, how="inner")
    is_train = labelled["chrom"].isin(train_chromosomes)
    train, test = labelled[is_train], labelled[~is_train]
    extra = [known_event_positives(vcf, bed, min_len=min_len) for vcf, bed in known_events or []]
    if extra:
        train = pd.concat([train, *extra], ignore_index=True)
    train, test = train.reindex(columns=columns), test.reindex(columns=columns)
    train.to_hdf(f"{output_prefix}.train.h5", key="data", mode="w")
    test.to_hdf(f"{output_prefix}.test.h5", key="data", mode="w")
    logger.info(
        f"training rows {len(train)} ({int(train['label'].sum())} true, {sum(len(x) for x in extra)} known-event "
        f"positives); test rows {len(test)} ({int(test['label'].sum())} true)"
    )
    return train, test


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="training_prep_depth_cnv_pipeline",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--call_vcf", required=True, help="Depth CNV calls of a sample with truth (any FILTER)")
    parser.add_argument("--truth_vcf", required=True, help="Truth SV VCF")
    parser.add_argument("--benchmark_bed", required=True, help="Benchmark regions of the truth")
    parser.add_argument("--train_chromosomes", nargs="+", required=True, help="Chromosomes used for training")
    parser.add_argument("--output_prefix", required=True, help="Output prefix (<prefix>.train.h5, <prefix>.test.h5)")
    parser.add_argument(
        "--known_events",
        nargs=2,
        action="append",
        metavar=("CALL_VCF", "EVENTS_BED"),
        default=None,
        help="Calls of a sample with known events and the events (chrom, start, end, svtype); positives are added "
        "to the training set (repeatable)",
    )
    parser.add_argument("--min_len", type=int, default=500, help="Minimal call length (default: 500)")
    parser.add_argument("--verbosity", default="INFO", help="Verbosity: ERROR, WARNING, INFO, DEBUG")
    return parser.parse_args(argv)


def run(argv: list[str]):
    args = parse_args(argv[1:])
    logger.setLevel(getattr(logging, args.verbosity))
    training_prep_depth_cnv(
        args.call_vcf,
        args.truth_vcf,
        args.benchmark_bed,
        args.train_chromosomes,
        args.output_prefix,
        [tuple(x) for x in args.known_events or []],
        args.min_len,
    )


def main():
    run(sys.argv)


if __name__ == "__main__":
    main()
