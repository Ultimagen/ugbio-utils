"""
cn.mops candidates for the depth CNV caller (optional, with the cn.mops calls of the sample).

cnmops_cnv_candidates: cn.mops calls without a same-type call with >= 50% reciprocal overlap are added to the calls
with CNV_SOURCE=cn.mops and the depth features of the other candidates (cnv_candidate_utils.score_with_depth),
without the depth-change check (cn.mops finds losses in low-mappability sequence where the all-read depth shows none).
"""

import argparse
import sys

import pandas as pd
import pysam
from ugbio_cnv.cnv_candidate_utils import CandidateParams, add_candidates_to_calls, score_with_depth
from ugbio_cnv.cnv_vcf_consts import CNV_SOURCE_CNMOPS
from ugbio_cnv.depth_cnv_caller import read_sex_chromosomes
from ugbio_core.logger import logger


def cnmops_cnv_candidates(
    input_vcf: str,
    cnmops_vcf: str,
    depth_mq0: str,
    depth_mq20: str,
    annotation: str,
    output_vcf: str,
    params: CandidateParams | None = None,
) -> int:
    """
    Add the cn.mops calls of cnmops_vcf that input_vcf does not have (>= min_overlap reciprocal, same type) as
    CNV_SOURCE=cn.mops candidates; returns the number added. Every DEL / DUP >= min_len is taken whatever its FILTER;
    calls on sex chromosomes the depth caller does not call, and calls without depth flanks, are skipped.
    """
    p = params or CandidateParams()
    sex = read_sex_chromosomes(input_vcf)
    with pysam.VariantFile(cnmops_vcf) as v:
        cand = pd.DataFrame(
            [
                (r.chrom, r.start, r.stop, r.info["SVTYPE"], 0)
                for r in v
                if r.info.get("SVTYPE") in ("DEL", "DUP")
                and r.chrom not in sex.skipped
                and r.stop - r.start >= p.min_len
            ],
            columns=["chrom", "start", "end", "svtype", "n_reads"],
        )
    scored = (
        score_with_depth(cand, depth_mq0, depth_mq20, annotation, sex, p, require_change=False)
        if len(cand)
        else pd.DataFrame(columns=["chrom", "start", "end", "svtype"])
    )
    n = add_candidates_to_calls(
        input_vcf, scored, output_vcf, p, source=CNV_SOURCE_CNMOPS, id_prefix="CNMOPS", count_tag=None, ci=(-500, 500)
    )
    logger.info(f"{output_vcf}: cn.mops calls {len(cand)}, added {n}")
    return n


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="cnmops_cnv_candidates", description="Add the cn.mops calls the depth caller does not have as candidates"
    )
    parser.add_argument("--input-vcf", required=True, help="Calls so far (split_read / pair_cnv_candidates output)")
    parser.add_argument("--cnmops-vcf", required=True, help="cn.mops calls of the sample (process_cnvs output)")
    parser.add_argument("--depth-mq0", required=True, help="mosdepth regions.bed.gz, all reads (with .csi)")
    parser.add_argument("--depth-mq20", required=True, help="mosdepth regions.bed.gz, MAPQ>=20 (with .csi)")
    parser.add_argument("--annotation", required=True, help="Reference annotation parquet")
    parser.add_argument("--output-vcf", required=True, help="Output .vcf.gz")
    return parser.parse_args(argv)


def run(argv: list[str]):
    a = parse_args(argv[1:])
    cnmops_cnv_candidates(a.input_vcf, a.cnmops_vcf, a.depth_mq0, a.depth_mq20, a.annotation, a.output_vcf)


def main():
    run(sys.argv)


if __name__ == "__main__":
    main()
