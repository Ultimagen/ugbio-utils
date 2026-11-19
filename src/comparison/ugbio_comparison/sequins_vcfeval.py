"""Sequins vcfeval: compare calls to the sequins truth with vcfeval and save the concordance dataframe."""

import argparse
import sys
from pathlib import Path

from ugbio_comparison.vcf_comparison_utils import VcfComparisonUtils, vcf2concordance

H5_KEY = "concordance"


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="sequins_vcfeval", description=__doc__)
    ap.add_argument("--call-vcf", required=True, help="Calls (vcf.gz)")
    ap.add_argument("--truth-vcf", required=True, help="Sequins truth (vcf.gz)")
    ap.add_argument("--eval-bed", required=True, help="Regions to evaluate")
    ap.add_argument("--ref-fasta", required=True, help="Reference fasta (indexed)")
    ap.add_argument("--sdf-dir", required=True, help="rtg SDF directory of the reference")
    ap.add_argument("--output-prefix", required=True, help="Output prefix; writes <prefix>.concordance.h5")
    return ap.parse_args(argv)


def run(argv: list[str]) -> str:
    args = parse_args(argv[1:])
    concordance_vcf = VcfComparisonUtils().run_vcfeval_concordance(
        input_file=args.call_vcf,
        truth_file=args.truth_vcf,
        output_prefix=args.output_prefix,
        ref_genome=args.ref_fasta,
        sdf_index=args.sdf_dir,
        evaluation_regions=args.eval_bed,
    )
    output = f"{args.output_prefix}.concordance.h5"
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    vcf2concordance(args.call_vcf, concordance_vcf).to_hdf(output, key=H5_KEY, mode="w")
    return output


def main() -> None:
    run(sys.argv)


if __name__ == "__main__":
    main()
