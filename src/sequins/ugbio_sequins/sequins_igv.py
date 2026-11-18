"""IGV batch file for a detected sequins genome: reads, ground-truth VCF and a BED of the discordant positions."""

import argparse
import os
import subprocess
import sys

from ugbio_sequins.discordant import read_discordant


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="sequins_igv", description=__doc__)
    ap.add_argument("--genome-name", required=True, help="Sequins genome name (used in the output file names)")
    ap.add_argument("--ref-fasta", required=True, help="Reference fasta of the sequins genome (.fai next to it)")
    ap.add_argument("--cram", required=True, help="CRAM with the reads (.crai next to it)")
    ap.add_argument("--truth-vcf", required=True, help="Ground-truth VCF (.tbi next to it)")
    ap.add_argument("--fp-vcf", required=True, help="False-positive VCF (rtg vcfeval fp.vcf.gz)")
    ap.add_argument("--fn-vcf", required=True, help="False-negative VCF (rtg vcfeval fn.vcf.gz)")
    ap.add_argument(
        "--output-prefix", required=True, help="Prefix of the outputs: <prefix>.igv_batch.txt, <prefix>.discordant.bed"
    )
    return ap.parse_args(argv)


def discordant_to_bed(fp_vcf: str, fn_vcf: str, bed: str) -> int:
    """Write the discordant variants as a sorted BED (name: class_type_ref>alt); returns the number of records."""
    records = [
        (v.chrom, v.pos - 1, v.pos - 1 + len(v.ref), f"{v.variant_class}_{v.variant_type}_{v.ref}>{v.alt}")
        for v in read_discordant(fp_vcf, fn_vcf)
    ]
    records.sort()
    with open(bed, "w") as fh:
        for rec in records:
            fh.write("\t".join(map(str, rec)) + "\n")
    return len(records)


def first_contig(vcf: str) -> str:
    """Contig of the first record of the (truth) VCF, i.e. the sequins contig."""
    cmd = ["bcftools", "query", "-f", "%CHROM\n", vcf]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout  # noqa: S603
    contigs = out.split("\n", 1)[0].strip()
    if not contigs:
        raise ValueError(f"No records in {vcf}")
    return contigs


def build_batch(ref_fasta: str, cram: str, truth_vcf: str, bed: str, contig: str) -> str:
    """IGV batch script. Paths are file names, resolved by IGV relative to the batch file's directory."""
    lines = [
        "new",
        f"genome {os.path.basename(ref_fasta)}",
        f"load {os.path.basename(cram)}",
        f"load {os.path.basename(truth_vcf)}",
        f"load {os.path.basename(bed)}",
        f"goto {contig}",
    ]
    return "\n".join(lines) + "\n"


def run(argv: list[str]) -> tuple[str, str]:
    args = parse_args(argv[1:])
    bed = f"{args.output_prefix}.discordant.bed"
    batch = f"{args.output_prefix}.igv_batch.txt"
    discordant_to_bed(args.fp_vcf, args.fn_vcf, bed)
    with open(batch, "w") as fh:
        fh.write(build_batch(args.ref_fasta, args.cram, args.truth_vcf, bed, first_contig(args.truth_vcf)))
    return batch, bed


def main() -> None:
    run(sys.argv)


if __name__ == "__main__":
    main()
