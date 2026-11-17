"""IGV batch file for a detected sequins genome: reads, ground-truth VCF and a BED of the discordant positions."""

import argparse
import csv
import os
import subprocess
import sys


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="sequins_igv", description=__doc__)
    ap.add_argument("--genome-name", required=True, help="Sequins genome name (used in the output file names)")
    ap.add_argument("--ref-fasta", required=True, help="Reference fasta of the sequins genome (.fai next to it)")
    ap.add_argument("--cram", required=True, help="CRAM with the reads (.crai next to it)")
    ap.add_argument("--truth-vcf", required=True, help="Ground-truth VCF (.tbi next to it)")
    ap.add_argument(
        "--discordant-table", required=True, help="Discordant variants table (class, type, chrom, pos, ref, alt, ...)"
    )
    ap.add_argument(
        "--output-prefix", required=True, help="Prefix of the outputs: <prefix>.igv_batch.txt, <prefix>.discordant.bed"
    )
    return ap.parse_args(argv)


def discordant_table_to_bed(table: str, bed: str) -> int:
    """Write the discordant variants as a sorted BED (name: class_type_ref>alt); returns the number of records."""
    with open(table, newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    records = []
    for r in rows:
        start = int(r["pos"]) - 1
        records.append((r["chrom"], start, start + len(r["ref"]), f"{r['class']}_{r['type']}_{r['ref']}>{r['alt']}"))
    records.sort(key=lambda rec: (rec[0], rec[1], rec[2], rec[3]))
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
    discordant_table_to_bed(args.discordant_table, bed)
    with open(batch, "w") as fh:
        fh.write(build_batch(args.ref_fasta, args.cram, args.truth_vcf, bed, first_contig(args.truth_vcf)))
    return batch, bed


def main() -> None:
    run(sys.argv)


if __name__ == "__main__":
    main()
