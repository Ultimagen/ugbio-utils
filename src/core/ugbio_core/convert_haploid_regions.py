"""Convert non-PAR chrX/Y genotypes to haploid for male samples.

Converts diploid GT/GQ/PL in non-PAR regions of chrX/Y to haploid using
PL-based reclassification. PAR regions stay diploid.

Reference auto-detection: by default, inspects VCF contig names to pick
hg38 or b37 non-PAR coordinates. Can also accept an explicit BED file.
"""

from __future__ import annotations

import argparse
import math
import sys

import pysam

# GRCh38/hg38 non-PAR regions (complement of PAR1 + PAR2)
# Source: GRC assembly report / UCSC genome browser
_HG38_NON_PAR = [
    ("chrX", 1, 10001),
    ("chrX", 2781479, 155701383),
    ("chrX", 156030895, 156040895),
    ("chrY", 1, 10001),
    ("chrY", 2781479, 56887903),
]

# GRCh37/b37 non-PAR regions (complement of PAR1 + PAR2)
_B37_NON_PAR = [
    ("X", 1, 60001),
    ("X", 2699520, 154931044),
    ("X", 155260560, 155270560),
    ("Y", 1, 10001),
    ("Y", 2649520, 59034050),
]

_PRESETS = {
    "hg38_non_par": _HG38_NON_PAR,
    "b37_non_par": _B37_NON_PAR,
}


def _detect_reference(vcf: pysam.VariantFile) -> str:
    contigs = set(vcf.header.contigs)
    if "chrX" in contigs:
        return "hg38_non_par"
    if "X" in contigs:
        return "b37_non_par"
    raise ValueError(
        f"Cannot auto-detect reference: VCF contigs contain neither 'chrX' (hg38) "
        f"nor 'X' (b37). Use --haploid_regions with a BED file or explicit preset."
    )


def _in_regions(chrom: str, pos: int, regions: list[tuple[str, int, int]]) -> bool:
    return any(c == chrom and s < pos <= e for c, s, e in regions)


def _convert_to_haploid(variant: pysam.VariantRecord) -> pysam.VariantRecord:
    call = variant.samples[0]
    pls = call["PL"]
    num_alleles = len(variant.alts) + 1
    if len(pls) == 2:
        return variant

    un_normalized = [10 ** (pl / -10) for pl in pls]
    hom_probs = []
    for i in range(num_alleles):
        idx = int(i * (i + 1) / 2 + i)
        hom_probs.append(un_normalized[idx])
    total = sum(hom_probs)
    haploid_pls = [int(-10 * math.log10(p / total)) for p in hom_probs]
    min_pl = min(haploid_pls)
    haploid_pls = [pl - min_pl for pl in haploid_pls]

    gq = 10000
    called = 0
    for i, pl in enumerate(haploid_pls):
        if pl == 0:
            called = i
        elif pl < gq:
            gq = pl

    if call["GT"][0] is None:
        called = None
    call["GT"] = called
    call["GQ"] = gq
    call["PL"] = haploid_pls
    return variant


def convert_haploid_regions(input_vcf: str, output_vcf: str, haploid_regions: str = "auto") -> None:
    reader = pysam.VariantFile(input_vcf)

    if haploid_regions == "auto":
        preset = _detect_reference(reader)
        print(f"Auto-detected reference: {preset}", file=sys.stderr)
        regions = _PRESETS[preset]
    elif haploid_regions in _PRESETS:
        regions = _PRESETS[haploid_regions]
    else:
        regions = []
        with open(haploid_regions) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    regions.append((parts[0], int(parts[1]), int(parts[2])))

    writer = pysam.VariantFile(output_vcf, mode="w", header=reader.header)
    for variant in reader:
        if _in_regions(variant.chrom, variant.pos, regions):
            writer.write(_convert_to_haploid(variant))
        else:
            writer.write(variant)
    writer.close()
    reader.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Convert non-PAR chrX/Y genotypes to haploid.",
        epilog="""\
--haploid_regions accepts:
  auto           Auto-detect hg38 vs b37 from VCF contigs (default)
  hg38_non_par   GRCh38 non-PAR regions (chrX/chrY naming)
  b37_non_par    GRCh37/b37 non-PAR regions (X/Y naming)
  <path.bed>     Custom BED file with 3 tab-separated columns:
                   chrom  start(0-based)  end
                 Example for a custom genome:
                   chrX  0      10000
                   chrX  2781479  155701383
                   chrY  0      10000
                   chrY  2781479  56887903
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input_vcf", required=True)
    parser.add_argument("--output_vcf", required=True)
    parser.add_argument(
        "--haploid_regions",
        default="auto",
        help="'auto' (detect from VCF), 'hg38_non_par', 'b37_non_par', or path to BED file (default: auto)",
    )
    args = parser.parse_args(argv)
    convert_haploid_regions(args.input_vcf, args.output_vcf, args.haploid_regions)


if __name__ == "__main__":
    main()
