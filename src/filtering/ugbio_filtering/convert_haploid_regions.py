"""Convert non-PAR sex-chromosome genotypes to haploid for male samples."""

from __future__ import annotations

import argparse

import numpy as np
import pysam
from ugbio_core.math_utils import phred, unphred


def _in_regions(chrom: str, pos: int, regions: list[tuple[str, int, int]]) -> bool:
    return any(c == chrom and s < pos <= e for c, s, e in regions)


def _read_par_regions(par_regions: str) -> dict[str, list[tuple[str, int, int]]]:
    regions_by_chrom: dict[str, list[tuple[str, int, int]]] = {}
    with open(par_regions) as par_file:
        for line in par_file:
            if not line.strip() or line.startswith("#"):
                continue
            chrom, start, end, *_ = line.rstrip().split("\t")
            regions_by_chrom.setdefault(chrom, []).append((chrom, int(start), int(end)))
    if not regions_by_chrom:
        raise ValueError("PAR regions BED contains no intervals")
    return regions_by_chrom


def _convert_to_haploid(variant: pysam.VariantRecord) -> pysam.VariantRecord:
    call = variant.samples[0]
    pls = call.get("PL")
    if pls is None:
        raise ValueError("PL field is required to convert genotypes to haploid")
    num_alleles = len(variant.alts) + 1
    if len(pls) <= num_alleles:
        return variant

    hom_pls = np.asarray([pls[i * (i + 1) // 2 + i] for i in range(num_alleles)])
    probabilities = unphred(hom_pls - hom_pls.min())
    normalized_probabilities = probabilities / probabilities.sum()
    haploid_pls = np.rint(phred(np.maximum(normalized_probabilities, 1e-300))).astype(int).tolist()
    min_pl = min(haploid_pls)
    haploid_pls = [pl - min_pl for pl in haploid_pls]

    called = 0
    for i, pl in enumerate(haploid_pls):
        if pl == 0:
            called = i
    nonzero_pls = [pl for pl in haploid_pls if pl > 0]
    gq = min(nonzero_pls) if nonzero_pls else 0

    if call["GT"][0] is None:
        called = None
    call["GT"] = (called,)
    call["GQ"] = 0 if called is None else gq
    call["PL"] = haploid_pls
    return variant


def convert_haploid_regions(input_vcf: str, output_vcf: str, par_regions: str) -> None:
    reader = pysam.VariantFile(input_vcf)

    if len(reader.header.samples) != 1:
        raise ValueError(f"Expected single-sample VCF, found {len(reader.header.samples)} samples")
    regions_by_chrom = _read_par_regions(par_regions)

    writer = pysam.VariantFile(output_vcf, mode="w", header=reader.header)
    for variant in reader:
        chrom_regions = regions_by_chrom.get(variant.chrom)
        if chrom_regions is not None and not _in_regions(variant.chrom, variant.pos, chrom_regions):
            writer.write(_convert_to_haploid(variant))
        else:
            writer.write(variant)
    writer.close()
    reader.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Convert non-PAR sex-chromosome genotypes to haploid.",
    )
    parser.add_argument("--input_vcf", required=True)
    parser.add_argument("--output_vcf", required=True)
    parser.add_argument("--par_regions", required=True, help="PAR regions BED file (0-based, half-open)")
    args = parser.parse_args(argv)
    convert_haploid_regions(args.input_vcf, args.output_vcf, args.par_regions)


if __name__ == "__main__":
    main()
