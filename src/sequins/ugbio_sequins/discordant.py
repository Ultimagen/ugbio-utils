"""Discordant (FP and FN) variants, read from the rtg vcfeval FP and FN VCFs."""

import subprocess
from dataclasses import dataclass

COLUMNS = ["class", "type", "chrom", "pos", "ref", "alt", "gt", "qual", "filter", "DP"]


@dataclass
class DiscordantVariant:
    variant_class: str  # "FP" or "FN"
    variant_type: str  # "snps" or "indels"
    chrom: str
    pos: int
    ref: str
    alt: str
    gt: str
    qual: str
    filter: str
    dp: str

    def as_row(self) -> list[str]:
        return [
            self.variant_class,
            self.variant_type,
            self.chrom,
            str(self.pos),
            self.ref,
            self.alt,
            self.gt,
            self.qual,
            self.filter,
            self.dp,
        ]


def _sample_field(fmt: str, sample: str, key: str) -> str | None:
    keys = fmt.split(":")
    values = sample.split(":")
    return values[keys.index(key)] if key in keys and keys.index(key) < len(values) else None


def read_vcf(vcf: str, variant_class: str) -> list[DiscordantVariant]:
    """Records of a VCF as discordant variants. DP is FORMAT/DP of the first sample, else INFO/DP, else ".".

    A SNP is a record whose REF and ALT alleles are all single bases; anything else is an indel.
    """
    cmd = ["bcftools", "view", "-H", vcf]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout  # noqa: S603
    variants = []
    for line in out.splitlines():
        f = line.split("\t")
        chrom, pos, ref, alt, qual, flt, info = f[0], int(f[1]), f[3], f[4], f[5], f[6], f[7]
        gt = dp = None
        if len(f) > 9:  # noqa: PLR2004
            gt = _sample_field(f[8], f[9], "GT")
            dp = _sample_field(f[8], f[9], "DP")
        if dp is None:
            dp = next((kv[3:] for kv in info.split(";") if kv.startswith("DP=")), None)
        is_snp = len(ref) == 1 and all(len(a) == 1 for a in alt.split(","))
        variants.append(
            DiscordantVariant(
                variant_class, "snps" if is_snp else "indels", chrom, pos, ref, alt, gt or ".", qual, flt, dp or "."
            )
        )
    return variants


def read_discordant(fp_vcf: str, fn_vcf: str) -> list[DiscordantVariant]:
    """FP then FN variants, each sorted by position."""
    out = []
    for vcf, variant_class in ((fp_vcf, "FP"), (fn_vcf, "FN")):
        out += sorted(read_vcf(vcf, variant_class), key=lambda v: (v.chrom, v.pos, v.ref, v.alt))
    return out
