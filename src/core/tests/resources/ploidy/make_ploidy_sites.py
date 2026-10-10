"""Build a de-identified ploidy test fixture from a VCF.

Stores only what estimate_ploidy uses: chrom, position (randomly shifted by 3-5 bp), DP, AD counts and the het
flag. No alleles, genotypes or INFO are kept.

Usage: python make_ploidy_sites.py <vcf> <sample_id> <exclude_regions_bed> <out.parquet>
"""

import sys

import numpy as np
import pandas as pd
from ugbio_core.estimate_ploidy import _iter_vcf_sites

COLUMNS = ["chrom", "pos", "dp", "ref_ad", "alt_ad", "is_het"]


def main(vcf: str, sample_id: str, exclude_regions_bed: str, out: str) -> None:
    sites = pd.DataFrame(_iter_vcf_sites(vcf, sample_id, exclude_regions_bed), columns=COLUMNS)
    rng = np.random.default_rng(0)
    sites["pos"] += rng.integers(3, 6, len(sites)) * rng.choice([-1, 1], len(sites))
    int16_max = np.iinfo(np.int16).max
    sites[["dp", "ref_ad", "alt_ad"]] = sites[["dp", "ref_ad", "alt_ad"]].clip(upper=int16_max)
    sites = sites.astype(
        {
            "chrom": pd.CategoricalDtype(list(dict.fromkeys(sites["chrom"]))),
            "pos": "int32",
            "dp": "int16",
            "ref_ad": "int16",
            "alt_ad": "int16",
            "is_het": "bool",
        }
    )
    sites.sort_values(["chrom", "pos"]).to_parquet(out, index=False, compression="zstd")


if __name__ == "__main__":
    main(*sys.argv[1:])
