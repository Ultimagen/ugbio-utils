"""
System test for `ugbio_cnv.lpa_caller`.

Exercises marker-based haplotype splitting and per-site small-variant inference
against a real Ultima BAM slice (KIV-2 region of Coriell HG01046).

End-to-end depth estimation (`_estimate_total_kiv2_copy_number`) requires the
full hg38 reference and the autosomal norm regions, so it is verified offline
on the 17-sample DRAGEN comparison rather than in CI. Here we pass the
DRAGEN-reported KIV-2 copy number for HG01046 as a literal so the test can run
against just the (~4 MB) KIV-2 slice.
"""

from pathlib import Path

import pysam
import pytest
from ugbio_cnv.lpa_caller import (
    DEFAULT_KIV2_CHROM,
    DEFAULT_MARKER_VARIANTS,
    DEFAULT_SMALL_VARIANTS,
    _build_marker,
    _call_heterozygous_markers,
    _call_small_variant,
    _count_alleles_at_positions,
)

# HG01046 DRAGEN-reported KIV-2 copy number (targeted-caller v4.4.7).
HG01046_KIV2_CN = 45.72


@pytest.fixture
def resources_dir() -> Path:
    return Path(__file__).parent.parent / "resources"


@pytest.fixture
def hg01046_bam(resources_dir: Path) -> Path:
    return resources_dir / "lpa_caller_HG01046_kiv2.bam"


def test_marker_split_hg01046(hg01046_bam: Path) -> None:
    markers = [_build_marker(s) for s in DEFAULT_MARKER_VARIANTS]
    with pysam.AlignmentFile(str(hg01046_bam), "rb") as bam:
        ref_cn, alt_cn, call_type, per_marker = _call_heterozygous_markers(
            bam, DEFAULT_KIV2_CHROM, markers, HG01046_KIV2_CN
        )

    assert call_type == "Heterozygous markers call"
    assert ref_cn is not None
    assert alt_cn is not None
    # Production VCF emitted 25 / 21; trimmed-mean correction may shift the
    # split by <=1 copy on either side.
    assert ref_cn == pytest.approx(24.7, abs=1.0)
    assert alt_cn == pytest.approx(21.0, abs=1.0)
    assert ref_cn + alt_cn == pytest.approx(HG01046_KIV2_CN, abs=1e-6)

    # Both markers must be informative (hundreds of supporting reads at 30x
    # across 6 KIV-2 copies).
    for counts in per_marker:
        assert counts.informative > 100


@pytest.mark.parametrize("hgvs", ["LPA:4925G>A", "LPA:4733G>A"])
def test_small_variants_hg01046_negative(hg01046_bam: Path, hgvs: str) -> None:
    """HG01046 carries no LPA small variants — both sites should call alt_cn=0."""
    spec = next(s for s in DEFAULT_SMALL_VARIANTS if s["hgvs"] == hgvs)
    sv = _build_marker(spec)

    with pysam.AlignmentFile(str(hg01046_bam), "rb") as bam:
        counts = _count_alleles_at_positions(bam, DEFAULT_KIV2_CHROM, sv.positions, sv.ref, sv.alt)
    call = _call_small_variant(counts, HG01046_KIV2_CN, sv.hgvs, sv.ref, sv.alt, sv.positions)

    assert counts.informative > 100
    assert counts.alt == 0
    assert call.alt_copy_number == 0
    assert call.qual == 0.0
