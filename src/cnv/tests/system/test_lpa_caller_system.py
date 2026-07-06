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

import json
from pathlib import Path

import pysam
import pytest
from ugbio_cnv import lpa_caller as lpa_caller_mod
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
# 5% tolerance for numerical comparisons on the KIV-2 slice.
CN_TOLERANCE_FRACTION = 0.05


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
    # A handful of low-quality noise ALT bases can slip in at MIN_BASE_QUALITY=1;
    # the binomial fit still assigns alt_cn=0 because the observed alt fraction
    # is well below LPA_VARIANT_NOISE_FLOOR.
    assert counts.alt / counts.informative < 0.005
    assert call.alt_copy_number == 0
    assert call.qual == 0.0


def test_run_end_to_end_on_kiv2_bam(hg01046_bam: Path, tmp_path: Path, monkeypatch) -> None:
    """
    End-to-end CLI run against the small (~4 MB) HG01046 KIV-2 BAM slice.

    The slice has reads only inside the KIV-2 region so we cannot compute an
    autosomal depth baseline from it. Per PR review, the normalization counts
    are fed in by patching ``_estimate_total_kiv2_copy_number`` to return the
    DRAGEN-reported HG01046 KIV-2 copy number, and REF-base validation is
    stubbed because the tiny synthetic FASTA below has no real chr6 sequence.
    Every other stage of the pipeline (marker split, small-variant binomial,
    JSON/VCF writers) runs against the real BAM reads.

    HG01046 is a known negative sample for LPA:4733G>A / LPA:4925G>A, so the
    caller must agree on non-existence at both sites (``altCopyNumber == 0``,
    ``qual == 0``). Numerical CN comparisons use a 5% tolerance.
    """
    # Minimal chr6-only stub FASTA: `_write_vcf` reads a single base at
    # kiv2_start and falls back to 'N' when out of range, so a 1-base contig
    # is sufficient.
    stub_fasta = tmp_path / "chr6_stub.fa"
    stub_fasta.write_text(">chr6\nN\n")

    # Feed the KIV-2 copy number directly and skip REF-base validation (both
    # require the full hg38 reference we deliberately don't ship here).
    monkeypatch.setattr(
        lpa_caller_mod,
        "_estimate_total_kiv2_copy_number",
        lambda *args, **kwargs: HG01046_KIV2_CN,
    )
    monkeypatch.setattr(
        lpa_caller_mod,
        "_validate_marker_positions",
        lambda *args, **kwargs: None,
    )

    output_prefix = tmp_path / "hg01046"
    lpa_caller_mod.run(
        [
            "--cram",
            str(hg01046_bam),
            "--reference",
            str(stub_fasta),
            "--sample-id",
            "HG01046",
            "--output-prefix",
            str(output_prefix),
            "--no-vcf",  # exercised separately below
        ]
    )

    # ------------------------------------------------------------------
    # JSON: total CN, marker split, and small-variant existence.
    # ------------------------------------------------------------------
    json_path = tmp_path / "hg01046.targeted.json"
    assert json_path.exists()
    payload = json.loads(json_path.read_text())
    lpa = payload["lpa"]

    assert payload["sampleId"] == "HG01046"
    assert payload["genomeBuild"] == "hg38"
    assert lpa["kiv2CopyNumber"] == pytest.approx(HG01046_KIV2_CN, rel=CN_TOLERANCE_FRACTION)
    assert lpa["type"] == "Heterozygous markers call"

    ref_cn = lpa["refMarkerAlleleCopyNumber"]
    alt_cn = lpa["altMarkerAlleleCopyNumber"]
    assert ref_cn is not None and alt_cn is not None
    assert ref_cn + alt_cn == pytest.approx(HG01046_KIV2_CN, abs=1e-6)
    # Production VCF emitted 25 / 21 (see test_marker_split_hg01046).
    assert ref_cn == pytest.approx(25.0, rel=CN_TOLERANCE_FRACTION)
    assert alt_cn == pytest.approx(21.0, rel=CN_TOLERANCE_FRACTION)

    # Must agree on small variant existence: HG01046 has neither site called.
    called_hgvs = {v["hgvs"]: v for v in lpa["variants"]}
    assert set(called_hgvs) == {"LPA:4733G>A", "LPA:4925G>A"}
    for hgvs, v in called_hgvs.items():
        assert v["altCopyNumber"] == 0, f"{hgvs} should not exist in HG01046"
        assert v["qual"] == 0.0, f"{hgvs} should not exist in HG01046"

    # ------------------------------------------------------------------
    # VCF: rerun with VCF enabled and check header + record shape.
    # ------------------------------------------------------------------
    vcf_prefix = tmp_path / "hg01046_vcf"
    lpa_caller_mod.run(
        [
            "--cram",
            str(hg01046_bam),
            "--reference",
            str(stub_fasta),
            "--sample-id",
            "HG01046",
            "--output-prefix",
            str(vcf_prefix),
        ]
    )
    vcf_path = tmp_path / "hg01046_vcf.targeted.vcf.gz"
    assert vcf_path.exists()

    with pysam.BGZFile(str(vcf_path), "rb") as fh:
        vcf_text = fh.read().decode()

    header_lines = [line for line in vcf_text.splitlines() if line.startswith("##")]
    body_lines = [line for line in vcf_text.splitlines() if line and not line.startswith("#")]

    assert any(line.startswith("##fileformat=VCFv4.2") for line in header_lines)
    # HG01046 has ~46 total KIV-2 copies -> more than diploid baseline (12).
    # Expect a DUP summary record.
    dup_rows = [line for line in body_lines if "<DUP>" in line.split("\t")[4]]
    assert len(dup_rows) == 1
    total_cn_str = [kv for kv in dup_rows[0].split("\t")[7].split(";") if kv.startswith("TOTAL_CN=")]
    assert total_cn_str
    total_cn = float(total_cn_str[0].split("=", 1)[1])
    assert total_cn == pytest.approx(HG01046_KIV2_CN, rel=CN_TOLERANCE_FRACTION)

    # Small variant rows must exist for both sites at every configured repeat
    # position and must all call REF-only genotypes (no ALT alleles).
    for spec in DEFAULT_SMALL_VARIANTS:
        for pos in spec["positions"]:
            matches = [
                line for line in body_lines if line.split("\t")[1] == str(pos) and line.split("\t")[4] == spec["alt"]
            ]
            assert len(matches) == 1, f"expected 1 row at chr6:{pos} for {spec['hgvs']}"
            sample_field = matches[0].split("\t")[-1]
            gt = sample_field.split(":")[0]
            # No ALT alleles in genotype -> negative small-variant call.
            assert "1" not in gt, f"{spec['hgvs']} at {pos} unexpectedly called ALT: GT={gt}"
