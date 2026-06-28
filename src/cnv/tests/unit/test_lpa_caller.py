"""Unit tests for the LPA KIV-2 targeted caller.

Covers the pure-Python helpers (marker projection, GC fraction/bucketing,
trimmed mean, binomial/Phred math, marker and small-variant calling,
CLI region parsing, and VCF row formatting). I/O paths that need a real BAM
are exercised via mocked ``pysam`` objects.
"""

from __future__ import annotations

import argparse
import gzip
import json
from unittest.mock import MagicMock

import numpy as np
import pytest
from ugbio_cnv import lpa_caller as lpa_caller_mod
from ugbio_cnv.lpa_caller import (
    DEFAULT_KIV2_REPEAT_COUNT,
    DEFAULT_KIV2_START,
    DEFAULT_KIV2_UNIT_LEN,
    GC_BIN_SIZE,
    GC_BUCKET_WIDTH,
    HET_MARKER_ALT_FRACTION_MAX,
    HET_MARKER_ALT_FRACTION_MIN,
    MAX_PHRED_QUAL,
    AlleleCounts,
    LpaCall,
    MarkerSpec,
    SmallVariantCall,
    _bucket_gc,
    _build_marker,
    _call_heterozygous_markers,
    _call_small_variant,
    _count_alleles_at_positions,
    _estimate_total_kiv2_copy_number,
    _format_dup_record,
    _format_small_variant_record,
    _gc_fraction,
    _mean_depth_over_bins,
    _parse_norm_regions,
    _trimmed_mean,
    _validate_marker_positions,
    _write_json,
    _write_vcf,
)

# ----------------------------------------------------------------------------
# Data classes
# ----------------------------------------------------------------------------


class TestAlleleCounts:
    def test_totals(self):
        c = AlleleCounts(ref=4, alt=3, other=2)
        assert c.total == 9
        assert c.informative == 7

    def test_defaults(self):
        c = AlleleCounts()
        assert c.total == 0
        assert c.informative == 0


# ----------------------------------------------------------------------------
# Marker projection
# ----------------------------------------------------------------------------


class TestBuildMarker:
    def test_explicit_positions_used_verbatim(self):
        spec = {
            "hgvs": "LPA:296T>G",
            "ref": "T",
            "alt": "G",
            "positions": [100, 200, 300, 400, 500, 600],
        }
        m = _build_marker(spec, kiv2_start=1, unit_len=10, n_repeats=2)
        # n_repeats / unit_len ignored when positions provided.
        assert m.positions == [100, 200, 300, 400, 500, 600]
        assert m.hgvs == "LPA:296T>G"
        assert m.ref == "T"
        assert m.alt == "G"

    def test_extrapolated_positions(self):
        spec = {"hgvs": "X:1A>C", "ref": "A", "alt": "C", "first_pos": 105}
        m = _build_marker(spec, kiv2_start=100, unit_len=10, n_repeats=4)
        # offset = 5; positions = 100+5, 100+15, 100+25, 100+35
        assert m.positions == [105, 115, 125, 135]


# ----------------------------------------------------------------------------
# GC helpers
# ----------------------------------------------------------------------------


class _FakeFasta:
    """Tiny pyfaidx.Fasta stand-in: fake[chrom][a:b] returns a string slice."""

    class _Contig:
        def __init__(self, seq: str):
            self._seq = seq

        def __getitem__(self, sl):
            return self._seq[sl]

    def __init__(self, sequences):
        self._sequences = sequences

    def keys(self):
        return self._sequences.keys()

    def __getitem__(self, chrom):
        return self._Contig(self._sequences.get(chrom, ""))


class TestGcHelpers:
    def test_gc_fraction_basic(self):
        fasta = _FakeFasta({"chr1": "AAGCGC"})
        # 4/6 GC
        assert _gc_fraction(fasta, "chr1", 1, 6) == pytest.approx(4 / 6)

    def test_gc_fraction_handles_lowercase(self):
        fasta = _FakeFasta({"chr1": "aagcgc"})
        assert _gc_fraction(fasta, "chr1", 1, 6) == pytest.approx(4 / 6)

    def test_gc_fraction_ignores_n(self):
        fasta = _FakeFasta({"chr1": "NNGCNN"})
        # Only 2 valid bases counted: both GC
        assert _gc_fraction(fasta, "chr1", 1, 6) == pytest.approx(1.0)

    def test_gc_fraction_empty_sequence(self):
        fasta = _FakeFasta({"chr1": ""})
        assert _gc_fraction(fasta, "chr1", 1, 10) == 0.0

    def test_gc_fraction_all_n(self):
        fasta = _FakeFasta({"chr1": "NNNNN"})
        assert _gc_fraction(fasta, "chr1", 1, 5) == 0.0

    @pytest.mark.parametrize(
        ("gc", "expected_bucket"),
        [
            (0.00, GC_BUCKET_WIDTH / 2),
            (0.02, GC_BUCKET_WIDTH / 2),
            (0.05, GC_BUCKET_WIDTH * 1.5),
            (0.47, GC_BUCKET_WIDTH * 9.5),
            (0.99, GC_BUCKET_WIDTH * 19.5),
        ],
    )
    def test_bucket_gc(self, gc, expected_bucket):
        assert _bucket_gc(gc) == pytest.approx(expected_bucket)


# ----------------------------------------------------------------------------
# Reference validation
# ----------------------------------------------------------------------------


def _make_fake_fasta(bases_by_position, contigs=("chr6",)):
    """Build a pyfaidx-style fake where bases_by_position[(chrom, 1based_pos)] picks the base."""
    by_contig = {c: ["N"] * 1000 for c in contigs}
    for (chrom, pos), base in bases_by_position.items():
        if chrom in by_contig and 1 <= pos <= 1000:
            by_contig[chrom][pos - 1] = base
    return _FakeFasta({c: "".join(seq) for c, seq in by_contig.items()})


class TestValidateMarkerPositions:
    def _markers(self):
        return [
            MarkerSpec(hgvs="LPA:test", ref="A", alt="G", positions=[100, 200, 300]),
        ]

    def test_matching_ref_bases_pass(self):
        fasta = _make_fake_fasta({("chr6", 100): "A", ("chr6", 200): "A", ("chr6", 300): "A"})
        # Should not raise.
        _validate_marker_positions(fasta, "chr6", self._markers(), genome_build="hg38")

    def test_case_insensitive_match(self):
        fasta = _make_fake_fasta({("chr6", 100): "a", ("chr6", 200): "a", ("chr6", 300): "a"})
        _validate_marker_positions(fasta, "chr6", self._markers(), genome_build="hg38")

    def test_mismatch_raises_with_helpful_message(self):
        fasta = _make_fake_fasta({("chr6", 100): "A", ("chr6", 200): "T", ("chr6", 300): "C"})
        with pytest.raises(RuntimeError) as exc:
            _validate_marker_positions(fasta, "chr6", self._markers(), genome_build="hg38")
        msg = str(exc.value)
        assert "hg38" in msg
        assert "LPA:test" in msg
        # Both mismatched sites should be reported.
        assert "chr6:200" in msg
        assert "chr6:300" in msg

    def test_missing_contig_raises(self):
        fasta = _make_fake_fasta({}, contigs=("1", "2"))
        with pytest.raises(RuntimeError) as exc:
            _validate_marker_positions(fasta, "chr6", self._markers(), genome_build="hg38")
        assert "chr6" in str(exc.value)

    def test_many_mismatches_truncated_in_message(self):
        markers = [
            MarkerSpec(
                hgvs="LPA:bulk",
                ref="A",
                alt="G",
                positions=list(range(1, 11)),
            )
        ]
        # Every position returns 'C' -> 10 mismatches
        fasta = _make_fake_fasta({("chr6", p): "C" for p in range(1, 11)})
        with pytest.raises(RuntimeError) as exc:
            _validate_marker_positions(fasta, "chr6", markers, genome_build="hg38")
        assert "+5 more" in str(exc.value)


# ----------------------------------------------------------------------------
# Trimmed mean
# ----------------------------------------------------------------------------


class TestTrimmedMean:
    def test_empty(self):
        assert _trimmed_mean([]) == 0.0

    def test_trim_removes_tails(self):
        # 0.10 of 10 = 1 trimmed per side, leaves [2..9] (mean=5.5)
        values = list(range(1, 11))
        assert _trimmed_mean(values, trim=0.1) == pytest.approx(5.5)

    def test_full_trim_falls_back_to_mean(self):
        # trim large enough that 2*k >= size -> ungrimmed mean
        assert _trimmed_mean([1, 2, 3], trim=0.5) == pytest.approx(2.0)

    def test_handles_numpy_array(self):
        arr = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        # trim=0.2 -> k=1 -> mean of [20, 30, 40]
        assert _trimmed_mean(arr, trim=0.2) == pytest.approx(30.0)


# ----------------------------------------------------------------------------
# Depth estimation (mocked pysam fetch)
# ----------------------------------------------------------------------------


class _FakeRead:
    """Minimal pysam AlignedSegment stand-in for _mean_depth_over_bins."""

    def __init__(
        self,
        reference_start: int,
        reference_end: int,
        mapping_quality: int = 60,
        *,
        is_unmapped: bool = False,
        is_secondary: bool = False,
        is_supplementary: bool = False,
        is_duplicate: bool = False,
        is_qcfail: bool = False,
    ):
        self.reference_start = reference_start
        self.reference_end = reference_end
        self.mapping_quality = mapping_quality
        self.is_unmapped = is_unmapped
        self.is_secondary = is_secondary
        self.is_supplementary = is_supplementary
        self.is_duplicate = is_duplicate
        self.is_qcfail = is_qcfail


def _make_bam_with_reads(reads_by_region):
    """Return a MagicMock bam whose ``fetch(chrom, start, end)`` yields the reads
    pre-staged under the matching (chrom, start, end) key. Other arg shapes
    default to an empty iterator.
    """
    bam = MagicMock()

    def fetch(chrom, start, end):
        return iter(reads_by_region.get((chrom, start, end), []))

    bam.fetch.side_effect = fetch
    return bam


def _uniform_reads(start0: int, length: int, depth: int, read_len: int = 100):
    """Generate reads so every base in [start0, start0+length) is covered by
    exactly ``depth`` reads. Reads are tiled with step == read_len and ``depth``
    copies are stacked at each offset.
    """
    reads = []
    for offset in range(0, length, read_len):
        rs = start0 + offset
        re = min(start0 + length, rs + read_len)
        for _ in range(depth):
            reads.append(_FakeRead(rs, re))
    return reads


class TestMeanDepthOverBins:
    def test_empty_bam_returns_no_buckets(self):
        bam = _make_bam_with_reads({})
        # Use a GC-50% sequence so bucket lookup is deterministic if any depth
        # appears (here none does).
        seq_len = 3 * GC_BIN_SIZE
        fasta = _FakeFasta({"chr1": "GC" * (seq_len // 2)})
        out = _mean_depth_over_bins(bam, fasta, [("chr1", 1, seq_len)])
        # Bins exist but each has zero depth -> still recorded as 0.0.
        assert sum(len(v) for v in out.values()) == 3
        for depths in out.values():
            assert all(d == 0.0 for d in depths)

    def test_region_smaller_than_bin_skipped(self):
        bam = _make_bam_with_reads({})
        fasta = _FakeFasta({"chr1": "A" * 100})
        # Region length < GC_BIN_SIZE -> no bins emitted.
        out = _mean_depth_over_bins(bam, fasta, [("chr1", 1, 100)])
        assert out == {}

    def test_uniform_depth_recovered(self):
        region_len = 2 * GC_BIN_SIZE
        reads = _uniform_reads(start0=0, length=region_len, depth=5, read_len=100)
        bam = _make_bam_with_reads({("chr1", 0, region_len): reads})
        fasta = _FakeFasta({"chr1": "GC" * (region_len // 2)})
        out = _mean_depth_over_bins(bam, fasta, [("chr1", 1, region_len)])
        all_depths = [d for depths in out.values() for d in depths]
        assert len(all_depths) == 2
        # Allow a small tolerance for tiling edge effects.
        for d in all_depths:
            assert d == pytest.approx(5.0, rel=0.05)

    def test_low_mapq_reads_filtered_out(self):
        region_len = GC_BIN_SIZE
        good = _uniform_reads(0, region_len, depth=3, read_len=100)
        bad = [_FakeRead(0, 100, mapping_quality=0) for _ in range(50)]
        bam = _make_bam_with_reads({("chr1", 0, region_len): good + bad})
        fasta = _FakeFasta({"chr1": "GC" * (region_len // 2)})
        out = _mean_depth_over_bins(bam, fasta, [("chr1", 1, region_len)], min_mapq=10)
        depths = [d for depths in out.values() for d in depths]
        assert len(depths) == 1
        # Only the MAPQ-passing reads contribute.
        assert depths[0] == pytest.approx(3.0, rel=0.05)

    def test_duplicate_and_secondary_reads_filtered_out(self):
        region_len = GC_BIN_SIZE
        good = _uniform_reads(0, region_len, depth=3, read_len=100)
        junk = [
            _FakeRead(0, 100, is_duplicate=True),
            _FakeRead(0, 100, is_secondary=True),
            _FakeRead(0, 100, is_supplementary=True),
            _FakeRead(0, 100, is_unmapped=True),
            _FakeRead(0, 100, is_qcfail=True),
        ] * 20
        bam = _make_bam_with_reads({("chr1", 0, region_len): good + junk})
        fasta = _FakeFasta({"chr1": "GC" * (region_len // 2)})
        out = _mean_depth_over_bins(bam, fasta, [("chr1", 1, region_len)])
        depths = [d for depths in out.values() for d in depths]
        assert depths[0] == pytest.approx(3.0, rel=0.05)


class TestEstimateTotalKiv2CopyNumber:
    """Drive the end-to-end depth-CN math via _mean_depth_over_bins monkeypatch.

    Patching ``_mean_depth_over_bins`` is cleaner than handcrafting reads for
    both the norm and KIV-2 regions; the GC-correction code path is exercised
    by feeding compatible buckets.
    """

    @staticmethod
    def _patch_depths(monkeypatch, depths_by_region):
        def fake(bam, fasta, regions, **_kwargs):
            out: dict[float, list[float]] = {}
            for r in regions:
                for gc, depths in depths_by_region.get(tuple(r), {}).items():
                    out.setdefault(gc, []).extend(depths)
            return out

        monkeypatch.setattr(lpa_caller_mod, "_mean_depth_over_bins", fake)

    def test_diploid_call_when_kiv2_matches_baseline(self, monkeypatch):
        # KIV-2 depth == autosomal baseline -> CN = 2 * n_ref_repeats = 12.
        baseline = {("chr1", 1, 1000): {0.5: [30.0] * 20}}
        kiv2 = {("chr6", 100, 200): {0.5: [30.0] * 4}}
        self._patch_depths(monkeypatch, {**baseline, **kiv2})
        bam = MagicMock()
        fasta = MagicMock()
        cn = _estimate_total_kiv2_copy_number(
            bam, fasta, "chr6", 100, 200, n_ref_repeats=6, norm_regions=[("chr1", 1, 1000)]
        )
        assert cn == pytest.approx(12.0)

    def test_amplified_kiv2_scales_linearly(self, monkeypatch):
        # KIV-2 depth 1.5x baseline -> CN = 1.5 * 2 * 6 = 18.
        baseline = {("chr1", 1, 1000): {0.5: [30.0] * 20}}
        kiv2 = {("chr6", 100, 200): {0.5: [45.0] * 4}}
        self._patch_depths(monkeypatch, {**baseline, **kiv2})
        cn = _estimate_total_kiv2_copy_number(
            MagicMock(), MagicMock(), "chr6", 100, 200, n_ref_repeats=6, norm_regions=[("chr1", 1, 1000)]
        )
        assert cn == pytest.approx(18.0)

    def test_gc_correction_uses_matching_bucket(self, monkeypatch):
        # Baseline depth differs between two GC buckets; KIV-2 sits entirely in
        # the high-GC bucket. After GC scaling the KIV-2 mean is normalized
        # against the same-bucket baseline, not the overall mean.
        baseline = {
            ("chr1", 1, 1000): {
                0.45: [20.0] * 20,
                0.55: [40.0] * 20,
            }
        }
        # KIV-2 depth = 60 in the 0.55 bucket -> 60/40 = 1.5x same-bucket baseline.
        kiv2 = {("chr6", 100, 200): {0.55: [60.0] * 4}}
        self._patch_depths(monkeypatch, {**baseline, **kiv2})
        cn = _estimate_total_kiv2_copy_number(
            MagicMock(), MagicMock(), "chr6", 100, 200, n_ref_repeats=6, norm_regions=[("chr1", 1, 1000)]
        )
        assert cn == pytest.approx(18.0)

    def test_zero_baseline_raises(self, monkeypatch):
        # All baseline bins return zero -> trimmed mean is 0 -> RuntimeError.
        self._patch_depths(monkeypatch, {("chr1", 1, 1000): {0.5: [0.0] * 20}})
        with pytest.raises(RuntimeError, match="Normalization regions have zero coverage"):
            _estimate_total_kiv2_copy_number(
                MagicMock(), MagicMock(), "chr6", 100, 200, n_ref_repeats=6, norm_regions=[("chr1", 1, 1000)]
            )

    def test_no_kiv2_bins_raises(self, monkeypatch):
        baseline = {("chr1", 1, 1000): {0.5: [30.0] * 20}}
        # No depth recorded for the KIV-2 region.
        self._patch_depths(monkeypatch, {**baseline, ("chr6", 100, 200): {}})
        with pytest.raises(RuntimeError, match="No KIV-2 depth bins"):
            _estimate_total_kiv2_copy_number(
                MagicMock(), MagicMock(), "chr6", 100, 200, n_ref_repeats=6, norm_regions=[("chr1", 1, 1000)]
            )


# ----------------------------------------------------------------------------
# Allele counting at pileup positions
# ----------------------------------------------------------------------------


class _FakePileupRead:
    def __init__(self, base, *, is_del=False, is_refskip=False, indel=0):
        self.is_del = is_del
        self.is_refskip = is_refskip
        self.indel = indel
        self.query_position = 0
        aln = MagicMock()
        aln.is_secondary = False
        aln.is_supplementary = False
        aln.is_duplicate = False
        aln.is_qcfail = False
        aln.is_unmapped = False
        aln.query_sequence = base
        self.alignment = aln


class _FakePileupColumn:
    def __init__(self, reference_pos, reads):
        self.reference_pos = reference_pos
        self.pileups = reads


class TestCountAllelesAtPositions:
    def test_case_insensitive_allele_matching(self):
        # Lowercase query bases (e.g. soft-masked reference, some aligners)
        # must match the configured uppercase REF/ALT, not fall through to
        # `other` and bias the allele fraction.
        bam = MagicMock()
        bam.pileup.return_value = [
            _FakePileupColumn(
                reference_pos=99,
                reads=(
                    [_FakePileupRead("a")] * 3
                    + [_FakePileupRead("A")] * 2
                    + [_FakePileupRead("g")] * 4
                    + [_FakePileupRead("G")] * 1
                    + [_FakePileupRead("c")]  # 'other'
                ),
            )
        ]
        counts = _count_alleles_at_positions(bam, "chr6", [100], ref_allele="A", alt_allele="G")
        assert counts.ref == 5
        assert counts.alt == 5
        assert counts.other == 1


# ----------------------------------------------------------------------------
# Small-variant calling
# ----------------------------------------------------------------------------


class TestCallSmallVariant:
    def test_no_coverage_returns_zero_call(self):
        counts = AlleleCounts(ref=0, alt=0)
        call = _call_small_variant(
            counts,
            kiv2_copy_number=12.0,
            hgvs="LPA:4733G>A",
            ref_allele="C",
            alt_allele="T",
            positions=[1, 2, 3],
        )
        assert call.alt_copy_number == 0
        assert call.qual == 0.0
        assert call.alt_copy_number_quality == 0.0
        assert call.ref_count == 0
        assert call.alt_count == 0
        assert call.positions == [1, 2, 3]

    def test_pure_reference_yields_alt_cn_zero(self):
        counts = AlleleCounts(ref=200, alt=0)
        call = _call_small_variant(
            counts,
            kiv2_copy_number=12.0,
            hgvs="LPA:4733G>A",
            ref_allele="C",
            alt_allele="T",
            positions=[1],
        )
        assert call.alt_copy_number == 0
        # By VCF convention QUAL is 0 when the best call is REF.
        assert call.qual == 0.0

    def test_half_alt_signal_for_cn12_yields_six(self):
        # ~50% ALT fraction with total CN ~12 -> best ALT copy number = 6
        counts = AlleleCounts(ref=100, alt=100)
        call = _call_small_variant(
            counts,
            kiv2_copy_number=12.0,
            hgvs="LPA:4733G>A",
            ref_allele="C",
            alt_allele="T",
            positions=[1],
        )
        assert call.alt_copy_number == 6
        assert call.qual > 50.0  # Strong signal

    def test_full_alt_signal_yields_total_cn(self):
        counts = AlleleCounts(ref=0, alt=200)
        call = _call_small_variant(
            counts,
            kiv2_copy_number=12.0,
            hgvs="LPA:4733G>A",
            ref_allele="C",
            alt_allele="T",
            positions=[1],
        )
        assert call.alt_copy_number == 12
        assert call.qual == pytest.approx(MAX_PHRED_QUAL)

    def test_low_coverage_a_few_alt_reads_kept_at_alt_cn_zero(self):
        # 2 alt out of 100 informative reads sits at/below the noise floor
        # (LPA_VARIANT_NOISE_FLOOR = 0.003) -> should call REF, not flap to
        # the lowest non-zero ALT-CN.
        counts = AlleleCounts(ref=98, alt=2)
        call = _call_small_variant(
            counts,
            kiv2_copy_number=12.0,
            hgvs="LPA:4733G>A",
            ref_allele="C",
            alt_allele="T",
            positions=[1],
        )
        assert call.alt_copy_number == 0
        assert call.qual == 0.0

    def test_extreme_high_cn_alt_cn_is_proportional(self):
        # Total CN ~40 with ~25% ALT -> best alt_cn ~ 10.
        counts = AlleleCounts(ref=300, alt=100)
        call = _call_small_variant(
            counts,
            kiv2_copy_number=40.0,
            hgvs="LPA:4733G>A",
            ref_allele="C",
            alt_allele="T",
            positions=[1],
        )
        assert call.alt_copy_number == 10
        assert 0 < call.qual <= MAX_PHRED_QUAL

    def test_zero_kiv2_cn_floors_to_one(self):
        # kiv2_copy_number == 0 would otherwise blow up the binomial loop;
        # the code floors total_cn to max(1, ...) so the call still returns.
        counts = AlleleCounts(ref=10, alt=10)
        call = _call_small_variant(
            counts,
            kiv2_copy_number=0.0,
            hgvs="LPA:4733G>A",
            ref_allele="C",
            alt_allele="T",
            positions=[1],
        )
        assert call.alt_copy_number in (0, 1)


# ----------------------------------------------------------------------------
# Heterozygous marker calling
# ----------------------------------------------------------------------------


def _make_marker_call(per_marker_counts):
    """Run _call_heterozygous_markers with mocked allele counting."""
    markers = [
        _build_marker(
            {"hgvs": "LPA:1A>C", "ref": "A", "alt": "C", "first_pos": 100},
            kiv2_start=1,
            unit_len=10,
            n_repeats=len(per_marker_counts),
        )
    ] * len(per_marker_counts)

    counts_iter = iter(per_marker_counts)

    def fake_count(*_args, **_kwargs):
        return next(counts_iter)

    bam = MagicMock()
    original = lpa_caller_mod._count_alleles_at_positions
    lpa_caller_mod._count_alleles_at_positions = fake_count
    try:
        return _call_heterozygous_markers(bam, "chr6", markers, kiv2_copy_number=12.0)
    finally:
        lpa_caller_mod._count_alleles_at_positions = original


class TestHeterozygousMarkerCalling:
    def test_no_coverage(self):
        ref_cn, alt_cn, call_type, _ = _make_marker_call([AlleleCounts(ref=0, alt=0), AlleleCounts(ref=0, alt=0)])
        assert ref_cn is None
        assert alt_cn is None
        assert "No coverage" in call_type

    def test_homozygous_ref(self):
        # ALT fraction below HET_MARKER_ALT_FRACTION_MIN
        ref_cn, alt_cn, call_type, _ = _make_marker_call([AlleleCounts(ref=100, alt=1), AlleleCounts(ref=100, alt=2)])
        assert ref_cn is None
        assert alt_cn is None
        assert call_type == "Homozygous REF markers call"

    def test_homozygous_alt(self):
        # ALT fraction above HET_MARKER_ALT_FRACTION_MAX
        ref_cn, alt_cn, call_type, _ = _make_marker_call([AlleleCounts(ref=1, alt=100), AlleleCounts(ref=2, alt=100)])
        assert ref_cn is None
        assert alt_cn is None
        assert call_type == "Homozygous ALT markers call"

    def test_heterozygous_split(self):
        # ALT fraction = 0.5 in the het window -> split kiv2_cn=12 evenly.
        ref_cn, alt_cn, call_type, _ = _make_marker_call([AlleleCounts(ref=50, alt=50), AlleleCounts(ref=50, alt=50)])
        assert call_type == "Heterozygous markers call"
        assert ref_cn == pytest.approx(6.0)
        assert alt_cn == pytest.approx(6.0)

    def test_boundary_fractions(self):
        # Just below and above the boundaries fall into the homozygous calls.
        assert HET_MARKER_ALT_FRACTION_MIN < HET_MARKER_ALT_FRACTION_MAX
        # 9 / (100+9) ~= 0.0826 < 0.10
        ref_cn, alt_cn, call_type, _ = _make_marker_call([AlleleCounts(ref=100, alt=9)])
        assert call_type == "Homozygous REF markers call"
        assert ref_cn is None and alt_cn is None

    def test_only_other_bases_is_no_coverage(self):
        # All reads carry a non-REF, non-ALT base -> informative count is 0 and
        # we fall back to the "no coverage" branch instead of splitting CN.
        ref_cn, alt_cn, call_type, _ = _make_marker_call(
            [AlleleCounts(ref=0, alt=0, other=100), AlleleCounts(ref=0, alt=0, other=100)]
        )
        assert ref_cn is None
        assert alt_cn is None
        assert "No coverage" in call_type


# ----------------------------------------------------------------------------
# CLI region parser
# ----------------------------------------------------------------------------


class TestParseNormRegions:
    def test_single_region(self):
        assert _parse_norm_regions("chr1:100-200") == [("chr1", 100, 200)]

    def test_multiple_regions(self):
        out = _parse_norm_regions("chr1:100-200, chr2:300-400 ,chr3:5-6")
        assert out == [("chr1", 100, 200), ("chr2", 300, 400), ("chr3", 5, 6)]

    def test_empty_tokens_skipped(self):
        assert _parse_norm_regions(",chr1:1-2,, ,") == [("chr1", 1, 2)]

    @pytest.mark.parametrize(
        "bad",
        [
            "chr1",  # missing ':'
            "chr1:100",  # missing '-'
            "chr1:abc-200",  # non-integer start
            "chr1:100-xyz",  # non-integer end
            "chr1:200-100",  # start > end
            "chr1:0-100",  # non-positive coordinate
            "",  # nothing parsed
        ],
    )
    def test_invalid_input_raises_argument_type_error(self, bad):
        with pytest.raises(argparse.ArgumentTypeError):
            _parse_norm_regions(bad)


# ----------------------------------------------------------------------------
# VCF row formatting
# ----------------------------------------------------------------------------


def _make_lpa_call(ref_cn=6.0, alt_cn=8.0, variants=None):
    return LpaCall(
        kiv2_copy_number=ref_cn + alt_cn,
        ref_marker_allele_copy_number=ref_cn,
        alt_marker_allele_copy_number=alt_cn,
        call_type="Heterozygous markers call",
        variants=variants or [],
    )


class TestFormatDupRecord:
    def test_heterozygous_record(self):
        call = _make_lpa_call(ref_cn=6.0, alt_cn=8.0)
        row = _format_dup_record(
            "chr6",
            start=100,
            end=200,
            ref_base="A",
            sample_id="S1",
            call=call,
            n_ref_repeats=6,
        )
        fields = row.split("\t")
        assert fields[0] == "chr6"
        assert fields[1] == "100"
        assert fields[3] == "A"
        # Single symbolic <DUP> ALT (the previous <DUP>,<DUP> form was ambiguous
        # to VCF parsers); per-haplotype values are reported via INFO/CN.
        # SVTYPE=DUP keeps the record compatible with downstream merge filters.
        assert fields[4] == "<DUP>"
        assert fields[6] == "PASS"
        info = dict(kv.split("=", 1) for kv in fields[7].split(";") if "=" in kv)
        assert info["SVTYPE"] == "DUP"
        assert info["END"] == "200"
        # 1-based inclusive: end - start + 1.
        assert info["SVLEN"] == "101"
        assert info["CN"] == "1.000000,1.333333"
        assert info["TOTAL_CN"] == "14.000000"
        # FORMAT/CN now reports diploid repeat-unit copy number, identical to
        # INFO/TOTAL_CN, matching the repo CNV VCF convention.
        sample_subfields = fields[9].split(":")
        assert sample_subfields[1] == "14.000000"

    def test_missing_marker_allele_falls_back(self):
        call = LpaCall(
            kiv2_copy_number=12.0,
            ref_marker_allele_copy_number=None,
            alt_marker_allele_copy_number=None,
            call_type="Homozygous REF markers call",
            variants=[],
        )
        row = _format_dup_record("chr6", 100, 200, "A", "S1", call, n_ref_repeats=6)
        fields = row.split("\t")
        assert fields[4] == "<DUP>"
        assert "CN=." in fields[7]

    def test_deletion_when_total_cn_below_baseline(self):
        # n_ref_repeats=6 -> diploid baseline = 12. total_cn=8 < 12 -> <DEL>.
        call = _make_lpa_call(ref_cn=4.0, alt_cn=4.0)
        row = _format_dup_record("chr6", 100, 200, "A", "S1", call, n_ref_repeats=6)
        fields = row.split("\t")
        assert fields[4] == "<DEL>"
        info = dict(kv.split("=", 1) for kv in fields[7].split(";") if "=" in kv)
        assert info["SVTYPE"] == "DEL"
        assert info["TOTAL_CN"] == "8.000000"


class TestFormatSmallVariantRecord:
    def _make_call(self, alt_cn, ref_count, alt_count, qual=40.0):
        return SmallVariantCall(
            hgvs="LPA:4733G>A",
            qual=qual,
            alt_copy_number=alt_cn,
            alt_copy_number_quality=35.0,
            ref_count=ref_count,
            alt_count=alt_count,
            positions=[100, 200, 300],
            ref_allele="C",
            alt_allele="T",
        )

    def test_one_row_per_position(self):
        call = self._make_call(alt_cn=3, ref_count=90, alt_count=90)
        rows = _format_small_variant_record("chr6", call, total_cn=6, phase_set=1)
        assert len(rows) == 3
        positions = [r.split("\t")[1] for r in rows]
        assert positions == ["100", "200", "300"]

    def test_pass_filter_for_consistent_partition(self):
        # alt_cn=3 / total_cn=6 -> expected_alt = 0.5 * (90+90) = 90 (exact).
        call = self._make_call(alt_cn=3, ref_count=90, alt_count=90, qual=40.0)
        rows = _format_small_variant_record("chr6", call, total_cn=6, phase_set=1)
        assert all(r.split("\t")[6] == "PASS" for r in rows)
        # GT is 3 REF + 3 ALT alleles
        gt = rows[0].split("\t")[-1].split(":")[0]
        assert gt == "0/0/0/1/1/1"

    def test_low_qual_filter(self):
        call = self._make_call(alt_cn=3, ref_count=90, alt_count=90, qual=1.0)
        rows = _format_small_variant_record("chr6", call, total_cn=6, phase_set=1)
        for r in rows:
            assert "TargetedLowQual" in r.split("\t")[6]

    def test_repeat_conflict_filter(self):
        # alt_cn=0 but observed alt_count=20 -> residual = 20/120 ~= 0.166 > 0.05
        call = self._make_call(alt_cn=0, ref_count=100, alt_count=20, qual=40.0)
        rows = _format_small_variant_record("chr6", call, total_cn=6, phase_set=1)
        for r in rows:
            assert "TargetedRepeatConflict" in r.split("\t")[6]


# ----------------------------------------------------------------------------
# JSON / VCF writers (end-to-end on a minimal call object)
# ----------------------------------------------------------------------------


def _build_minimal_call():
    return LpaCall(
        kiv2_copy_number=12.0,
        ref_marker_allele_copy_number=6.0,
        alt_marker_allele_copy_number=6.0,
        call_type="Heterozygous markers call",
        variants=[
            SmallVariantCall(
                hgvs="LPA:4733G>A",
                qual=42.5,
                alt_copy_number=2,
                alt_copy_number_quality=30.0,
                ref_count=80,
                alt_count=20,
                positions=[1000, 2000],
                ref_allele="C",
                alt_allele="T",
            )
        ],
    )


class TestWriteJson:
    def test_payload_round_trip(self, tmp_path):
        out = tmp_path / "x.targeted.json"
        _write_json(out, "S1", _build_minimal_call(), "hg38")
        payload = json.loads(out.read_text())
        assert payload["sampleId"] == "S1"
        assert payload["genomeBuild"] == "hg38"
        lpa = payload["lpa"]
        assert lpa["kiv2CopyNumber"] == 12.0
        assert lpa["refMarkerAlleleCopyNumber"] == 6.0
        assert lpa["altMarkerAlleleCopyNumber"] == 6.0
        assert lpa["type"] == "Heterozygous markers call"
        assert len(lpa["variants"]) == 1
        v = lpa["variants"][0]
        assert v["hgvs"] == "LPA:4733G>A"
        assert v["altCopyNumber"] == 2


class TestWriteVcf:
    def test_writes_header_and_rows(self, tmp_path):
        fasta = _FakeFasta({"chr6": "A" * 1000})
        out = tmp_path / "x.targeted.vcf.gz"
        _write_vcf(
            out,
            sample_id="S1",
            call=_build_minimal_call(),
            chrom="chr6",
            kiv2_start=100,
            kiv2_end=200,
            n_ref_repeats=6,
            fasta=fasta,
            contigs=[("chr6", 170_000_000)],
        )
        with gzip.open(out, "rt") as f:
            text = f.read()
        assert text.startswith("##fileformat=VCFv4.2")
        assert "##contig=<ID=chr6,length=170000000>" in text
        # Sample column header
        assert text.split("\n")[-2 - 2].endswith("S1") or "\tS1" in text
        # One SV row + one row per variant position
        body = [line for line in text.splitlines() if line and not line.startswith("#")]
        assert len(body) == 1 + 2  # 1 DUP row, 2 small-variant rows


# ----------------------------------------------------------------------------
# Sanity check on module defaults (catch accidental edits)
# ----------------------------------------------------------------------------


class TestModuleDefaults:
    def test_default_repeat_count_matches_unit_len(self):
        # 6 repeats x ~5552 bp ~ 33312 bp; KIV-2 array span is 33507 bp
        approx_len = DEFAULT_KIV2_REPEAT_COUNT * DEFAULT_KIV2_UNIT_LEN
        assert 33_000 < approx_len < 34_000

    def test_default_kiv2_start_is_positive(self):
        assert DEFAULT_KIV2_START > 0
