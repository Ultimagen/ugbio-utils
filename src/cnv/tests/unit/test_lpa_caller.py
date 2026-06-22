"""Unit tests for the LPA KIV-2 targeted caller.

Covers the pure-Python helpers (marker projection, GC fraction/bucketing,
trimmed mean, binomial/Phred math, marker and small-variant calling,
CLI region parsing, and VCF row formatting). I/O paths that need a real BAM
are exercised via mocked ``pysam`` objects.
"""

from __future__ import annotations

import gzip
import json
from unittest.mock import MagicMock

import numpy as np
import pytest
from ugbio_cnv.lpa_caller import (
    DEFAULT_KIV2_REPEAT_COUNT,
    DEFAULT_KIV2_START,
    DEFAULT_KIV2_UNIT_LEN,
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
    _format_dup_record,
    _format_small_variant_record,
    _gc_fraction,
    _log_binomial,
    _parse_norm_regions,
    _phred,
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
# Binomial / Phred math
# ----------------------------------------------------------------------------


class TestBinomialAndPhred:
    def test_log_binomial_zero_n_is_zero(self):
        assert _log_binomial(0, 0, 0.5) == 0.0

    def test_log_binomial_known_value(self):
        # log(C(4,2) * 0.5^4) = log(6/16)
        import math

        assert _log_binomial(4, 2, 0.5) == pytest.approx(math.log(6 / 16))

    def test_log_binomial_clamps_extremes(self):
        # p=0 would log(0) without clamping; should be finite
        val = _log_binomial(10, 0, 0.0)
        assert np.isfinite(val)
        val = _log_binomial(10, 10, 1.0)
        assert np.isfinite(val)

    def test_phred_caps_at_max(self):
        assert _phred(1e-50) == pytest.approx(MAX_PHRED_QUAL)

    def test_phred_near_one_is_small(self):
        assert _phred(0.999999) < 1.0

    def test_phred_half_is_three(self):
        assert _phred(0.5) == pytest.approx(3.0103, abs=1e-3)


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


# ----------------------------------------------------------------------------
# Heterozygous marker calling
# ----------------------------------------------------------------------------


def _make_marker_call(per_marker_counts):
    """Run _call_heterozygous_markers with mocked allele counting."""
    from ugbio_cnv import lpa_caller

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
    original = lpa_caller._count_alleles_at_positions
    lpa_caller._count_alleles_at_positions = fake_count
    try:
        return _call_heterozygous_markers(bam, "chr6", markers, kiv2_copy_number=12.0)
    finally:
        lpa_caller._count_alleles_at_positions = original


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
