"""Tests for split_by_vaf module."""

from pathlib import Path

import pandas as pd
import polars as pl
import pytest
from ugbio_mrd.split_by_vaf import (
    FIRST_BIN_MULTI_READ_LABEL,
    FIRST_BIN_SINGLE_READ_LABEL,
    SUBSTITUTION_ORDER,
    TRINUC_ORDER,
    VAF_BINS,
    _assign_vaf_bin,
    _canonical_trinuc_change,
    _revcomp,
    get_vaf_bin_labels,
    split_by_vaf,
)


@pytest.fixture
def resources_dir():
    return Path(__file__).parent.parent / "resources"


@pytest.fixture
def featuremap_parquet(resources_dir):
    return str(resources_dir / "416119_L7402.featuremap_df.10k.parquet")


class TestRevcomp:
    def test_basic(self):
        assert _revcomp("ACGT") == "ACGT"

    def test_single_base(self):
        assert _revcomp("A") == "T"
        assert _revcomp("C") == "G"
        assert _revcomp("G") == "C"
        assert _revcomp("T") == "A"

    def test_palindrome(self):
        assert _revcomp("AT") == "AT"

    def test_complex(self):
        assert _revcomp("AACGT") == "ACGTT"


class TestCanonicalTrinucChange:
    def test_pyrimidine_context_stays(self):
        """C>A with context ACG should stay as A[C>A]G."""
        result = _canonical_trinuc_change("A", "C", "G", "A")
        assert result == "A[C>A]G"

    def test_purine_context_flipped(self):
        """G>T is flipped to C>A context."""
        result = _canonical_trinuc_change("A", "G", "C", "T")
        # revcomp of AGC = GCT, alt revcomp of T = A -> G[C>A]T
        assert result == "G[C>A]T"

    def test_t_ref_stays(self):
        """T>A stays in pyrimidine context."""
        result = _canonical_trinuc_change("G", "T", "C", "A")
        assert result == "G[T>A]C"

    def test_a_ref_flipped(self):
        """A>G is flipped: revcomp of context, revcomp of alt."""
        result = _canonical_trinuc_change("G", "A", "C", "G")
        # revcomp of GAC = GTC, alt revcomp of G = C -> G[T>C]C
        assert result == "G[T>C]C"

    def test_ref_equals_alt_returns_none(self):
        assert _canonical_trinuc_change("A", "C", "G", "C") is None

    def test_invalid_base_returns_none(self):
        assert _canonical_trinuc_change("N", "C", "G", "A") is None

    def test_lowercase_handled(self):
        result = _canonical_trinuc_change("a", "c", "g", "a")
        assert result == "A[C>A]G"

    def test_all_substitution_types_present(self):
        """Verify that all 6 canonical substitution types can be produced."""
        cases = [
            ("A", "C", "G", "A"),  # C>A
            ("A", "C", "G", "G"),  # C>G
            ("A", "C", "G", "T"),  # C>T
            ("A", "T", "G", "A"),  # T>A
            ("A", "T", "G", "C"),  # T>C
            ("A", "T", "G", "G"),  # T>G
        ]
        for left, ref, right, alt in cases:
            result = _canonical_trinuc_change(left, ref, right, alt)
            assert result is not None
            sub = result[2:5]
            assert sub in SUBSTITUTION_ORDER


class TestAssignVafBin:
    def test_zero_vaf(self):
        assert _assign_vaf_bin(0.0) == "0-0.5%"

    def test_zero_vaf_single_read(self):
        assert _assign_vaf_bin(0.0, supporting_read_count=1) == FIRST_BIN_SINGLE_READ_LABEL

    def test_zero_vaf_multi_read(self):
        assert _assign_vaf_bin(0.0, supporting_read_count=2) == FIRST_BIN_MULTI_READ_LABEL

    def test_mid_vaf(self):
        assert _assign_vaf_bin(0.25) == "10-30%"

    def test_high_vaf(self):
        assert _assign_vaf_bin(0.75) == "50-100%"

    def test_boundary_lower(self):
        assert _assign_vaf_bin(0.005) == "0.5-5%"

    def test_boundary_exact(self):
        assert _assign_vaf_bin(0.05) == "5-10%"

    def test_one_is_included(self):
        assert _assign_vaf_bin(1.0) == "50-100%"

    def test_negative_returns_none(self):
        assert _assign_vaf_bin(-0.1) is None


class TestSplitByVaf:
    def test_basic_run(self, featuremap_parquet, tmp_path):
        """Test split_by_vaf on real parquet data."""
        counts_csv, histogram_png = split_by_vaf(
            input_parquet=featuremap_parquet,
            output_dir=str(tmp_path),
            basename="test_sample",
            snvq_threshold=30,
            mapq_threshold=30,
        )

        # Output files exist
        assert Path(counts_csv).exists()
        assert Path(histogram_png).exists()

        # Verify CSV structure
        counts_df = pd.read_csv(counts_csv)
        assert "trinuc_substitution" in counts_df.columns
        bin_labels = get_vaf_bin_labels()
        for label in bin_labels:
            assert label in counts_df.columns

        # All trinuc entries should be in TRINUC_ORDER
        assert set(counts_df["trinuc_substitution"].tolist()).issubset(set(TRINUC_ORDER))
        assert len(counts_df) == len(TRINUC_ORDER)

        # All values should be non-negative integers
        for label in bin_labels:
            assert (counts_df[label] >= 0).all()
            assert counts_df[label].dtype in ("int64", "int32")

    def test_strict_thresholds_yield_fewer_rows(self, featuremap_parquet, tmp_path):
        """Higher quality thresholds should yield fewer or equal counts."""
        out1 = tmp_path / "loose"
        out2 = tmp_path / "strict"

        counts_csv_loose, _ = split_by_vaf(
            input_parquet=featuremap_parquet,
            output_dir=str(out1),
            basename="loose",
            snvq_threshold=30,
            mapq_threshold=30,
        )
        counts_csv_strict, _ = split_by_vaf(
            input_parquet=featuremap_parquet,
            output_dir=str(out2),
            basename="strict",
            snvq_threshold=50,
            mapq_threshold=50,
        )

        df_loose = pd.read_csv(counts_csv_loose)
        df_strict = pd.read_csv(counts_csv_strict)
        bin_labels = get_vaf_bin_labels()

        total_loose = df_loose[bin_labels].sum().sum()
        total_strict = df_strict[bin_labels].sum().sum()
        assert total_strict <= total_loose

    def test_first_vaf_bin_is_split_by_supporting_read_count(self, tmp_path):
        """First VAF bin should separate single-read and multi-read variants."""
        variant_rows = [
            {
                "CHROM": "chr1",
                "POS": 100,
                "REF": "C",
                "ALT": "A",
                "X_PREV1": "A",
                "X_NEXT1": "G",
                "VAF": 0.001,
                "SNVQ": 80.0,
                "FILT": 1,
                "MAPQ": 70,
            },
            {
                "CHROM": "chr1",
                "POS": 200,
                "REF": "C",
                "ALT": "T",
                "X_PREV1": "T",
                "X_NEXT1": "A",
                "VAF": 0.003,
                "SNVQ": 80.0,
                "FILT": 1,
                "MAPQ": 70,
            },
            {
                "CHROM": "chr1",
                "POS": 200,
                "REF": "C",
                "ALT": "T",
                "X_PREV1": "T",
                "X_NEXT1": "A",
                "VAF": 0.003,
                "SNVQ": 80.0,
                "FILT": 1,
                "MAPQ": 70,
            },
            {
                "CHROM": "chr1",
                "POS": 300,
                "REF": "T",
                "ALT": "G",
                "X_PREV1": "G",
                "X_NEXT1": "C",
                "VAF": 0.02,
                "SNVQ": 80.0,
                "FILT": 1,
                "MAPQ": 70,
            },
        ]
        parquet_path = tmp_path / "split_first_bin.parquet"
        pl.DataFrame(variant_rows).write_parquet(parquet_path)

        counts_csv, _ = split_by_vaf(
            input_parquet=str(parquet_path),
            output_dir=str(tmp_path),
            basename="split_case",
        )

        counts_df = pd.read_csv(counts_csv).set_index("trinuc_substitution")
        assert counts_df[FIRST_BIN_SINGLE_READ_LABEL].sum() == 1
        assert counts_df[FIRST_BIN_MULTI_READ_LABEL].sum() == 2
        assert counts_df["0.5-5%"].sum() == 1
        assert counts_df.loc["A[C>A]G", FIRST_BIN_SINGLE_READ_LABEL] == 1
        assert counts_df.loc["T[C>T]A", FIRST_BIN_MULTI_READ_LABEL] == 2
        assert counts_df.loc["G[T>G]C", "0.5-5%"] == 1

    def test_missing_columns_raises(self, tmp_path):
        """Parquet missing required columns should raise ValueError."""
        bad_df = pl.DataFrame({"A": [1, 2], "B": [3, 4]})
        bad_path = str(tmp_path / "bad.parquet")
        bad_df.write_parquet(bad_path)

        with pytest.raises(ValueError, match="Missing required columns"):
            split_by_vaf(
                input_parquet=bad_path,
                output_dir=str(tmp_path),
                basename="bad",
            )

    def test_output_filenames(self, featuremap_parquet, tmp_path):
        """Output file names should match the basename pattern."""
        counts_csv, histogram_png = split_by_vaf(
            input_parquet=featuremap_parquet,
            output_dir=str(tmp_path),
            basename="my_sample",
            snvq_threshold=30,
            mapq_threshold=30,
        )
        assert counts_csv.endswith("my_sample.trinuc_counts.csv")
        assert histogram_png.endswith("my_sample.trinuc_histogram.png")


class TestConstants:
    def test_vaf_bins_cover_full_range(self):
        """VAF bins should cover [0, 1]."""
        assert VAF_BINS[0][0] == 0.0
        assert VAF_BINS[-1][1] > 1.0

    def test_trinuc_order_length(self):
        """Should be 96 trinucleotide substitutions."""
        assert len(TRINUC_ORDER) == 96

    def test_substitution_order_length(self):
        assert len(SUBSTITUTION_ORDER) == 6
