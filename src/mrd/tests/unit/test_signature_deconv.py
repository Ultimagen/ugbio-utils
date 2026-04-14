"""Tests for signature_deconv module."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ugbio_mrd.signature_deconv import signature_deconv
from ugbio_mrd.split_by_vaf import TRINUC_ORDER, VAF_BINS


@pytest.fixture
def tmp_output(tmp_path):
    return tmp_path / "output"


@pytest.fixture
def trinuc_counts_csv(tmp_path):
    """Create a synthetic trinuc counts CSV matching the split_by_vaf output format."""
    bin_labels = [label for _, _, label in VAF_BINS]
    rng = np.random.default_rng(42)
    data = {"trinuc_substitution": TRINUC_ORDER}
    for label in bin_labels:
        data[label] = rng.integers(0, 100, size=len(TRINUC_ORDER))
    counts_df = pd.DataFrame(data)
    csv_path = tmp_path / "test.trinuc_counts.csv"
    counts_df.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def cosmic_signatures_file(tmp_path):
    """Create a minimal COSMIC-like signatures TSV file with correct 96 trinuc types."""
    data = {"Type": TRINUC_ORDER}
    rng = np.random.default_rng(0)
    for sig_name in ["SBS1", "SBS2", "SBS5", "SBS13", "SBS40"]:
        vals = rng.random(len(TRINUC_ORDER))
        data[sig_name] = vals / vals.sum()  # normalize to probability
    cosmic_df = pd.DataFrame(data)
    tsv_path = tmp_path / "COSMIC_v3.4_SBS_GRCh38.txt"
    cosmic_df.to_csv(tsv_path, sep="\t", index=False)
    return str(tsv_path)


class TestSignatureDeconv:
    def test_basic_run(self, trinuc_counts_csv, cosmic_signatures_file, tmp_output):
        """Test full deconvolution pipeline produces expected outputs."""
        weights_csv, plot_png = signature_deconv(
            trinuc_counts_csv=trinuc_counts_csv,
            cosmic_signatures_file=cosmic_signatures_file,
            signatures_to_include=["SBS1", "SBS5", "SBS40"],
            output_dir=str(tmp_output),
            basename="test_sample",
        )

        assert Path(weights_csv).exists()
        assert Path(plot_png).exists()

        # Verify weights CSV structure
        df_weights = pd.read_csv(weights_csv, index_col=0)
        assert set(df_weights.columns).issubset({"SBS1", "SBS5", "SBS40"} | {label for _, _, label in VAF_BINS})
        assert len(df_weights) > 0

    def test_output_filenames(self, trinuc_counts_csv, cosmic_signatures_file, tmp_output):
        """Output filenames follow expected pattern."""
        weights_csv, plot_png = signature_deconv(
            trinuc_counts_csv=trinuc_counts_csv,
            cosmic_signatures_file=cosmic_signatures_file,
            signatures_to_include=["SBS1", "SBS5"],
            output_dir=str(tmp_output),
            basename="my_run",
        )
        assert weights_csv.endswith("my_run.signature_weights.csv")
        assert plot_png.endswith("my_run.signature_deconv.png")

    def test_missing_signatures_warns(self, trinuc_counts_csv, cosmic_signatures_file, tmp_output):
        """Missing signatures should log warning but still run with available ones."""
        weights_csv, plot_png = signature_deconv(
            trinuc_counts_csv=trinuc_counts_csv,
            cosmic_signatures_file=cosmic_signatures_file,
            signatures_to_include=["SBS1", "NONEXISTENT_SIG"],
            output_dir=str(tmp_output),
            basename="test_missing",
        )
        # Should still produce output with the available signature
        assert Path(weights_csv).exists()
        assert Path(plot_png).exists()

    def test_all_signatures_missing_raises(self, trinuc_counts_csv, cosmic_signatures_file, tmp_output):
        """If none of the requested signatures exist, should raise ValueError."""
        with pytest.raises(ValueError, match="None of the requested signatures found"):
            signature_deconv(
                trinuc_counts_csv=trinuc_counts_csv,
                cosmic_signatures_file=cosmic_signatures_file,
                signatures_to_include=["FAKE1", "FAKE2"],
                output_dir=str(tmp_output),
                basename="test_none",
            )

    def test_single_vaf_bin(self, tmp_path, cosmic_signatures_file, tmp_output):
        """Works with a single VAF bin column."""
        data = {"trinuc_substitution": TRINUC_ORDER}
        rng = np.random.default_rng(42)
        data["0-0.5%"] = rng.integers(0, 50, size=len(TRINUC_ORDER))
        single_bin_df = pd.DataFrame(data)
        csv_path = tmp_path / "single_bin.csv"
        single_bin_df.to_csv(csv_path, index=False)

        weights_csv, plot_png = signature_deconv(
            trinuc_counts_csv=str(csv_path),
            cosmic_signatures_file=cosmic_signatures_file,
            signatures_to_include=["SBS1", "SBS5"],
            output_dir=str(tmp_output),
            basename="single_bin",
        )
        assert Path(weights_csv).exists()
        assert Path(plot_png).exists()

    def test_weights_are_non_negative(self, trinuc_counts_csv, cosmic_signatures_file, tmp_output):
        """Signature weights should be non-negative."""
        weights_csv, _ = signature_deconv(
            trinuc_counts_csv=trinuc_counts_csv,
            cosmic_signatures_file=cosmic_signatures_file,
            signatures_to_include=["SBS1", "SBS2", "SBS5"],
            output_dir=str(tmp_output),
            basename="test_nonneg",
        )
        df_weights = pd.read_csv(weights_csv, index_col=0)
        assert (df_weights.to_numpy() >= 0).all()
