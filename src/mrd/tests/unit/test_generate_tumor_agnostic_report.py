"""Tests for generate_tumor_agnostic_report module."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from ugbio_mrd.generate_tumor_agnostic_report import (
    HTML_REPORT_SUFFIX,
    TEMPLATE_NOTEBOOK,
    generate_tumor_agnostic_report,
)
from ugbio_mrd.split_by_vaf import FIRST_BIN_SINGLE_READ_LABEL, TRINUC_ORDER, get_vaf_bin_labels


@pytest.fixture
def sample_inputs(tmp_path):
    """Create minimal sample input files for the report."""
    # Create trinuc histogram PNG (1x1 pixel stub)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.bar([0], [1])
    histogram_png = str(tmp_path / "test.trinuc_histogram.png")
    fig.savefig(histogram_png)
    plt.close(fig)

    # Create trinuc counts CSV
    bin_labels = get_vaf_bin_labels()
    rng = np.random.default_rng(42)
    data = {"trinuc_substitution": TRINUC_ORDER}
    for label in bin_labels:
        data[label] = rng.integers(0, 50, size=len(TRINUC_ORDER))
    counts_csv = str(tmp_path / "test.trinuc_counts.csv")
    pd.DataFrame(data).to_csv(counts_csv, index=False)

    # Create signature plot PNG
    fig2, ax2 = plt.subplots()
    ax2.bar([0, 1], [0.5, 0.3])
    sig_plot_png = str(tmp_path / "test.signature_deconv.png")
    fig2.savefig(sig_plot_png)
    plt.close(fig2)

    # Create signature weights CSV
    weights_csv = str(tmp_path / "test.signature_weights.csv")
    pd.DataFrame(
        {"SBS1": [0.3, 0.4], "SBS5": [0.7, 0.6]},
        index=[FIRST_BIN_SINGLE_READ_LABEL, "0.5-5%"],
    ).to_csv(weights_csv)

    return {
        "trinuc_histogram_png": histogram_png,
        "trinuc_counts_csv": counts_csv,
        "signature_plot_png": sig_plot_png,
        "signature_weights_csv": weights_csv,
    }


class TestGenerateTumorAgnosticReport:
    def test_template_notebook_exists(self):
        """The report template notebook should exist in the package."""
        assert TEMPLATE_NOTEBOOK.exists(), f"Template notebook not found: {TEMPLATE_NOTEBOOK}"

    @patch("ugbio_mrd.generate_tumor_agnostic_report.generate_report")
    def test_calls_generate_report_with_correct_params(self, mock_generate_report, sample_inputs, tmp_path):
        """Verify that generate_report is called with the correct parameters."""
        output_dir = str(tmp_path / "report_output")
        basename = "test_run"

        # Mock generate_report to just create the expected output file
        expected_html = Path(output_dir) / (basename + HTML_REPORT_SUFFIX)

        def side_effect(template_notebook_path, parameters, output_report_html_path):
            output_report_html_path.parent.mkdir(parents=True, exist_ok=True)
            output_report_html_path.write_text("<html>mock</html>")

        mock_generate_report.side_effect = side_effect

        result = generate_tumor_agnostic_report(
            trinuc_histogram_png=sample_inputs["trinuc_histogram_png"],
            trinuc_counts_csv=sample_inputs["trinuc_counts_csv"],
            signature_plot_png=sample_inputs["signature_plot_png"],
            signature_weights_csv=sample_inputs["signature_weights_csv"],
            output_dir=output_dir,
            basename=basename,
        )

        # Verify generate_report was called
        mock_generate_report.assert_called_once()
        call_kwargs = mock_generate_report.call_args
        assert call_kwargs.kwargs["template_notebook_path"] == TEMPLATE_NOTEBOOK
        assert call_kwargs.kwargs["parameters"]["trinuc_histogram_png"] == sample_inputs["trinuc_histogram_png"]
        assert call_kwargs.kwargs["parameters"]["trinuc_counts_csv"] == sample_inputs["trinuc_counts_csv"]
        assert call_kwargs.kwargs["parameters"]["signature_plot_png"] == sample_inputs["signature_plot_png"]
        assert call_kwargs.kwargs["parameters"]["signature_weights_csv"] == sample_inputs["signature_weights_csv"]
        assert call_kwargs.kwargs["parameters"]["basename"] == basename

        # Verify return value
        assert result == expected_html

    @patch("ugbio_mrd.generate_tumor_agnostic_report.generate_report")
    def test_output_path_uses_correct_suffix(self, mock_generate_report, sample_inputs, tmp_path):
        """Output HTML file should use the expected suffix."""
        output_dir = str(tmp_path / "out")
        basename = "sample_123"

        def side_effect(template_notebook_path, parameters, output_report_html_path):
            output_report_html_path.parent.mkdir(parents=True, exist_ok=True)
            output_report_html_path.write_text("<html></html>")

        mock_generate_report.side_effect = side_effect

        result = generate_tumor_agnostic_report(
            trinuc_histogram_png=sample_inputs["trinuc_histogram_png"],
            trinuc_counts_csv=sample_inputs["trinuc_counts_csv"],
            signature_plot_png=sample_inputs["signature_plot_png"],
            signature_weights_csv=sample_inputs["signature_weights_csv"],
            output_dir=output_dir,
            basename=basename,
        )
        assert result.name == f"sample_123{HTML_REPORT_SUFFIX}"

    @patch("ugbio_mrd.generate_tumor_agnostic_report.generate_report")
    def test_creates_output_directory(self, mock_generate_report, sample_inputs, tmp_path):
        """Output directory should be created if it doesn't exist."""
        output_dir = str(tmp_path / "new" / "nested" / "dir")

        def side_effect(template_notebook_path, parameters, output_report_html_path):
            output_report_html_path.parent.mkdir(parents=True, exist_ok=True)
            output_report_html_path.write_text("<html></html>")

        mock_generate_report.side_effect = side_effect

        generate_tumor_agnostic_report(
            trinuc_histogram_png=sample_inputs["trinuc_histogram_png"],
            trinuc_counts_csv=sample_inputs["trinuc_counts_csv"],
            signature_plot_png=sample_inputs["signature_plot_png"],
            signature_weights_csv=sample_inputs["signature_weights_csv"],
            output_dir=output_dir,
            basename="test",
        )
        assert Path(output_dir).exists()
