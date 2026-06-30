import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ugbio_mrd.generate_mrd_report import MrdReportInputs, generate_mrd_report
from ugbio_mrd.mrd_report_renderer import render_read_length_histogram


@pytest.fixture
def resources_dir():
    return Path(__file__).parent.parent / "resources" / "report"


@pytest.fixture
def output_path(tmpdir):
    return Path(tmpdir)


@pytest.fixture
def mrd_report_inputs(output_path, resources_dir):
    return MrdReportInputs(
        intersected_featuremaps_parquet=[
            str(resources_dir / "Pa_46_333_LuNgs_08.Pa_46_FreshFrozen.matched.intersection.parquet"),
            str(resources_dir / "Pa_46_333_LuNgs_08.Pa_67_FFPE.control.intersection.parquet"),
            str(resources_dir / "Pa_46_333_LuNgs_08.syn0_Pa_46_FreshFrozen.db_control.intersection.parquet"),
            str(resources_dir / "Pa_46_333_LuNgs_08.syn1_Pa_46_FreshFrozen.db_control.intersection.parquet"),
            str(resources_dir / "Pa_46_333_LuNgs_08.syn2_Pa_46_FreshFrozen.db_control.intersection.parquet"),
            str(resources_dir / "Pa_46_333_LuNgs_08.syn3_Pa_46_FreshFrozen.db_control.intersection.parquet"),
            str(resources_dir / "Pa_46_333_LuNgs_08.syn4_Pa_46_FreshFrozen.db_control.intersection.parquet"),
        ],
        matched_signatures_vcf_files=[str(resources_dir / "Pa_46_FreshFrozen.ann.chr20.filtered.vcf.gz")],
        control_signatures_vcf_files=[str(resources_dir / "Pa_67_FFPE.ann.chr20.filtered.vcf.gz")],
        db_control_signatures_vcf_files=[
            str(resources_dir / "syn0_Pa_46_FreshFrozen.ann.chr20.filtered_pancan_pcawg_2020.chr20.filtered.vcf.gz"),
            str(resources_dir / "syn1_Pa_46_FreshFrozen.ann.chr20.filtered_pancan_pcawg_2020.chr20.filtered.vcf.gz"),
            str(resources_dir / "syn2_Pa_46_FreshFrozen.ann.chr20.filtered_pancan_pcawg_2020.chr20.filtered.vcf.gz"),
            str(resources_dir / "syn3_Pa_46_FreshFrozen.ann.chr20.filtered_pancan_pcawg_2020.chr20.filtered.vcf.gz"),
            str(resources_dir / "syn4_Pa_46_FreshFrozen.ann.chr20.filtered_pancan_pcawg_2020.chr20.filtered.vcf.gz"),
        ],
        coverage_bed=str(resources_dir / "Pa_46_333_LuNgs_08.regions.bed.gz"),
        output_dir=output_path,
        output_basename="test_report",
        featuremap_file=str(resources_dir / "Pa_46_333_LuNgs_08.featuremap_df.10k.parquet"),
        signature_filter_query="(norm_coverage <= 2.5) and (norm_coverage >= 0.6)",
        read_filter_query="filt>0 and snvq>60 and mapq>=60",
        srsnv_metadata_json=str(resources_dir / "Pa_46_333_LuNgs_08.srsnv_metadata.json"),
    )


def test_generate_mrd_report(output_path, resources_dir, mrd_report_inputs):
    results_html, qc_html = generate_mrd_report(mrd_report_inputs)

    # assert both report HTMLs exist and are non-empty
    assert results_html.exists()
    assert results_html.stat().st_size > 0
    assert qc_html.exists()
    assert qc_html.stat().st_size > 0

    # test h5 output — results report writes primary keys, QC report appends secondary keys
    h5_output = str(output_path / "test_report.ctdna_vaf.h5")
    h5_expected = str(resources_dir / "test_report.ctdna_vaf.expected_output.h5")
    with pd.HDFStore(h5_expected) as store:
        h5_keys = store.keys()
    with pd.HDFStore(h5_output) as store:
        h5_keys_output = store.keys()

    # assert that the keys in the output h5 file are the same as the keys in the expected
    assert h5_keys == h5_keys_output

    # assert h5 output values are as expected
    for key in h5_keys:
        pd.testing.assert_frame_equal(pd.read_hdf(h5_output, key), pd.read_hdf(h5_expected, key))


def test_generate_mrd_report_detection_output(output_path, mrd_report_inputs):
    """Test that results report generates detection result JSON with expected fields."""
    generate_mrd_report(mrd_report_inputs)

    # Verify detection JSON was created
    detection_json_path = output_path / "test_report.detection_result.json"
    assert detection_json_path.exists(), "Detection result JSON not generated"

    # Load and validate detection results
    with open(detection_json_path) as f:
        detection = json.load(f)

    # Verify all expected fields are present
    expected_fields = [
        "call",
        "detected",
        "p_value",
        "matched_supporting_reads",
        "matched_ctdna_vaf",
        "null_median_reads",
        "null_max_reads",
        "n_synthetic_controls",
        "personal_lod",
        "signature_size",
        "mean_coverage",
        "corrected_coverage",
        "alpha",
    ]
    for field in expected_fields:
        assert field in detection, f"Missing field: {field}"

    # Verify detection call is one of valid values
    assert detection["call"] in ("MRD Detected", "MRD Not Detected", "Indeterminate")

    # Verify p-value is in valid range [0, 1]
    assert 0 <= detection["p_value"] <= 1.0

    # Verify we used 5 synthetic controls from the test data
    assert detection["n_synthetic_controls"] == 5

    # Verify supporting reads is non-negative integer
    assert detection["matched_supporting_reads"] >= 0
    assert isinstance(detection["matched_supporting_reads"], int)

    # Verify signature size > 0 (we have real data)
    assert detection["signature_size"] > 0

    # Verify mean coverage is reasonable
    assert detection["mean_coverage"] > 0


def test_generate_mrd_report_html_contains_detection_banner(output_path, mrd_report_inputs):
    """Test that the results HTML report contains the detection result banner."""
    results_html, _qc_html = generate_mrd_report(mrd_report_inputs)

    html_content = results_html.read_text()

    # Report should contain the detection call
    assert any(
        call in html_content for call in ("MRD Detected", "MRD Not Detected", "Indeterminate")
    ), "Detection call not found in HTML report"

    # Report should contain key metrics
    assert "p-value" in html_content.lower() or "p_value" in html_content.lower()
    assert "Personal LOD" in html_content or "personal_lod" in html_content.lower()
    assert "Supporting Reads" in html_content

    # Report should contain the assay metrics section
    assert "Signature Size" in html_content
    assert "Mean Coverage" in html_content


class TestRenderReadLengthHistogram:
    """Tests for render_read_length_histogram column-name normalisation."""

    def _make_df(self, col_name: str) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        n = 50
        return pd.DataFrame(
            {
                col_name: rng.integers(100, 300, size=n),
                "signature_type": (["matched"] * (n // 2)) + (["control"] * (n // 2)),
            }
        )

    def test_lowercase_x_length_column(self):
        """x_length (lowercased by read_and_filter_features_parquet) must produce a histogram."""
        df_features = self._make_df("x_length")
        result = render_read_length_histogram(df_features)
        assert result != "", "Expected non-empty base64 image for x_length column"

    def test_uppercase_x_length_column(self):
        """X_LENGTH (original casing) must also produce a histogram."""
        df_features = self._make_df("X_LENGTH")
        result = render_read_length_histogram(df_features)
        assert result != "", "Expected non-empty base64 image for X_LENGTH column"

    def test_missing_length_column_returns_empty(self):
        """DataFrame without any length column must return empty string."""
        df_features = pd.DataFrame({"signature_type": ["matched", "control"]})
        result = render_read_length_histogram(df_features)
        assert result == ""


def test_generate_mrd_report_with_noise_filter(output_path, resources_dir):
    """
    Verify generate_mrd_report works end-to-end with the noise locus filter enabled,
    and that the no-noise secondary H5 key is present in the output.
    """
    import os

    out_dir = str(output_path / "noise_test")
    os.makedirs(out_dir, exist_ok=True)
    noise_inputs = MrdReportInputs(
        intersected_featuremaps_parquet=[
            str(resources_dir / "Pa_46_333_LuNgs_08.Pa_46_FreshFrozen.matched.intersection.parquet"),
            str(resources_dir / "Pa_46_333_LuNgs_08.Pa_67_FFPE.control.intersection.parquet"),
            str(resources_dir / "Pa_46_333_LuNgs_08.syn0_Pa_46_FreshFrozen.db_control.intersection.parquet"),
            str(resources_dir / "Pa_46_333_LuNgs_08.syn1_Pa_46_FreshFrozen.db_control.intersection.parquet"),
            str(resources_dir / "Pa_46_333_LuNgs_08.syn2_Pa_46_FreshFrozen.db_control.intersection.parquet"),
            str(resources_dir / "Pa_46_333_LuNgs_08.syn3_Pa_46_FreshFrozen.db_control.intersection.parquet"),
            str(resources_dir / "Pa_46_333_LuNgs_08.syn4_Pa_46_FreshFrozen.db_control.intersection.parquet"),
        ],
        matched_signatures_vcf_files=[str(resources_dir / "Pa_46_FreshFrozen.ann.chr20.filtered.vcf.gz")],
        control_signatures_vcf_files=[str(resources_dir / "Pa_67_FFPE.ann.chr20.filtered.vcf.gz")],
        db_control_signatures_vcf_files=[
            str(resources_dir / "syn0_Pa_46_FreshFrozen.ann.chr20.filtered_pancan_pcawg_2020.chr20.filtered.vcf.gz"),
            str(resources_dir / "syn1_Pa_46_FreshFrozen.ann.chr20.filtered_pancan_pcawg_2020.chr20.filtered.vcf.gz"),
            str(resources_dir / "syn2_Pa_46_FreshFrozen.ann.chr20.filtered_pancan_pcawg_2020.chr20.filtered.vcf.gz"),
            str(resources_dir / "syn3_Pa_46_FreshFrozen.ann.chr20.filtered_pancan_pcawg_2020.chr20.filtered.vcf.gz"),
            str(resources_dir / "syn4_Pa_46_FreshFrozen.ann.chr20.filtered_pancan_pcawg_2020.chr20.filtered.vcf.gz"),
        ],
        coverage_bed=str(resources_dir / "Pa_46_333_LuNgs_08.regions.bed.gz"),
        output_dir=out_dir,
        output_basename="noise_test",
        featuremap_file=str(resources_dir / "Pa_46_333_LuNgs_08.featuremap_df.10k.parquet"),
        signature_filter_query="(norm_coverage <= 2.5) and (norm_coverage >= 0.6)",
        read_filter_query="filt>0 and snvq>60 and mapq>=60",
        srsnv_metadata_json=str(resources_dir / "Pa_46_333_LuNgs_08.srsnv_metadata.json"),
        thresh_noise_lq_reads=1,
        thresh_noise_hq_exemption=3,
    )

    results_html, qc_html = generate_mrd_report(noise_inputs)

    assert results_html.exists() and results_html.stat().st_size > 0
    assert qc_html.exists() and qc_html.stat().st_size > 0

    h5_path = str(Path(out_dir) / "noise_test.ctdna_vaf.h5")
    with pd.HDFStore(h5_path) as store:
        h5_keys = store.keys()
    assert "/df_ctdna_vaf_filt_signature_filt_no_noise_filter" in h5_keys

    qc_content = qc_html.read_text()
    assert "Noise Locus Filter" in qc_content
