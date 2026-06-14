"""Integration test: generate report with synthetic signatures from PCAWG.

This test uses pre-generated test resources (synthetic signature VCFs and
intersection parquets from bcftools isec) to run the full report pipeline
and verify detection statistics are meaningful.

Parametrized over [5, 20, 100] synthetic controls. With N>=100 controls,
p-value resolution reaches < 0.01 (full statistical power).

The N=100 case is marked @pytest.mark.slow and excluded from the default
pytest run (CI). Run it explicitly with:
    pytest -m slow src/mrd/tests/unit/test_mrd_report_with_synthetics.py
"""

import json
from pathlib import Path

import pytest
from ugbio_mrd.generate_mrd_report import MrdReportInputs, generate_mrd_report


@pytest.fixture(scope="module")
def resources_dir():
    return Path(__file__).parent.parent / "resources" / "report"


@pytest.mark.parametrize("n_controls", [5, 20, pytest.param(100, marks=pytest.mark.slow)])
def test_generate_report_with_synthetics(resources_dir, tmp_path_factory, n_controls):
    """Full report generation with N synthetic controls.

    Verifies the detection framework produces valid results.
    With N>=100 controls, p-value resolution reaches p < 0.01.
    """
    # Build file lists for the requested number of controls
    vcf_files = []
    parquet_files = []
    for i in range(n_controls):
        vcf = resources_dir / f"syn{i}_Pa_46_FreshFrozen.ann.chr20.filtered_pancan_pcawg_2020.chr20.filtered.vcf.gz"
        parquet = resources_dir / f"Pa_46_333_LuNgs_08.syn{i}_Pa_46_FreshFrozen.db_control.intersection.parquet"
        if not vcf.exists() or not parquet.exists():
            pytest.skip(f"Missing resource for syn{i}")
        vcf_files.append(str(vcf))
        parquet_files.append(str(parquet))

    output_path = tmp_path_factory.mktemp(f"report_output_{n_controls}")

    mrd_report_inputs = MrdReportInputs(
        intersected_featuremaps_parquet=[
            str(resources_dir / "Pa_46_333_LuNgs_08.Pa_46_FreshFrozen.matched.intersection.parquet"),
            str(resources_dir / "Pa_46_333_LuNgs_08.Pa_67_FFPE.control.intersection.parquet"),
        ]
        + parquet_files,
        matched_signatures_vcf_files=[str(resources_dir / "Pa_46_FreshFrozen.ann.chr20.filtered.vcf.gz")],
        control_signatures_vcf_files=[str(resources_dir / "Pa_67_FFPE.ann.chr20.filtered.vcf.gz")],
        db_control_signatures_vcf_files=vcf_files,
        coverage_bed=str(resources_dir / "Pa_46_333_LuNgs_08.regions.bed.gz"),
        output_dir=str(output_path),
        output_basename=f"test_{n_controls}syn",
        featuremap_file=str(resources_dir / "Pa_46_333_LuNgs_08.featuremap_df.10k.parquet"),
        signature_filter_query="(norm_coverage <= 2.5) and (norm_coverage >= 0.6)",
        read_filter_query="filt>0 and snvq>60 and mapq>=60",
        srsnv_metadata_json=str(resources_dir / "Pa_46_333_LuNgs_08.srsnv_metadata.json"),
    )

    output_report_html = generate_mrd_report(mrd_report_inputs)

    assert output_report_html.exists()
    assert output_report_html.stat().st_size > 0

    detection_json_path = output_path / f"test_{n_controls}syn.detection_result.json"
    assert detection_json_path.exists()

    with open(detection_json_path) as f:
        detection = json.load(f)

    assert detection["n_synthetic_controls"] == n_controls
    assert 0 < detection["p_value"] <= 1.0
    assert detection["call"] in ("MRD Detected", "MRD Not Detected", "Indeterminate")
    assert detection["matched_supporting_reads"] >= 0
    assert detection["signature_size"] > 0
    assert detection["mean_coverage"] > 0

    # With N>=100, matched signal should exceed all synthetic noise → detection
    if n_controls >= 100:
        assert detection["p_value"] < 0.05, (
            f"Expected detection with {n_controls} controls, got p={detection['p_value']}"
        )
        assert detection["detected"] is True
        assert detection["personal_lod"] is not None

