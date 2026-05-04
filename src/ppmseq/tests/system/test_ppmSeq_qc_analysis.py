# noqa: N999
from pathlib import Path

import pytest
from ugbio_ppmseq import ppmSeq_qc_analysis


@pytest.fixture
def resources_dir():
    return Path(__file__).parent.parent / "resources"


@pytest.fixture
def subsampled_sam(resources_dir):
    return resources_dir / "ppmseq_sr_tag" / "Z0263_sample.sam.gz"


@pytest.mark.parametrize("adapter_version", ["v1", "legacy_v5"])
def test_ppmseq_qc_analysis_runs(tmp_path, subsampled_sam, adapter_version):
    """Smoke-test the SAM-only happy path across every supported adapter version."""
    ppmSeq_qc_analysis.run(
        [
            "ppmSeq_qc_analysis",
            "--adapter-version",
            adapter_version,
            "--subsampled-sam",
            str(subsampled_sam),
            "--output-path",
            str(tmp_path),
            "--output-basename",
            f"ppmseq_sr_tag_{adapter_version}",
            "--generate-report",
            "",
        ]
    )
    h5 = tmp_path / f"ppmseq_sr_tag_{adapter_version}.ppmSeq.applicationQC.h5"
    assert h5.exists()
    json_out = tmp_path / f"ppmseq_sr_tag_{adapter_version}.ppmSeq.applicationQC.json"
    assert json_out.exists()


def test_ppmseq_qc_analysis_with_report(tmp_path, subsampled_sam):
    """End-to-end check that report generation and the notebook template actually work.
    Per the reviewer: the system test should exercise the report path so notebook-template
    breakage is caught by CI."""
    ppmSeq_qc_analysis.run(
        [
            "ppmSeq_qc_analysis",
            "--adapter-version",
            "v1",
            "--subsampled-sam",
            str(subsampled_sam),
            "--output-path",
            str(tmp_path),
            "--output-basename",
            "ppmseq_sr_tag_report",
        ]
    )
    html = tmp_path / "ppmseq_sr_tag_report.ppmSeq.applicationQC.html"
    assert html.exists()


def test_ppmseq_qc_analysis_no_sr(tmp_path, resources_dir):
    """When no read in the subsampled SAM carries an sr tag, sections 1.4 and 1.5 (and the
    associated plots) must be omitted from the rendered report — everything else still
    renders normally."""
    sam_no_sr = resources_dir / "ppmseq_sr_tag" / "Z0263_sample_no_sr.sam.gz"
    ppmSeq_qc_analysis.run(
        [
            "ppmSeq_qc_analysis",
            "--adapter-version",
            "v1",
            "--subsampled-sam",
            str(sam_no_sr),
            "--output-path",
            str(tmp_path),
            "--output-basename",
            "ppmseq_no_sr",
        ]
    )
    html = tmp_path / "ppmseq_no_sr.ppmSeq.applicationQC.html"
    assert html.exists()
    text = html.read_text()
    # Sections 1.1 / 1.2 / 1.3 / 2.x / 3.x / 4.x should all still be there.
    assert "1.1 Strand-ratio category percentages" in text
    assert "1.3 Start / end tag concordance" in text
    assert "3.2 Read-length distribution" in text
    # The sr-only sections and figures must be gone.
    assert "1.4 Overall strand-ratio distribution" not in text
    assert "1.5 Strand-ratio by end-tag category" not in text
    assert "Figure 1.3." not in text  # the sr-hist caption
    assert "Figure 1.4." not in text  # the sr-by-et caption


def test_ppmseq_qc_analysis_with_sorter_csv(tmp_path, resources_dir, subsampled_sam):
    """End-to-end run with sorter stats + failure codes + a free-form --extra-arg
    surfaced in the report header."""
    sorter_csv = resources_dir / "130713-UGAv3-51.sorter_stats.csv"
    trimmer_failure_codes = resources_dir / "412884-L6860-Z0293-CATGTGAGCGGTGAT_trimmer-failure_codes.csv"
    ppmSeq_qc_analysis.run(
        [
            "ppmSeq_qc_analysis",
            "--adapter-version",
            "v1",
            "--subsampled-sam",
            str(subsampled_sam),
            "--sorter-stats-csv",
            str(sorter_csv),
            "--trimmer-failure-codes-csv",
            str(trimmer_failure_codes),
            "--extra-arg",
            "version=1.2.3.4",
            "--output-path",
            str(tmp_path),
            "--output-basename",
            "ppmseq_sr_tag_full",
        ]
    )
    html = tmp_path / "ppmseq_sr_tag_full.ppmSeq.applicationQC.html"
    assert html.exists()
    # The --extra-arg key/value should appear in the rendered report body.
    html_text = html.read_text()
    assert "version" in html_text
    assert "1.2.3.4" in html_text
