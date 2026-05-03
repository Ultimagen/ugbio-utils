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


@pytest.mark.parametrize("adapter_version", ["v1", "legacy_v5", "Solaris_1", "Solaris_2"])
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


def test_ppmseq_qc_analysis_with_sorter_csv(tmp_path, resources_dir, subsampled_sam):
    """Per reviewer: make sure the sorter_csv input is still exercised end-to-end."""
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
            "--output-path",
            str(tmp_path),
            "--output-basename",
            "ppmseq_sr_tag_full",
        ]
    )
    html = tmp_path / "ppmseq_sr_tag_full.ppmSeq.applicationQC.html"
    assert html.exists()
