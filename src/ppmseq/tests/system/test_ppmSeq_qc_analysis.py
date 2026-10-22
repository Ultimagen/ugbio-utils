# noqa: N999
import base64
import re
from pathlib import Path

import pytest
from ugbio_ppmseq import ppmSeq_qc_analysis
from ugbio_ppmseq.ppmSeq_utils import BASE_PATH, REPORTS_DIR

# Section headers / rows that only make sense when the read has an end tag.
END_TAG_REPORT_MARKERS = (
    "Start / end tag concordance",
    "Strand-ratio by end-tag category",
    "PCT_MIXED_both_tags",
    "PCT_read_end_unreached",
    "strand_ratio_category_concordance",
    "strand_ratio_category_consensus",
    "strand_ratio_category_end",
    "end-loop bars",
)


def _run_report(tmp_path, sam, adapter_version, basename):
    ppmSeq_qc_analysis.run(
        [
            "ppmSeq_qc_analysis",
            "--adapter-version",
            adapter_version,
            "--subsampled-sam",
            str(sam),
            "--output-path",
            str(tmp_path),
            "--output-basename",
            basename,
        ]
    )
    html = tmp_path / f"{basename}.ppmSeq.applicationQC.html"
    assert html.exists()
    return html.read_text()


def _embeds(html_text, png_name):
    encoded = base64.b64encode((BASE_PATH / REPORTS_DIR / png_name).read_bytes()).decode()
    return encoded in html_text


@pytest.fixture
def resources_dir():
    return Path(__file__).parent.parent / "resources"


@pytest.fixture
def subsampled_sam(resources_dir):
    return resources_dir / "ppmseq_sr_tag" / "Z0263_sample.sam.gz"


@pytest.mark.parametrize("adapter_version", ["v1", "legacy_v5", "Solaris2_ppmSeq", "Solaris2_ppmSeq_plus"])
def test_ppmseq_qc_analysis_runs(tmp_path, subsampled_sam, adapter_version):
    """Smoke-test the SAM-only happy path across every supported adapter version, with
    report generation disabled (skips the jupyter execution path so this is pure-Python)."""
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
            "--no-generate-report",
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
    """When no read in the subsampled SAM carries an sr tag, the sr-only sections (§4 and
    §5 in the flat-numbered report) must be omitted — everything else still renders."""
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
    # Always-present sections from the flat numbering.
    assert "1. Key metrics" in text
    assert "3. Start / end tag concordance" in text
    assert "6. Read-length distribution" in text
    # The sr-only section headers must be absent entirely.
    assert "4. Overall strand-ratio distribution" not in text
    assert "5. Strand-ratio by end-tag category" not in text


@pytest.mark.parametrize("adapter_version", ["v1", "Solaris2_ppmSeq"])
def test_ppmseq_qc_report_with_end_tag(tmp_path, subsampled_sam, adapter_version):
    """v1 and Solaris2_ppmSeq (regular ppmSeq) keep every end-tag section and the
    two-loop read-structure diagram."""
    text = _run_report(tmp_path, subsampled_sam, adapter_version, f"report_{adapter_version}")
    for marker in END_TAG_REPORT_MARKERS:
        assert marker in text, marker
    for header in ("4. Overall strand-ratio distribution", "6. Read-length distribution", "9.3 Read structure"):
        assert header in text
    assert _embeds(text, "ppmSeq_v1_illustration.png")
    assert not _embeds(text, "ppmSeq_plus_illustration.png")


def test_ppmseq_qc_report_solaris2_ppmseq_matches_v1(tmp_path, subsampled_sam):
    """Solaris2_ppmSeq is regular ppmSeq: its report text is identical to v1."""
    text_v1 = _run_report(tmp_path, subsampled_sam, "v1", "same")
    text_s2 = _run_report(tmp_path / "s2", subsampled_sam, "Solaris2_ppmSeq", "same")

    def _strip_volatile(text):
        # Drop embedded images, generated element ids, the report date and output paths.
        text = re.sub(r'src="data:image/png;base64,[^"]+"', "", text)
        text = re.sub(r'id="[^"]*"', "", text)
        text = re.sub(r"\d{4}-\d{2}-\d{2}", "", text)
        return text.replace(str(tmp_path / "s2"), str(tmp_path))

    assert _strip_volatile(text_v1) == _strip_volatile(text_s2)


@pytest.mark.parametrize(
    ("sam_name", "expected_headers"),
    [
        (
            "Z0263_sample.sam.gz",
            ("3. Overall strand-ratio distribution", "4. Read-length distribution", "7.3 Read structure"),
        ),
        ("Z0263_sample_no_sr.sam.gz", ("3. Read-length distribution", "6.3 Read structure")),
    ],
)
def test_ppmseq_qc_report_ppmseq_plus_has_no_end_tag(tmp_path, resources_dir, sam_name, expected_headers):
    """Solaris2_ppmSeq_plus (single start loop, no end tag): every end-tag section, row
    and table is dropped, the remaining sections are renumbered, and the single-loop
    read-structure diagram is shown."""
    sam = resources_dir / "ppmseq_sr_tag" / sam_name
    text = _run_report(tmp_path, sam, "Solaris2_ppmSeq_plus", "report_plus")
    for marker in END_TAG_REPORT_MARKERS:
        assert marker not in text, marker
    assert "PCT_MIXED_start_tag" in text
    for header in expected_headers:
        assert header in text
    assert _embeds(text, "ppmSeq_plus_illustration.png")
    assert not _embeds(text, "ppmSeq_v1_illustration.png")


def test_ppmseq_qc_analysis_with_sorter_csv(tmp_path, resources_dir, subsampled_sam):
    """End-to-end run with sorter stats + failure codes + a free-form --extra-arg
    surfaced in the report header."""
    sorter_csv = resources_dir / "ppmseq_sr_tag" / "Z0263_sorter_stats.csv"
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
    # The raw string "version" is a generic word that can appear incidentally in any HTML
    # file, so assert only on the uniquely-identifying value.
    assert "1.2.3.4" in html_text
