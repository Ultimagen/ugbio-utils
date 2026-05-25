from pathlib import Path

import pytest
from ugbio_core.sorter_multi_sample_report import (
    _build_multi_summary_table_html,
    generate_multi_sample_report,
    run,
)
from ugbio_core.sorter_sample_discovery import (
    SampleData,
    _derive_sample_label,
    _discover_samples_local,
    _has_named_sample,
    _load_sample,
    _parse_library_info_xml,
)


@pytest.fixture
def resources_dir():
    return Path(__file__).parent.parent / "resources"


@pytest.fixture
def sample_1_json(resources_dir):
    return resources_dir / "603559-L13064-Z0152-CATGCAACACTAGAT.json"


@pytest.fixture
def sample_1_csv(resources_dir):
    return resources_dir / "603559-L13064-Z0152-CATGCAACACTAGAT.csv"


@pytest.fixture
def sample_2_json(resources_dir):
    return resources_dir / "026532-Lb_1866-Z0058-CATCTCAGTGCAATGAT.json"


@pytest.fixture
def sample_2_csv(resources_dir):
    return resources_dir / "026532-Lb_1866-Z0058-CATCTCAGTGCAATGAT.csv"


@pytest.fixture
def run_dir_with_samples(tmp_path, sample_1_json, sample_1_csv, sample_2_json, sample_2_csv):
    d1 = tmp_path / "603559-L13064-Z0152-CATGCAACACTAGAT"
    d1.mkdir()
    (d1 / sample_1_json.name).symlink_to(sample_1_json)
    (d1 / sample_1_csv.name).symlink_to(sample_1_csv)

    d2 = tmp_path / "026532-Lb_1866-Z0058-CATCTCAGTGCAATGAT"
    d2.mkdir()
    (d2 / sample_2_json.name).symlink_to(sample_2_json)
    (d2 / sample_2_csv.name).symlink_to(sample_2_csv)

    return tmp_path


@pytest.fixture
def run_dir_with_xml(tmp_path, sample_1_json, sample_1_csv):
    """Run dir with a LibraryInfo XML that only lists sample 1."""
    d1 = tmp_path / "603559-L13064-Z0152-CATGCAACACTAGAT"
    d1.mkdir()
    (d1 / sample_1_json.name).symlink_to(sample_1_json)
    (d1 / sample_1_csv.name).symlink_to(sample_1_csv)

    d2 = tmp_path / "603559-UGAv3-1000-CGTGCAATGCGCATGAT"
    d2.mkdir()
    (d2 / sample_1_json.name).symlink_to(sample_1_json)
    (d2 / sample_1_csv.name).symlink_to(sample_1_csv)

    xml_content = """<SampleInfo RunId="603559" Library_Pool="TEST">
  <Samples>
    <Sample Id="L13064@L13064" Index_Label="Z0152" Index_Sequence="CATGCAACACTAGAT">
    </Sample>
  </Samples>
</SampleInfo>"""
    (tmp_path / "603559_LibraryInfo.xml").write_text(xml_content)

    return tmp_path


@pytest.fixture
def two_samples(run_dir_with_samples):
    samples = []
    for d in sorted(run_dir_with_samples.iterdir()):
        if d.is_dir():
            s = _load_sample(d)
            if s is not None:
                samples.append(s)
    return samples


class TestDeriveLabel:
    def test_strips_run_id_prefix(self):
        assert _derive_sample_label("603559-L13064-Z0152-CATGCAACACTAGAT") == "L13064-Z0152-CATGCAACACTAGAT"

    def test_no_prefix(self):
        assert _derive_sample_label("sample_only") == "sample_only"


class TestHasNamedSample:
    def test_named_sample(self):
        assert _has_named_sample("603559-L13064-Z0152-CATGCAACACTAGAT") is True

    def test_junk_barcode(self):
        assert _has_named_sample("603559-UGAv3-1000-CGTGCAATGCGCATGAT") is True

    def test_barcode_only(self):
        assert _has_named_sample("603559-Z0152-CATGCAACACTAGAT") is False

    def test_special_tt(self):
        assert _has_named_sample("603559-TT-TT") is False


class TestParseLibraryInfoXml:
    def test_parses_samples(self, run_dir_with_xml):
        xml_path = run_dir_with_xml / "603559_LibraryInfo.xml"
        suffixes = _parse_library_info_xml(xml_path)
        assert "L13064-Z0152-CATGCAACACTAGAT" in suffixes
        assert len(suffixes) == 1

    def test_handles_invalid_xml(self, tmp_path):
        bad_xml = tmp_path / "bad_LibraryInfo.xml"
        bad_xml.write_text("not valid xml <<<<")
        suffixes = _parse_library_info_xml(bad_xml)
        assert suffixes == set()


class TestDiscoverSamplesLocal:
    def test_finds_named_samples(self, run_dir_with_samples):
        found = _discover_samples_local(run_dir_with_samples)
        assert len(found) == 2

    def test_filters_by_xml(self, run_dir_with_xml):
        found = _discover_samples_local(run_dir_with_xml)
        assert len(found) == 1
        assert "L13064" in found[0].name

    def test_skips_junk_without_xml(self, tmp_path, sample_1_json, sample_1_csv):
        junk = tmp_path / "603559-TT-TT"
        junk.mkdir()
        (junk / sample_1_json.name).symlink_to(sample_1_json)
        (junk / sample_1_csv.name).symlink_to(sample_1_csv)
        found = _discover_samples_local(tmp_path)
        assert len(found) == 0

    def test_skips_empty_dirs(self, tmp_path):
        (tmp_path / "603559-L13064-Z0152-CATGCAACACTAGAT").mkdir()
        found = _discover_samples_local(tmp_path)
        assert len(found) == 0

    def test_skips_files(self, tmp_path):
        (tmp_path / "file.txt").write_text("hello")
        found = _discover_samples_local(tmp_path)
        assert len(found) == 0


class TestLoadSample:
    def test_loads_valid_sample(self, run_dir_with_samples):
        sample_dir = sorted(run_dir_with_samples.iterdir())[0]
        sample = _load_sample(sample_dir)
        assert sample is not None
        assert isinstance(sample, SampleData)
        assert sample.label
        assert sample.stats_json
        assert sample.csv_df is not None
        assert "metric" in sample.csv_df.columns
        assert "value" in sample.csv_df.columns
        assert sample.base_coverage

    def test_returns_none_for_empty_dir(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        assert _load_sample(empty) is None


class TestSummaryTable:
    def test_table_contains_sample_labels(self, two_samples):
        html = _build_multi_summary_table_html(two_samples)
        for s in two_samples:
            assert s.label in html

    def test_table_has_metrics_as_columns(self, two_samples):
        html = _build_multi_summary_table_html(two_samples)
        assert "Mean_cvg" in html
        assert "PF_Barcode_reads" in html
        assert "median_cvg" in html


class TestGenerateReport:
    def test_generates_html_with_all_figures(self, tmp_path, two_samples):
        output_html = tmp_path / "report.html"
        result = generate_multi_sample_report(two_samples, output_html, "Test Report")
        assert result == output_html
        assert output_html.exists()
        content = output_html.read_text()
        assert "Ultima Genomics Multi-Sample QC Report" in content
        assert "Test Report" in content
        assert "plotly" in content.lower()
        for s in two_samples:
            assert s.label in content

    def test_creates_parent_dirs(self, tmp_path, two_samples):
        output_html = tmp_path / "nested" / "dir" / "report.html"
        result = generate_multi_sample_report(two_samples, output_html, "Test")
        assert result.exists()


class TestCLI:
    def test_run_with_run_dir(self, run_dir_with_samples):
        output = run_dir_with_samples / "multi_sample_report.html"
        run(["sorter_multi_sample_report", "--run-dir", str(run_dir_with_samples)])
        assert output.exists()
        content = output.read_text()
        assert "Multi-Sample QC Report" in content

    def test_run_with_input_dirs(self, tmp_path, run_dir_with_samples):
        sample_dirs = sorted(d for d in run_dir_with_samples.iterdir() if d.is_dir())
        output = tmp_path / "out.html"
        run(
            [
                "sorter_multi_sample_report",
                "--input-dir",
                str(sample_dirs[0]),
                "--input-dir",
                str(sample_dirs[1]),
                "--output",
                str(output),
                "--title",
                "Custom Title",
            ]
        )
        assert output.exists()
        content = output.read_text()
        assert "Custom Title" in content

    def test_run_no_args_raises(self):
        with pytest.raises(ValueError, match="At least one"):
            run(["sorter_multi_sample_report"])
