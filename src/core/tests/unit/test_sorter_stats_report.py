from pathlib import Path

import pytest
from ugbio_core.sorter_stats_report import generate_sorter_stats_report, run


@pytest.fixture
def resources_dir():
    return Path(__file__).parent.parent / "resources"


@pytest.fixture
def sorter_json(resources_dir):
    return resources_dir / "603559-L13064-Z0152-CATGCAACACTAGAT.json"


@pytest.fixture
def sorter_csv(resources_dir):
    return resources_dir / "603559-L13064-Z0152-CATGCAACACTAGAT.csv"


def test_generate_sorter_stats_report(tmp_path, sorter_json, sorter_csv):
    output_html = tmp_path / "report.html"
    result = generate_sorter_stats_report(sorter_json, sorter_csv, output_html)
    assert result == output_html
    assert output_html.exists()
    content = output_html.read_text()
    assert "Ultima Genomics Sequencing QC Report" in content
    assert "603559-L13064-Z0152-CATGCAACACTAGAT" in content
    assert "plotly" in content.lower()


def test_generate_report_creates_parent_dirs(tmp_path, sorter_json, sorter_csv):
    output_html = tmp_path / "nested" / "dir" / "report.html"
    result = generate_sorter_stats_report(sorter_json, sorter_csv, output_html)
    assert result.exists()


def test_run_with_explicit_paths(tmp_path, sorter_json, sorter_csv):
    output_html = tmp_path / "report.html"
    run(["sorter_stats_report", "--json", str(sorter_json), "--csv", str(sorter_csv), "--output", str(output_html)])
    assert output_html.exists()


def test_run_with_input_dir(tmp_path, sorter_json, sorter_csv):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / sorter_json.name).symlink_to(sorter_json)
    (input_dir / sorter_csv.name).symlink_to(sorter_csv)
    output_html = tmp_path / "603559-L13064-Z0152-CATGCAACACTAGAT.html"
    run(["sorter_stats_report", "--input-dir", str(input_dir), "--output", str(output_html)])
    assert output_html.exists()
