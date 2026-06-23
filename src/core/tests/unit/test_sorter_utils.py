from pathlib import Path

import pandas as pd
import pytest
from ugbio_core.sorter_utils import get_base_coverage_from_sorter, read_sorter_statistics_csv


@pytest.fixture
def resources_dir():
    return Path(__file__).parent.parent / "resources"


@pytest.fixture
def sorter_json(resources_dir):
    return str(resources_dir / "603559-L13064-Z0152-CATGCAACACTAGAT.json")


@pytest.fixture
def sorter_csv(resources_dir):
    return str(resources_dir / "603559-L13064-Z0152-CATGCAACACTAGAT.csv")


def test_get_base_coverage_from_sorter(sorter_json):
    result = get_base_coverage_from_sorter(sorter_json)
    assert isinstance(result, dict)
    assert len(result) > 0
    assert "Genome" in result
    for region, series in result.items():
        assert isinstance(series, pd.Series)
        assert series.name == "count"
        assert len(series) > 0


def test_get_base_coverage_from_sorter_regions(sorter_json):
    result = get_base_coverage_from_sorter(sorter_json)
    expected_regions = {"Genome", "Exome", "Unique", "Non-unique"}
    assert expected_regions.issubset(set(result.keys()))


def test_read_sorter_statistics_csv_series_default(sorter_csv):
    result = read_sorter_statistics_csv(sorter_csv)
    assert isinstance(result, pd.Series)
    assert result.index.name == "metric"


def test_read_sorter_statistics_csv_as_dataframe(sorter_csv):
    result = read_sorter_statistics_csv(sorter_csv, edit_metric_names=False, as_dataframe=True)
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["metric", "value"]
    assert len(result) > 0
    # value column is numeric (parsed from the raw CSV strings)
    assert pd.api.types.is_numeric_dtype(result["value"])


def test_read_sorter_statistics_csv_dataframe_matches_series(sorter_csv):
    series = read_sorter_statistics_csv(sorter_csv, edit_metric_names=False)
    result = read_sorter_statistics_csv(sorter_csv, edit_metric_names=False, as_dataframe=True)
    assert result["metric"].tolist() == series.index.tolist()
    assert result["value"].tolist() == series.tolist()


def test_read_sorter_statistics_csv_with_category_row(tmp_path):
    csv_content = "PF_Barcode_reads,1000000\nMean_cvg,30.5\nCATEGORY,PAIRED\nMean_Read_Length,150.0\n"
    csv_file = tmp_path / "sorter_stats.csv"
    csv_file.write_text(csv_content)

    result = read_sorter_statistics_csv(str(csv_file))

    assert isinstance(result, pd.Series)
    assert "CATEGORY" not in result.index
    assert result["PF_Barcode_reads"] == 1000000
    assert result["Mean_cvg"] == 30.5
    assert result["Mean_Read_Length"] == 150.0
