from pathlib import Path

import pandas as pd
import pytest
from ugbio_core.sorter_utils import get_base_coverage_from_sorter


@pytest.fixture
def resources_dir():
    return Path(__file__).parent.parent / "resources"


@pytest.fixture
def sorter_json(resources_dir):
    return str(resources_dir / "603559-L13064-Z0152-CATGCAACACTAGAT.json")


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
