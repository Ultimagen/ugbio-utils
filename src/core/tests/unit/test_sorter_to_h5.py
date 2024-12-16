from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from ugbio_core.sorter_to_h5 import sorter_to_h5

BASE_PATH = Path(__file__).parent.parent.parent
REPORTS_PATH = BASE_PATH / "ugbio_core" / "reports"


@pytest.fixture
def resources_dir():
    return Path(__file__).parent.parent / "resources"


def test_sorter_to_h5(tmp_dir, resources_dir):
    base_file_name = "026532-Lb_1866-Z0058-CATCTCAGTGCAATGAT"
    output_h5 = sorter_to_h5(
        input_csv_file=resources_dir / f"{base_file_name}.csv",
        input_json_file=resources_dir / f"{base_file_name}.json",
        metric_mapping_file=REPORTS_PATH / "sorter_output_to_aggregated_metrics_h5.csv",
        output_dir=tmp_dir,
    )
    expected_output_file = resources_dir / f"{base_file_name}.aggregated_metrics.h5"
    with pd.HDFStore(output_h5) as hdf:
        output_h5_keys = hdf.keys()
    with pd.HDFStore(expected_output_file) as hdf:
        expected_output_file_keys = hdf.keys()
    assert output_h5_keys == expected_output_file_keys
    for key in output_h5_keys:
        assert_frame_equal(pd.read_hdf(output_h5, key), pd.read_hdf(expected_output_file, key))
