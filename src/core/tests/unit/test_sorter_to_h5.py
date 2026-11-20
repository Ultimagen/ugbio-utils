from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from ugbio_core.sorter_to_h5 import sorter_to_h5


@pytest.fixture
def resources_dir():
    return Path(__file__).parent.parent / "resources"


def test_sorter_to_h5(tmpdir, resources_dir):
    base_file_name = "026532-Lb_1866-Z0058-CATCTCAGTGCAATGAT"
    output_h5_file = Path(tmpdir) / f"{base_file_name}.aggregated_metrics.h5"
    sorter_to_h5(
        input_csv_file=resources_dir / f"{base_file_name}.csv",
        input_json_file=resources_dir / f"{base_file_name}.json",
        output_h5_file=output_h5_file,
    )
    expected_output_file = resources_dir / f"{base_file_name}.aggregated_metrics.h5"
    with pd.HDFStore(output_h5_file) as hdf:
        output_h5_keys = hdf.keys()
    with pd.HDFStore(expected_output_file) as hdf:
        expected_output_file_keys = hdf.keys()
    assert output_h5_keys == expected_output_file_keys
    for key in output_h5_keys:
        assert_frame_equal(pd.read_hdf(output_h5_file, key), pd.read_hdf(expected_output_file, key))


def test_sorter_to_h5_pct_contamination(tmpdir, resources_dir):
    """PCT_contamination is mapped to Contamination/PCT_CONTAMINATION when present in the sorter csv."""
    base_file_name = "026532-Lb_1866-Z0058-CATCTCAGTGCAATGAT"
    input_csv_file = Path(tmpdir) / f"{base_file_name}.csv"
    with open(resources_dir / f"{base_file_name}.csv", encoding="utf-8") as src:
        csv_content = src.read()
    if not csv_content.endswith("\n"):
        csv_content += "\n"
    input_csv_file.write_text(csv_content + "PCT_contamination,0.28\n", encoding="utf-8")
    output_h5_file = Path(tmpdir) / f"{base_file_name}.aggregated_metrics.h5"
    sorter_to_h5(
        input_csv_file=input_csv_file,
        input_json_file=resources_dir / f"{base_file_name}.json",
        output_h5_file=output_h5_file,
    )
    expected_output_file = resources_dir / f"{base_file_name}.aggregated_metrics.h5"
    with pd.HDFStore(output_h5_file) as hdf:
        output_h5_keys = hdf.keys()
    with pd.HDFStore(expected_output_file) as hdf:
        expected_output_file_keys = hdf.keys()
    assert sorted(output_h5_keys) == sorted([*expected_output_file_keys, "/Contamination"])
    for key in expected_output_file_keys:
        assert_frame_equal(pd.read_hdf(output_h5_file, key), pd.read_hdf(expected_output_file, key))
    assert pd.read_hdf(output_h5_file, "Contamination")["PCT_CONTAMINATION"].iloc[0] == pytest.approx(0.28)
