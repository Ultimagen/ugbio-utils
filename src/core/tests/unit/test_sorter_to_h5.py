from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from ugbio_core.sorter_to_h5 import run, sorter_to_h5


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


def test_sorter_to_h5_with_contamination(tmpdir, resources_dir):
    base_file_name = "026532-Lb_1866-Z0058-CATCTCAGTGCAATGAT"
    output_h5_file = Path(tmpdir) / f"{base_file_name}.aggregated_metrics.h5"
    sorter_to_h5(
        input_csv_file=resources_dir / f"{base_file_name}.csv",
        input_json_file=resources_dir / f"{base_file_name}.json",
        output_h5_file=output_h5_file,
        contamination=0.0075,
    )
    expected_output_file = resources_dir / f"{base_file_name}.aggregated_metrics.h5"
    with pd.HDFStore(output_h5_file) as hdf:
        output_h5_keys = hdf.keys()
    with pd.HDFStore(expected_output_file) as hdf:
        expected_output_file_keys = hdf.keys()
    assert sorted(output_h5_keys) == sorted([*expected_output_file_keys, "/contamination"])
    for key in expected_output_file_keys:
        assert_frame_equal(pd.read_hdf(output_h5_file, key), pd.read_hdf(expected_output_file, key))
    contamination_df = pd.read_hdf(output_h5_file, "contamination")
    assert list(contamination_df.columns) == ["contamination"]
    assert contamination_df["contamination"].iloc[0] == pytest.approx(0.0075)


def test_sorter_to_h5_cli_contamination(tmpdir, resources_dir):
    base_file_name = "026532-Lb_1866-Z0058-CATCTCAGTGCAATGAT"
    output_h5_file = Path(tmpdir) / f"{base_file_name}.aggregated_metrics.h5"
    run(
        [
            "sorter_to_h5",
            "--input_csv_file",
            str(resources_dir / f"{base_file_name}.csv"),
            "--input_json_file",
            str(resources_dir / f"{base_file_name}.json"),
            "--output_h5_file",
            str(output_h5_file),
            "--contamination",
            "0.0003",
        ]
    )
    assert pd.read_hdf(output_h5_file, "contamination")["contamination"].iloc[0] == pytest.approx(0.0003)
