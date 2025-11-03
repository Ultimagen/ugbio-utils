import json
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from ugbio_core.sorter_to_h5 import sorter_to_h5
from ugbio_core.sorter_utils import merge_sorter_json_files


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


def test_merge_sorter_json_files(tmpdir, resources_dir):  # noqa C901,PLR0912
    # Use the same JSON twice; expect all numeric counts doubled, metadata unchanged
    input_json = resources_dir / "435783-L10049-L10052-Z0145-CATGCTATAGCAATGAT.json"
    output_json = Path("/tmp") / "merged.json"

    merge_sorter_json_files(
        sorter_json_stats_files=[input_json, input_json],
        output_json_file=str(output_json),
        stringent_mode=True,
    )

    with open(input_json) as f:
        original = json.load(f)
    with open(output_json) as f:
        merged = json.load(f)

    def assert_doubled(orig_val, merged_val, path="root"):
        """Recursively verify that numeric values are doubled."""
        if isinstance(orig_val, int):
            assert merged_val == orig_val * 2, f"Mismatch at {path}: {merged_val} != {orig_val * 2}"
        elif isinstance(orig_val, list):
            assert len(merged_val) == len(orig_val), f"List length mismatch at {path}"
            for i, (ov, mv) in enumerate(zip(orig_val, merged_val)):
                if isinstance(ov, int):
                    assert mv == ov * 2, f"Mismatch at {path}[{i}]: {mv} != {ov * 2}"
                elif isinstance(ov, list):
                    assert_doubled(ov, mv, f"{path}[{i}]")
                else:
                    assert mv == ov, f"Non-numeric mismatch at {path}[{i}]"
        elif isinstance(orig_val, dict):
            assert set(orig_val.keys()) == set(merged_val.keys()), f"Keys mismatch at {path}"
            for k in orig_val:
                assert_doubled(orig_val[k], merged_val[k], f"{path}.{k}")
        else:
            # Non-numeric metadata should be identical
            assert merged_val == orig_val, f"Non-numeric mismatch at {path}: {merged_val} != {orig_val}"

    # Verify all fields
    for k, v in original.items():
        assert k in merged, f"Key {k} missing in merged output"
        if k == "extra_information":
            # Metadata should be unchanged
            assert merged[k] == v, f"Metadata changed: {merged[k]} != {v}"
        else:
            # All other values should be doubled
            assert_doubled(v, merged[k], k)


def test_merge_sorter_json_files_different_lengths(tmpdir, resources_dir):
    """Test merging histograms with different lengths."""
    json1 = Path(tmpdir) / "test1.json"
    json2 = Path(tmpdir) / "test2.json"
    output = Path(tmpdir) / "merged.json"

    data1 = {"histogram": [1, 2, 3], "count": 10, "extra_information": {"sample": "A"}}
    data2 = {"histogram": [4, 5], "count": 20, "extra_information": {"sample": "A"}}

    with open(json1, "w") as f:
        json.dump(data1, f)
    with open(json2, "w") as f:
        json.dump(data2, f)

    merge_sorter_json_files([json1, json2], str(output), stringent_mode=True)

    with open(output) as f:
        result = json.load(f)

    assert result["histogram"] == [5, 7, 3]  # [1+4, 2+5, 3+0]
    assert result["count"] == 30
    assert result["extra_information"] == {"sample": "A"}


def test_merge_sorter_json_files_missing_keys_stringent(tmpdir):
    """Test that missing keys raise error in stringent mode."""
    json1 = Path(tmpdir) / "test1.json"
    json2 = Path(tmpdir) / "test2.json"
    output = Path(tmpdir) / "merged.json"

    data1 = {"field1": 10, "extra_information": {}}
    data2 = {"field2": 20, "extra_information": {}}

    with open(json1, "w") as f:
        json.dump(data1, f)
    with open(json2, "w") as f:
        json.dump(data2, f)

    with pytest.raises(KeyError, match="missing in input file"):
        merge_sorter_json_files([json1, json2], str(output), stringent_mode=True)


def test_merge_sorter_json_files_metadata_mismatch(tmpdir):
    """Test that metadata mismatches raise error in stringent mode."""
    json1 = Path(tmpdir) / "test1.json"
    json2 = Path(tmpdir) / "test2.json"
    output = Path(tmpdir) / "merged.json"

    data1 = {"count": 10, "extra_information": {"sample": "A"}}
    data2 = {"count": 20, "extra_information": {"sample": "B"}}

    with open(json1, "w") as f:
        json.dump(data1, f)
    with open(json2, "w") as f:
        json.dump(data2, f)

    with pytest.raises(ValueError, match="Metadata 'extra_information' differs"):
        merge_sorter_json_files([json1, json2], str(output), stringent_mode=True)
