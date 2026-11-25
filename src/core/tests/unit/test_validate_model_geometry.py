import json

import onnx
import pytest
from onnx import TensorProto, helper
from ugbio_core.validate_model_geometry import (
    FAIL,
    METADATA_KEY,
    PASS,
    SKIP,
    WARN,
    normalize_run_settings,
    read_model,
    run,
    run_checks,
)

METADATA = {
    "schema_version": 1,
    "image": {"height": 121, "width": 221, "num_channels": 9, "channels": None},
    "sample_heights": [100, 21],
    "reference_rows": 5,
    "optimal_coverages": [1000],
    "cap_at_optimal_coverage": False,
    "prioritize_alt_supporting_reads": False,
    "add_ins_size_channel": True,
    "max_ins_size": 50,
    "haps_height": 16,
    "paired_end_mode": False,
    "make_examples_docker": "make_examples:3.4.1",
}

RUN = {
    "sample_heights": [100, 21],
    "optimal_coverages": "1000",
    "cap_at_optimal_coverage": False,
    "prioritize_alt_supporting_reads": False,
    "ensemble_reference_rows": 5,
    "add_ins_size_channel": True,
    "make_examples_extra_args": "--max-ins-size 50",
}


def make_model(path, shape=("batch", 121, 221, 9), metadata=None):
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, list(shape))
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, list(shape))
    graph = helper.make_graph([helper.make_node("Identity", ["input"], ["output"])], "g", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    if metadata is not None:
        model.metadata_props.add(key=METADATA_KEY, value=json.dumps(metadata))
    onnx.save(model, str(path))
    return str(path)


def statuses(results):
    return {r.name: r.status for r in results}


def test_read_model(tmp_path):
    shape, metadata = read_model(make_model(tmp_path / "m.onnx", metadata=METADATA))
    assert shape == ["batch", 121, 221, 9]
    assert metadata == METADATA
    assert read_model(make_model(tmp_path / "legacy.onnx"))[1] is None


def test_normalize_run_settings():
    s = normalize_run_settings(
        {
            "sample_heights": ["50", "50"],
            "optimal_coverages": "500;120",
            "make_examples_extra_args": "--cap-at-optimal-coverage --channels a,b --image-width 21",
        }
    )
    assert s["sample_heights"] == [50, 50]
    assert s["optimal_coverages"] == [500, 120]
    assert s["cap_at_optimal_coverage"] is True
    assert s["channels"] == ["a", "b"]
    assert s["image_width"] == 21
    assert s["reference_rows"] is None
    assert normalize_run_settings({"sample_heights": [100], "optimal_coverages": ""})["optimal_coverages"] is None


def test_matching_model_passes():
    results = run_checks(["batch", 121, 221, 9], METADATA, normalize_run_settings(RUN))
    assert set(statuses(results).values()) == {PASS}


def test_legacy_model_shape_only():
    run_settings = normalize_run_settings(RUN)
    result = statuses(run_checks([100, 121, 221, 9], None, run_settings))
    assert result.pop("dimensions") == PASS
    assert set(result.values()) == {SKIP}
    assert statuses(run_checks([100, 200, 221, 9], None, run_settings))["dimensions"] == FAIL


def test_dimension_and_channel_failures():
    run_settings = normalize_run_settings(dict(RUN, sample_heights=[100, 100]))
    assert statuses(run_checks(["batch", 121, 221, 9], METADATA, run_settings))["dimensions"] == FAIL
    run_settings = normalize_run_settings(dict(RUN, make_examples_extra_args="--image-width 101"))
    assert statuses(run_checks(["batch", 121, 221, 9], METADATA, run_settings))["dimensions"] == FAIL
    run_settings = normalize_run_settings(dict(RUN, add_ins_size_channel=False))
    assert statuses(run_checks(["batch", 121, 221, 9], METADATA, run_settings))["channels"] == FAIL


@pytest.mark.parametrize(
    "change, check",
    [
        ({"sample_heights": [96, 25]}, "sample_heights"),
        ({"optimal_coverages": "40"}, "coverage"),
        ({"cap_at_optimal_coverage": True}, "coverage"),
        ({"prioritize_alt_supporting_reads": True}, "prioritize_alt"),
        ({"ensemble_reference_rows": 3}, "reference_rows"),
        ({"make_examples_extra_args": "--channels a,b"}, "channel_names"),
    ],
)
def test_setting_differences_warn_or_fail_in_strict(change, check):
    run_settings = normalize_run_settings(dict(RUN, **change))
    assert statuses(run_checks(["batch", 121, 221, 9], METADATA, run_settings))[check] == WARN
    assert statuses(run_checks(["batch", 121, 221, 9], METADATA, run_settings, strict=True))[check] == FAIL


def test_unknown_schema_version_falls_back_to_shape():
    metadata = dict(METADATA, schema_version=99)
    result = statuses(run_checks(["batch", 121, 221, 9], metadata, normalize_run_settings(RUN)))
    assert result["metadata"] == WARN and result["dimensions"] == PASS and result["coverage"] == SKIP


def test_run_cli(tmp_path):
    model = make_model(tmp_path / "m.onnx", metadata=METADATA)
    settings = tmp_path / "run.json"
    report = tmp_path / "report.json"
    settings.write_text(json.dumps(dict(RUN, optimal_coverages="40")))
    args = ["validate_model_geometry", "--model", model, "--run-settings", str(settings), "--output-json", str(report)]
    assert run(args) == 0
    assert json.loads(report.read_text())["passed"] is True
    assert run([*args, "--strict"]) == 1
    assert json.loads(report.read_text())["passed"] is False

    settings.write_text(json.dumps(dict(RUN, sample_heights=[100])))
    assert run(args) == 1
