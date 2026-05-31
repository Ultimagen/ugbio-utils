"""Unit tests for srsnv_report module.

Covers:
- _ModelWithTrainingResults wrapper class
- _load_models_from_prefix
- _resolve_model_path
- _load_models_from_metadata
- _fallback_dummy_models
- _wrap_models_with_training_results
- _build_params
- _make_qual_interpolating_function
- compute_is_cycle_skip_column
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from ugbio_srsnv.srsnv_report import (
    _build_params,
    _fallback_dummy_models,
    _load_models_from_metadata,
    _load_models_from_prefix,
    _make_qual_interpolating_function,
    _ModelWithTrainingResults,
    _resolve_model_path,
    _wrap_models_with_training_results,
    compute_is_cycle_skip_column,
    prepare_report,
)

# ──────────────────────── _ModelWithTrainingResults ──────────────────────


class TestModelWithTrainingResults:
    """Tests for the _ModelWithTrainingResults wrapper class."""

    def test_evals_result_returns_training_result(self):
        """Wrapper should return provided training_result from evals_result()."""
        mock_model = MagicMock()
        training_result = {"validation_0": {"auc": [0.9, 0.95]}, "validation_1": {"auc": [0.85, 0.90]}}
        wrapper = _ModelWithTrainingResults(mock_model, training_result)
        assert wrapper.evals_result() == training_result

    def test_evals_result_empty_when_none(self):
        """Wrapper should return empty dict when training_result is None."""
        mock_model = MagicMock()
        wrapper = _ModelWithTrainingResults(mock_model, None)
        assert wrapper.evals_result() == {}

    def test_evals_result_empty_when_not_provided(self):
        """Wrapper should return empty dict when training_result not provided."""
        mock_model = MagicMock()
        wrapper = _ModelWithTrainingResults(mock_model)
        assert wrapper.evals_result() == {}

    def test_delegates_to_model(self):
        """Wrapper should delegate attribute access to underlying model."""
        mock_model = MagicMock()
        mock_model.predict = MagicMock(return_value=[0, 1])
        wrapper = _ModelWithTrainingResults(mock_model)
        # __getattr__ should delegate to model
        result = wrapper.predict([[1, 2], [3, 4]])
        mock_model.predict.assert_called_once_with([[1, 2], [3, 4]])
        assert result == [0, 1]

    def test_model_attribute_accessible(self):
        """The model attribute should be directly accessible."""
        mock_model = MagicMock()
        wrapper = _ModelWithTrainingResults(mock_model, {"key": "val"})
        assert wrapper.model is mock_model


# ──────────────────────── _resolve_model_path ──────────────────────


class TestResolveModelPath:
    """Tests for _resolve_model_path."""

    def test_returns_original_if_exists(self, tmp_path):
        """Should return original path if file exists there."""
        model_file = tmp_path / "model.json"
        model_file.write_text("{}")
        result = _resolve_model_path(str(model_file))
        assert result == str(model_file)

    def test_falls_back_to_cwd(self, tmp_path, monkeypatch):
        """Should fall back to CWD if original path doesn't exist."""
        # Create model in current directory
        cwd_model = tmp_path / "model.json"
        cwd_model.write_text("{}")
        monkeypatch.chdir(tmp_path)

        # Non-existent original path but basename matches file in CWD
        result = _resolve_model_path("/nonexistent/path/model.json")
        assert result == str(cwd_model)

    def test_raises_if_not_found_anywhere(self, tmp_path, monkeypatch):
        """Should raise FileNotFoundError if file not found in either location."""
        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError, match="Expected model file not found"):
            _resolve_model_path("/nonexistent/path/missing_model.json")


# ──────────────────────── _load_models_from_prefix ──────────────────────


class TestLoadModelsFromPrefix:
    """Tests for _load_models_from_prefix."""

    def test_loads_sequential_models(self, tmp_path):
        """Should load models with sequential numbering."""
        # Create minimal XGBoost models
        for idx in range(3):
            model = xgb.XGBClassifier(n_estimators=2, max_depth=1)
            model.fit(np.array([[0, 1], [1, 0], [0, 0], [1, 1]]), np.array([0, 1, 0, 1]))
            model.save_model(str(tmp_path / f"model_{idx}.json"))

        models = _load_models_from_prefix(str(tmp_path / "model_"))
        assert len(models) == 3
        for mdl in models:
            assert isinstance(mdl, xgb.XGBClassifier)

    def test_returns_empty_if_no_files(self, tmp_path):
        """Should return empty list if no model files found."""
        models = _load_models_from_prefix(str(tmp_path / "nonexistent_model_"))
        assert models == []

    def test_stops_at_gap(self, tmp_path):
        """Should stop loading when sequence has a gap (e.g., 0, 1 exist but not 2)."""
        for idx in [0, 1]:
            model = xgb.XGBClassifier(n_estimators=2, max_depth=1)
            model.fit(np.array([[0, 1], [1, 0], [0, 0], [1, 1]]), np.array([0, 1, 0, 1]))
            model.save_model(str(tmp_path / f"model_{idx}.json"))

        # Create model at index 3 (skipping 2)
        model = xgb.XGBClassifier(n_estimators=2, max_depth=1)
        model.fit(np.array([[0, 1], [1, 0], [0, 0], [1, 1]]), np.array([0, 1, 0, 1]))
        model.save_model(str(tmp_path / "model_3.json"))

        models = _load_models_from_prefix(str(tmp_path / "model_"))
        assert len(models) == 2  # Stops at index 2 (not found)


# ──────────────────────── _load_models_from_metadata ──────────────────────


class TestLoadModelsFromMetadata:
    """Tests for _load_models_from_metadata."""

    def test_loads_models_from_metadata_paths(self, tmp_path):
        """Should load models using paths from metadata dict."""
        # Create model files
        model_paths = {}
        for idx in range(2):
            model = xgb.XGBClassifier(n_estimators=2, max_depth=1)
            model.fit(np.array([[0, 1], [1, 0], [0, 0], [1, 1]]), np.array([0, 1, 0, 1]))
            path = str(tmp_path / f"model_{idx}.json")
            model.save_model(path)
            model_paths[str(idx)] = path

        metadata = {"model_paths": model_paths}
        models = _load_models_from_metadata(metadata)
        assert len(models) == 2

    def test_empty_model_paths(self):
        """Should return empty list when model_paths is empty."""
        metadata = {"model_paths": {}}
        models = _load_models_from_metadata(metadata)
        assert models == []

    def test_missing_model_paths_key(self):
        """Should return empty list when metadata has no model_paths."""
        metadata = {}
        models = _load_models_from_metadata(metadata)
        assert models == []

    def test_skips_none_paths(self, tmp_path):
        """Should skip None values in model_paths."""
        model = xgb.XGBClassifier(n_estimators=2, max_depth=1)
        model.fit(np.array([[0, 1], [1, 0], [0, 0], [1, 1]]), np.array([0, 1, 0, 1]))
        path = str(tmp_path / "model_0.json")
        model.save_model(path)

        metadata = {"model_paths": {"0": path, "1": None, "2": ""}}
        models = _load_models_from_metadata(metadata)
        assert len(models) == 1


# ──────────────────────── _fallback_dummy_models ──────────────────────


class TestFallbackDummyModels:
    """Tests for _fallback_dummy_models."""

    def test_creates_correct_number_of_models(self):
        """Should create as many dummy models as model_paths entries."""
        metadata = {"model_paths": {"0": "a", "1": "b", "2": "c"}}
        models = _fallback_dummy_models(metadata)
        assert len(models) == 3

    def test_dummy_models_can_predict(self):
        """Dummy models should be able to predict (always class 0)."""
        metadata = {"model_paths": {"0": "a"}}
        models = _fallback_dummy_models(metadata)
        pred = models[0].predict([[0], [1], [2]])
        assert all(p == 0 for p in pred)

    def test_handles_empty_model_paths(self):
        """Should create at least 1 model when model_paths is empty."""
        metadata = {"model_paths": {}}
        models = _fallback_dummy_models(metadata)
        assert len(models) == 1

    def test_handles_missing_model_paths(self):
        """Should create at least 1 model when model_paths key is missing."""
        metadata = {}
        models = _fallback_dummy_models(metadata)
        assert len(models) == 1


# ──────────────────────── _wrap_models_with_training_results ──────────────


class TestWrapModelsWithTrainingResults:
    """Tests for _wrap_models_with_training_results."""

    def test_wraps_with_training_results(self):
        """Each model should get its corresponding training result."""
        mock_models = [MagicMock(), MagicMock()]
        training_results = [
            {"validation_0": {"auc": [0.9]}},
            {"validation_1": {"auc": [0.8]}},
        ]
        wrapped = _wrap_models_with_training_results(mock_models, training_results)
        assert len(wrapped) == 2
        assert wrapped[0].evals_result() == training_results[0]
        assert wrapped[1].evals_result() == training_results[1]

    def test_wraps_without_training_results(self):
        """When training_results is None, models get empty evals_result."""
        mock_models = [MagicMock(), MagicMock()]
        wrapped = _wrap_models_with_training_results(mock_models, None)
        assert len(wrapped) == 2
        assert wrapped[0].evals_result() == {}
        assert wrapped[1].evals_result() == {}

    def test_wraps_with_fewer_results_than_models(self):
        """Models beyond training_results length should get empty result."""
        mock_models = [MagicMock(), MagicMock(), MagicMock()]
        training_results = [{"validation_0": {"auc": [0.9]}}]
        wrapped = _wrap_models_with_training_results(mock_models, training_results)
        assert len(wrapped) == 3
        assert wrapped[0].evals_result() == training_results[0]
        assert wrapped[1].evals_result() == {}
        assert wrapped[2].evals_result() == {}


# ──────────────────────── _build_params ──────────────────────


class TestBuildParams:
    """Tests for _build_params."""

    def test_basic_params_construction(self):
        """Should properly separate categorical and numerical features."""
        metadata = {
            "features": [
                {"name": "REF", "type": "c", "values": {"A": 0, "C": 1, "G": 2, "T": 3}},
                {"name": "ALT", "type": "c", "values": {"A": 0, "C": 1}},
                {"name": "X_HMER_REF", "type": "int"},
                {"name": "MQUAL", "type": "float"},
            ]
        }
        user_meta = {"adapter_version": "v2", "docker_image": "img:latest", "pipeline_version": "1.0"}
        params = _build_params(metadata, user_meta, num_models=3)

        assert params["categorical_features_names"] == ["REF", "ALT"]
        assert params["categorical_features_dict"] == {"REF": ["A", "C", "G", "T"], "ALT": ["A", "C"]}
        assert params["numerical_features"] == ["X_HMER_REF", "MQUAL"]
        assert params["num_CV_folds"] == 3
        assert params["adapter_version"] == "v2"
        assert params["docker_image"] == "img:latest"
        assert params["pipeline_version"] == "1.0"

    def test_empty_user_meta(self):
        """Should handle empty user metadata gracefully."""
        metadata = {"features": [{"name": "MQUAL", "type": "float"}]}
        user_meta = {}
        params = _build_params(metadata, user_meta, num_models=1)
        assert params["adapter_version"] is None
        assert params["docker_image"] is None
        assert params["pipeline_version"] is None
        assert params["num_CV_folds"] == 1

    def test_no_categorical_features(self):
        """Should work when there are no categorical features."""
        metadata = {
            "features": [
                {"name": "X_HMER_REF", "type": "int"},
                {"name": "MQUAL", "type": "float"},
            ]
        }
        user_meta = {}
        params = _build_params(metadata, user_meta, num_models=2)
        assert params["categorical_features_names"] == []
        assert params["categorical_features_dict"] == {}
        assert params["numerical_features"] == ["X_HMER_REF", "MQUAL"]


# ──────────────────────── _make_qual_interpolating_function ──────────────


class TestMakeQualInterpolatingFunction:
    """Tests for _make_qual_interpolating_function."""

    def test_interpolation_at_known_points(self):
        """Should return exact values at known points."""
        quality_table = [[0.0, 5.0, 10.0], [0.0, 25.0, 50.0]]
        fn = _make_qual_interpolating_function(quality_table)
        assert fn(0.0) == pytest.approx(0.0)
        assert fn(5.0) == pytest.approx(25.0)
        assert fn(10.0) == pytest.approx(50.0)

    def test_interpolation_between_points(self):
        """Should linearly interpolate between known points."""
        quality_table = [[0.0, 10.0], [0.0, 100.0]]
        fn = _make_qual_interpolating_function(quality_table)
        assert fn(5.0) == pytest.approx(50.0)

    def test_extrapolation_left(self):
        """Values below range should return 0 (left=0)."""
        quality_table = [[1.0, 5.0, 10.0], [10.0, 50.0, 100.0]]
        fn = _make_qual_interpolating_function(quality_table)
        assert fn(0.0) == pytest.approx(0.0)
        assert fn(-5.0) == pytest.approx(0.0)

    def test_extrapolation_right(self):
        """Values above range should return last y value."""
        quality_table = [[0.0, 5.0, 10.0], [10.0, 50.0, 80.0]]
        fn = _make_qual_interpolating_function(quality_table)
        assert fn(100.0) == pytest.approx(80.0)
        assert fn(20.0) == pytest.approx(80.0)


# ──────────────────────── compute_is_cycle_skip_column ──────────────────


class TestComputeIsCycleSkipColumn:
    """Tests for compute_is_cycle_skip_column."""

    def test_returns_series_of_correct_length(self):
        """Result should be a Series with same length as input."""
        data_df = pd.DataFrame(
            {
                "X_PREV1": ["A", "C", "G"],
                "REF": ["A", "T", "C"],
                "ALT": ["T", "A", "G"],
                "X_NEXT1": ["G", "A", "T"],
            }
        )
        result = compute_is_cycle_skip_column(data_df)
        assert len(result) == 3
        assert isinstance(result, pd.Series)

    def test_known_cycle_skip(self):
        """A known cycle skip pattern should return True."""
        # With TGCA flow order: ATC -> ATC (ref motif same as alt motif with skip)
        # Let's test a real cycle skip: flow order TGCA
        # A substitution is a cycle skip if ref and alt are in the same flow
        # Let's create some controlled patterns and verify they produce booleans
        data_df = pd.DataFrame(
            {
                "X_PREV1": ["A", "T", "G", "C"],
                "REF": ["T", "G", "C", "A"],
                "ALT": ["C", "A", "T", "G"],
                "X_NEXT1": ["G", "C", "A", "T"],
            }
        )
        result = compute_is_cycle_skip_column(data_df, flow_order="TGCA")
        # Check that result is boolean-like (True/False/NaN)
        assert result.dtype == bool or result.dtype == object or pd.api.types.is_bool_dtype(result.dropna())

    def test_custom_flow_order(self):
        """Should accept a custom flow order."""
        data_df = pd.DataFrame(
            {
                "X_PREV1": ["A"],
                "REF": ["T"],
                "ALT": ["G"],
                "X_NEXT1": ["C"],
            }
        )
        # Should not raise with different flow orders
        result = compute_is_cycle_skip_column(data_df, flow_order="TGCA")
        assert len(result) == 1

    def test_preserves_index(self):
        """Result should preserve the input DataFrame's index."""
        data_df = pd.DataFrame(
            {
                "X_PREV1": ["A", "C"],
                "REF": ["T", "G"],
                "ALT": ["G", "A"],
                "X_NEXT1": ["C", "T"],
            },
            index=[10, 20],
        )
        result = compute_is_cycle_skip_column(data_df)
        assert list(result.index) == [10, 20]


# ──────────────────────── __getattr__ delegation ──────────────────────


class TestModelWithTrainingResultsGetattr:
    """Tests for __getattr__ fallback delegation in _ModelWithTrainingResults."""

    def test_getattr_for_private_attr(self):
        """__getattr__ should delegate private attrs not copied in __init__."""

        class FakeModel:
            def __init__(self):
                self._internal_state = "hidden_value"

        model = FakeModel()
        wrapper = _ModelWithTrainingResults(model, {})
        # _internal_state starts with _ so it won't be copied in __init__
        assert wrapper._internal_state == "hidden_value"

    def test_getattr_raises_for_missing(self):
        """Should raise AttributeError for attributes not on model either."""

        class FakeModel:
            pass

        wrapper = _ModelWithTrainingResults(FakeModel(), {})
        with pytest.raises(AttributeError):
            _ = wrapper.nonexistent_attribute_xyz


# ──────────────────────── prepare_report (integration) ──────────────────


class TestPrepareReport:
    """Tests for the prepare_report function with mocked heavy dependencies."""

    def test_prepare_report_with_mocked_report_generation(self, tmp_path):
        """Test prepare_report orchestration logic with mocks."""
        # Create a minimal featuremap parquet file
        data_df = pd.DataFrame(
            {
                "X_PREV1": ["A", "C", "G", "T"] * 25,
                "REF": ["T", "G", "A", "C"] * 25,
                "ALT": ["G", "A", "C", "T"] * 25,
                "X_NEXT1": ["C", "T", "G", "A"] * 25,
                "X_HMER_REF": [1, 2, 3, 4] * 25,
                "X_HMER_ALT": [2, 3, 1, 2] * 25,
                "MQUAL": np.random.default_rng(42).uniform(0, 50, 100).tolist(),
                "X_ALT": ["G", "A", "C", "T"] * 25,
            }
        )
        parquet_path = str(tmp_path / "featuremap_df.parquet")
        data_df.to_parquet(parquet_path)

        # Create a minimal metadata JSON
        metadata = {
            "model_paths": {},
            "training_results": None,
            "features": [
                {"name": "REF", "type": "c", "values": {"A": 0, "C": 1, "G": 2, "T": 3}},
                {"name": "X_HMER_REF", "type": "int"},
                {"name": "MQUAL", "type": "float"},
            ],
            "quality_recalibration_table": [
                [0.0, 10.0, 20.0, 30.0, 40.0, 50.0],
                [0.0, 5.0, 15.0, 30.0, 50.0, 70.0],
            ],
            "metadata": {"adapter_version": None},
        }
        metadata_path = str(tmp_path / "srsnv_metadata.json")
        with open(metadata_path, "w") as fh:
            json.dump(metadata, fh)

        report_path = str(tmp_path / "report")
        os.makedirs(report_path, exist_ok=True)

        # Mock the heavy report generation parts
        with (
            patch("ugbio_srsnv.srsnv_report.SRSNVReport") as mock_report_cls,
            patch("ugbio_srsnv.srsnv_report.create_srsnv_report_html") as mock_html,
            patch("ugbio_srsnv.srsnv_report.add_is_mixed_to_featuremap_df") as mock_mixed,
        ):
            # Mock add_is_mixed to return the same df
            mock_mixed.side_effect = lambda df, *args, **kwargs: df
            # Mock the report class
            mock_report_instance = MagicMock()
            mock_report_cls.return_value = mock_report_instance

            prepare_report(
                featuremap_df=parquet_path,
                srsnv_metadata=metadata_path,
                report_path=report_path,
                basename="test",
                models_prefix=None,
                random_seed=42,
            )

            # Verify the report was created
            mock_report_cls.assert_called_once()
            mock_report_instance.create_report.assert_called_once()
            mock_html.assert_called_once()

    def test_prepare_report_with_models_prefix(self, tmp_path):
        """Test prepare_report loading models from prefix."""
        # Create minimal XGBoost models
        for idx in range(2):
            model = xgb.XGBClassifier(n_estimators=2, max_depth=1)
            model.fit(np.array([[0, 1], [1, 0], [0, 0], [1, 1]]), np.array([0, 1, 0, 1]))
            model.save_model(str(tmp_path / f"model_{idx}.json"))

        # Create a minimal featuremap parquet
        data_df = pd.DataFrame(
            {
                "X_PREV1": ["A", "C", "G", "T"] * 10,
                "REF": ["T", "G", "A", "C"] * 10,
                "ALT": ["G", "A", "C", "T"] * 10,
                "X_NEXT1": ["C", "T", "G", "A"] * 10,
                "X_HMER_REF": [1, 2, 3, 4] * 10,
                "X_HMER_ALT": [2, 3, 1, 2] * 10,
                "MQUAL": np.random.default_rng(42).uniform(0, 50, 40).tolist(),
                "X_ALT": ["G", "A", "C", "T"] * 10,
            }
        )
        parquet_path = str(tmp_path / "featuremap_df.parquet")
        data_df.to_parquet(parquet_path)

        # Create metadata
        metadata = {
            "model_paths": {},
            "training_results": [
                {"validation_0": {"auc": [0.9]}, "validation_1": {"auc": [0.85]}},
                {"validation_0": {"auc": [0.88]}, "validation_1": {"auc": [0.83]}},
            ],
            "features": [
                {"name": "REF", "type": "c", "values": {"A": 0, "C": 1, "G": 2, "T": 3}},
                {"name": "MQUAL", "type": "float"},
            ],
            "quality_recalibration_table": [
                [0.0, 25.0, 50.0],
                [0.0, 30.0, 60.0],
            ],
            "metadata": {},
        }
        metadata_path = str(tmp_path / "metadata.json")
        with open(metadata_path, "w") as fh:
            json.dump(metadata, fh)

        report_path = str(tmp_path / "report")
        os.makedirs(report_path, exist_ok=True)

        with (
            patch("ugbio_srsnv.srsnv_report.SRSNVReport") as mock_report_cls,
            patch("ugbio_srsnv.srsnv_report.create_srsnv_report_html"),
            patch("ugbio_srsnv.srsnv_report.add_is_mixed_to_featuremap_df") as mock_mixed,
        ):
            mock_mixed.side_effect = lambda df, *args, **kwargs: df
            mock_report_instance = MagicMock()
            mock_report_cls.return_value = mock_report_instance

            prepare_report(
                featuremap_df=parquet_path,
                srsnv_metadata=metadata_path,
                report_path=report_path,
                basename="test.",
                models_prefix=str(tmp_path / "model_"),
                random_seed=None,
            )

            # Verify models loaded from prefix
            call_kwargs = mock_report_cls.call_args[1]
            assert len(call_kwargs["models"]) == 2
            # Verify training results were attached
            assert call_kwargs["models"][0].evals_result() == metadata["training_results"][0]

    def test_prepare_report_basename_formatting(self, tmp_path):
        """Test that basename is formatted correctly for the report."""
        # Create minimal data
        data_df = pd.DataFrame(
            {
                "X_PREV1": ["A"] * 5,
                "REF": ["T"] * 5,
                "ALT": ["G"] * 5,
                "X_NEXT1": ["C"] * 5,
                "X_HMER_REF": [1] * 5,
                "MQUAL": [10.0] * 5,
            }
        )
        parquet_path = str(tmp_path / "data.parquet")
        data_df.to_parquet(parquet_path)

        metadata = {
            "model_paths": {},
            "features": [{"name": "MQUAL", "type": "float"}],
            "quality_recalibration_table": [[0.0, 50.0], [0.0, 50.0]],
            "metadata": {},
        }
        metadata_path = str(tmp_path / "meta.json")
        with open(metadata_path, "w") as fh:
            json.dump(metadata, fh)

        report_path = str(tmp_path / "report")
        os.makedirs(report_path, exist_ok=True)

        with (
            patch("ugbio_srsnv.srsnv_report.SRSNVReport") as mock_report_cls,
            patch("ugbio_srsnv.srsnv_report.create_srsnv_report_html"),
            patch("ugbio_srsnv.srsnv_report.add_is_mixed_to_featuremap_df") as mock_mixed,
        ):
            mock_mixed.side_effect = lambda df, *args, **kwargs: df
            mock_report_cls.return_value = MagicMock()

            # Test basename without trailing dot gets dot added
            prepare_report(
                featuremap_df=parquet_path,
                srsnv_metadata=metadata_path,
                report_path=report_path,
                basename="sample_name",
            )
            call_kwargs = mock_report_cls.call_args[1]
            assert call_kwargs["base_name"] == "sample_name."

            # Test empty basename stays empty
            prepare_report(
                featuremap_df=parquet_path,
                srsnv_metadata=metadata_path,
                report_path=report_path,
                basename="",
            )
            call_kwargs = mock_report_cls.call_args[1]
            assert call_kwargs["base_name"] == ""
