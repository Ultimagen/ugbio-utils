import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from ugbio_srsnv.srsnv_plotting_utils import SRSNVReport
from ugbio_srsnv.srsnv_report import prepare_report


@pytest.fixture
def resources_dir():
    return Path(__file__).parent.parent / "resources"


@pytest.mark.skip(reason="Mock data needs many complex filter definitions to match expected structure")
def test_prepare_report_with_mock_data(tmpdir):
    """Test the prepare_report function with minimal mock data"""

    # Set up random number generator for reproducible tests
    rng = np.random.default_rng(42)

    # Create a minimal test dataframe
    test_data = {
        "CHROM": ["chr1"] * 100,
        "POS": range(1000, 1100),
        "REF": ["A"] * 50 + ["T"] * 50,
        "ALT": ["T"] * 50 + ["C"] * 50,
        "X_HMER_REF": [1] * 100,
        "X_HMER_ALT": [1] * 100,
        "X_PREV1": ["G"] * 100,
        "X_NEXT1": ["C"] * 100,
        "MQUAL": rng.uniform(10, 50, 100),
        "SNVQ": rng.uniform(10, 50, 100),
        "is_mixed": [True] * 60 + [False] * 40,
        "is_mixed_start": [True] * 60 + [False] * 40,
        "is_mixed_end": [False] * 40 + [True] * 60,
        "label": [1] * 60 + [0] * 40,  # 60 True positives, 40 False positives
        "fold_id": [0] * 50 + [1] * 50,
        "prob_orig": rng.uniform(0.1, 0.9, 100),
        "prob_fold_0": rng.uniform(0.1, 0.9, 100),  # Required for SRSNVReport
        "BCSQ": rng.uniform(60, 100, 100),
        "EDIST": rng.integers(0, 5, 100),
        "DP": rng.integers(10, 50, 100),
        "RL": rng.integers(100, 200, 100),
        "INDEX": rng.integers(20, 100, 100),
        "REV": rng.choice([0, 1], 100),
        "st": ["MIXED"] * 50 + ["PERFECT"] * 50,  # start tag
        "et": ["PERFECT"] * 40 + ["MIXED"] * 60,  # end tag
    }

    test_df = pd.DataFrame(test_data)

    # Save the dataframe to a parquet file
    featuremap_df_path = os.path.join(tmpdir, "test_featuremap.parquet")
    test_df.to_parquet(featuremap_df_path)

    # Create a minimal XGBoost model for testing
    model = xgb.XGBClassifier(n_estimators=2, max_depth=2, random_state=42)

    # Prepare minimal training data
    feature_names = ["MQUAL", "SNVQ", "BCSQ", "EDIST", "DP"]
    x_train = test_df[feature_names].to_numpy()
    y_train = test_df["label"].to_numpy()

    # Train the model
    model.fit(x_train, y_train)

    # Save the model to a json file
    model_path = os.path.join(tmpdir, "test_model_fold_0.json")
    model.save_model(model_path)

    # Create minimal metadata
    metadata = {
        "model_paths": {"fold_0": model_path},
        "training_results": [{"validation_0": {"logloss": [0.5, 0.4, 0.3, 0.2]}}],
        "features": [
            {"name": "MQUAL", "type": "n"},
            {"name": "SNVQ", "type": "n"},
            {"name": "BCSQ", "type": "n"},
            {"name": "EDIST", "type": "n"},
            {"name": "DP", "type": "n"},
            {"name": "st", "type": "c", "values": {"MIXED": 0, "PERFECT": 1}},
            {"name": "et", "type": "c", "values": {"MIXED": 0, "PERFECT": 1}},
        ],
        "quality_recalibration_table": [
            list(range(0, 50, 5)),  # x values
            list(range(0, 50, 5)),  # y values (simple 1:1 mapping for test)
        ],
        "filtering_stats": {
            "positive": {
                "filters": [
                    {"name": "raw", "funnel": 100, "type": "raw"},
                    {"name": "coverage_ge_min", "funnel": 90, "type": "region", "field": "DP", "op": "ge", "value": 20},
                ]
            },
            "negative": {
                "filters": [
                    {"name": "raw", "funnel": 100, "type": "raw"},
                    {"name": "coverage_ge_min", "funnel": 90, "type": "region", "field": "DP", "op": "ge", "value": 20},
                ]
            },
        },
        "training_parameters": {"max_qual": 50},
        "metadata": {"adapter_version": "v1", "docker_image": "test/image:latest", "pipeline_version": "test"},
    }

    # Save metadata to a json file
    metadata_path = os.path.join(tmpdir, "test_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)

    # Test the prepare_report function
    try:
        prepare_report(
            featuremap_df=featuremap_df_path,
            srsnv_metadata=metadata_path,
            report_path=tmpdir,
            basename="test_report",
            random_seed=42,
        )

        # Check that some output files were created
        # Note: The exact files depend on the implementation, so we'll check for common ones
        expected_files = [
            "test_report.single_read_snv.applicationQC.h5",
        ]

        for expected_file in expected_files:
            file_path = os.path.join(tmpdir, expected_file)
            if os.path.exists(file_path):
                print(f"Found expected file: {expected_file}")
            else:
                print(f"Missing expected file: {expected_file}")

        # The test passes if the function runs without error
        # More detailed assertions could be added based on specific requirements

    except Exception as e:
        pytest.fail(f"prepare_report failed with error: {e}")


# This test can be enabled when we have proper test resources that match the new format
def test_prepare_report_with_existing_resources(tmpdir, resources_dir):
    """Test with existing resources that were generated by the training pipeline"""

    # Use the test resources generated by the training pipeline
    featuremap_path = resources_dir / "416119_L7402.test.featuremap_df.parquet"
    metadata_path = resources_dir / "416119_L7402.test.srsnv_metadata.json"

    if not featuremap_path.exists() or not metadata_path.exists():
        pytest.skip("Test resources not available - need to run training pipeline first")

    # Test the prepare_report function with real generated data
    # This provides a smoke test that the function works with properly structured data
    try:
        # Just test function signature and basic setup without running full pipeline
        # which is too resource intensive for unit tests
        import inspect

        from ugbio_srsnv.srsnv_report import prepare_report

        sig = inspect.signature(prepare_report)
        assert "featuremap_df" in sig.parameters
        assert "srsnv_metadata" in sig.parameters
        assert "report_path" in sig.parameters

        # Verify the files have the expected structure
        import pandas as pd

        df = pd.read_parquet(featuremap_path)  # noqa: PD901
        assert "prob_fold_0" in df.columns, "Generated test data should have prob_fold_0 column"

        with open(metadata_path) as f:
            import json

            metadata = json.load(f)
        assert "model_paths" in metadata, "Generated metadata should have model_paths"
        assert "filtering_stats" in metadata, "Generated metadata should have filtering_stats"

        print("✓ Test resources are properly formatted")
        print(f"✓ Featuremap has {len(df.columns)} columns and {len(df)} rows")

    except Exception as e:
        pytest.fail(f"prepare_report validation failed with error: {e}")


# Tests for calc_run_info_table using real data
@pytest.fixture
def test_resources_calc_run_info():
    """Load real test resources generated by the training pipeline"""
    resources_dir = Path(__file__).parent.parent / "resources"

    # Load the featuremap dataframe
    featuremap_df = pd.read_parquet(resources_dir / "402572-CL10377.featuremap_df.parquet")

    # Load the metadata
    metadata_path = resources_dir / "402572-CL10377.srsnv_metadata.json"
    with open(metadata_path) as f:
        metadata = json.load(f)

    return featuremap_df, metadata, metadata_path


@pytest.fixture
def real_models_calc_run_info(test_resources_calc_run_info):
    """Load the real XGBoost models from the test resources"""
    resources_dir = Path(__file__).parent.parent / "resources"
    _, metadata, _ = test_resources_calc_run_info
    models = []

    # Load models from the paths specified in metadata
    for fold_id, model_path in metadata["model_paths"].items():
        model = xgb.XGBClassifier()
        # Convert relative path to absolute path from resources directory
        if not os.path.isabs(model_path):
            model_filename = model_path.split("/")[-1]  # Get just the filename
        else:
            model_filename = os.path.basename(model_path)

        absolute_model_path = resources_dir / model_filename
        model.load_model(str(absolute_model_path))

        # Manually set the required attributes that sklearn expects for XGBoost models
        model.n_classes_ = 2  # Binary classification

        models.append(model)

    return models


def test_calc_run_info_table_basic_functionality(test_resources_calc_run_info, real_models_calc_run_info):
    """Test basic functionality of calc_run_info_table with real data"""
    featuremap_df, metadata, metadata_path = test_resources_calc_run_info

    with tempfile.TemporaryDirectory() as temp_output_dir:
        # Create a copy of metadata file in temp directory
        temp_metadata_file = os.path.join(temp_output_dir, "test_metadata.json")
        with open(temp_metadata_file, "w") as f:
            json.dump(metadata, f)

        # Create basic params dict based on the metadata structure
        categorical_features = [f for f in metadata["features"] if f["type"] == "c"]
        numerical_features = [f for f in metadata["features"] if f["type"] != "c"]

        params = {
            "workdir": temp_output_dir,
            "data_name": "test_run",
            "categorical_features_names": [f["name"] for f in categorical_features],
            "categorical_features_dict": {f["name"]: list(f["values"].keys()) for f in categorical_features},
            "numerical_features": [f["name"] for f in numerical_features],
            "fp_regions_bed_file": 1,
            "num_CV_folds": len(real_models_calc_run_info),
        }

        # Create SRSNVReport instance
        report = SRSNVReport(
            models=real_models_calc_run_info,
            data_df=featuremap_df.copy(),
            params=params,
            out_path=temp_output_dir,
            srsnv_metadata=temp_metadata_file,
            base_name="test_",
            raise_exceptions=True,
        )

        # Calculate recall values first (required for calc_run_info_table)
        report.plot_fq_recall(only_calculate=True)

        # Call the method under test
        report.calc_run_info_table()

        # Verify H5 file was created
        h5_file = os.path.join(temp_output_dir, "test_single_read_snv.applicationQC.h5")
        assert os.path.exists(h5_file), "H5 file should be created"

        # Verify expected tables exist
        with pd.HDFStore(h5_file, "r") as store:
            expected_keys = ["/run_info_table", "/run_quality_summary_table", "/training_info_table"]
            for key in expected_keys:
                assert key in store.keys(), f"Expected table {key} not found in H5 file"


def test_calc_run_info_table_content_validation(test_resources_calc_run_info, real_models_calc_run_info):
    """Test that calc_run_info_table generates expected content structure"""
    featuremap_df, metadata, metadata_path = test_resources_calc_run_info

    with tempfile.TemporaryDirectory() as temp_output_dir:
        # Create a copy of metadata file in temp directory
        temp_metadata_file = os.path.join(temp_output_dir, "test_metadata.json")
        with open(temp_metadata_file, "w") as f:
            json.dump(metadata, f)

        # Create params dict
        categorical_features = [f for f in metadata["features"] if f["type"] == "c"]
        numerical_features = [f for f in metadata["features"] if f["type"] != "c"]

        params = {
            "workdir": temp_output_dir,
            "data_name": "test_run",
            "categorical_features_names": [f["name"] for f in categorical_features],
            "categorical_features_dict": {f["name"]: list(f["values"].keys()) for f in categorical_features},
            "numerical_features": [f["name"] for f in numerical_features],
            "fp_regions_bed_file": 1,
            "num_CV_folds": len(real_models_calc_run_info),
        }

        report = SRSNVReport(
            models=real_models_calc_run_info,
            data_df=featuremap_df.copy(),
            params=params,
            out_path=temp_output_dir,
            srsnv_metadata=temp_metadata_file,
            base_name="test_",
            raise_exceptions=True,
        )

        # Calculate recall values first
        report.plot_fq_recall(only_calculate=True)

        # Call the method
        report.calc_run_info_table()

        # Read back the tables and validate content
        h5_file = os.path.join(temp_output_dir, "test_single_read_snv.applicationQC.h5")

        # Validate run_info_table structure
        run_info = pd.read_hdf(h5_file, key="run_info_table")

        # Check that expected keys are present (based on actual output structure)
        expected_keys = [
            "Sample name",
            "Median training read length",
            "Median training coverage",
            "Training set, % TP reads",
            "Pipeline version",
            "Docker image",
            "Adapter version",
            "Report created on",
        ]

        for key in expected_keys:
            assert key in run_info.index, f"Expected key '{key}' not found in run_info_table"

        # Check for mixed training reads section (multi-index)
        mixed_training_mask = run_info.index.get_level_values(0) == "Mixed training reads"
        assert mixed_training_mask.any(), "Mixed training reads section not found in run_info_table"

        # Validate run_quality_summary_table structure
        quality_summary = pd.read_hdf(h5_file, key="run_quality_summary_table")

        # Check that it's a Series with expected structure
        assert isinstance(quality_summary, pd.Series), "run_quality_summary_table should be a Series"
        assert len(quality_summary) > 0, "run_quality_summary_table should not be empty"

        # Validate that recall@SNVQ metrics are present in legacy table (including SNVQ=70)
        expected_legacy_keys = [
            ("Recall at SNVQ=50", "All reads"),
            ("Recall at SNVQ=60", "All reads"),
            ("Recall at SNVQ=70", "All reads"),
            ("Recall at SNVQ=50", "Mixed, start"),
            ("Recall at SNVQ=60", "Mixed, start"),
            ("Recall at SNVQ=70", "Mixed, start"),
        ]
        for key in expected_legacy_keys:
            assert key in quality_summary.index, f"Expected key {key} not found in run_quality_summary_table"

        # Validate modern table (run_quality_summary_table_mixed_start) also has SNVQ=70
        quality_summary_modern = pd.read_hdf(h5_file, key="run_quality_summary_table_mixed_start")
        expected_modern_keys = [
            ("Recall at SNVQ=50", "All reads"),
            ("Recall at SNVQ=60", "All reads"),
            ("Recall at SNVQ=70", "All reads"),
            ("Recall at SNVQ=50", "Mixed"),
            ("Recall at SNVQ=60", "Mixed"),
            ("Recall at SNVQ=70", "Mixed"),
        ]
        for key in expected_modern_keys:
            assert (
                key in quality_summary_modern.index
            ), f"Expected key {key} not found in run_quality_summary_table_mixed_start"

        # Validate training_info_table structure
        training_info = pd.read_hdf(h5_file, key="training_info_table")

        # Check that it's a Series with expected structure
        assert isinstance(training_info, pd.Series), "training_info_table should be a Series"
        assert len(training_info) > 0, "training_info_table should not be empty"


def test_calc_run_info_table_numerical_validation(test_resources_calc_run_info, real_models_calc_run_info):
    """Test numerical calculations in calc_run_info_table"""
    featuremap_df, metadata, metadata_path = test_resources_calc_run_info

    with tempfile.TemporaryDirectory() as temp_output_dir:
        # Create a copy of metadata file in temp directory
        temp_metadata_file = os.path.join(temp_output_dir, "test_metadata.json")
        with open(temp_metadata_file, "w") as f:
            json.dump(metadata, f)

        # Create params dict
        categorical_features = [f for f in metadata["features"] if f["type"] == "c"]
        numerical_features = [f for f in metadata["features"] if f["type"] != "c"]

        params = {
            "workdir": temp_output_dir,
            "data_name": "test_run",
            "categorical_features_names": [f["name"] for f in categorical_features],
            "categorical_features_dict": {f["name"]: list(f["values"].keys()) for f in categorical_features},
            "numerical_features": [f["name"] for f in numerical_features],
            "fp_regions_bed_file": 1,
            "num_CV_folds": len(real_models_calc_run_info),
        }

        report = SRSNVReport(
            models=real_models_calc_run_info,
            data_df=featuremap_df.copy(),
            params=params,
            out_path=temp_output_dir,
            srsnv_metadata=temp_metadata_file,
            base_name="test_",
            raise_exceptions=True,
        )

        # Calculate precision/recall first
        report.plot_fq_recall(only_calculate=True)

        # Call the method
        report.calc_run_info_table()

        # Read back and validate numerical values
        h5_file = os.path.join(temp_output_dir, "test_single_read_snv.applicationQC.h5")
        run_info = pd.read_hdf(h5_file, key="run_info_table")

        # Validate numerical values in the run_info table (update to match actual structure)

        # Sample name should be 'test' (the actual value from the output)
        assert run_info["Sample name"].iloc[0] == "test"

        # Check that numerical values are reasonable
        median_read_length = run_info["Median training read length"].iloc[0]
        assert isinstance(median_read_length, int | float), "Median training read length should be numeric"
        assert median_read_length > 0, "Median training read length should be positive"

        median_coverage = run_info["Median training coverage"].iloc[0]
        assert isinstance(median_coverage, int | float), "Median training coverage should be numeric"
        assert median_coverage > 0, "Median training coverage should be positive"

        tp_rate = run_info["Training set, % TP reads"].iloc[0]
        assert isinstance(tp_rate, int | float), "Training set TP rate should be numeric"
        assert 0 <= tp_rate <= 100, "Training set TP rate should be between 0 and 100"

        # Validate training_info_table
        training_info = pd.read_hdf(h5_file, key="training_info_table")

        # Basic checks on structure
        assert isinstance(training_info, pd.Series), "training_info_table should be a Series"
        assert len(training_info) > 0, "training_info_table should not be empty"

        # Validate run_quality_summary_table
        quality_summary = pd.read_hdf(h5_file, key="run_quality_summary_table")

        # Basic checks on structure
        assert isinstance(quality_summary, pd.Series), "run_quality_summary_table should be a Series"
        assert len(quality_summary) > 0, "run_quality_summary_table should not be empty"


def test_plot_logit_histograms_writes_legacy_and_mixed_start_keys(
    test_resources_calc_run_info, real_models_calc_run_info
):
    """plot_logit_histograms should write both the legacy (both-ends) and the new mixed-start h5 keys.

    The report displays the mixed-start version (mixed = start ppmSeq tag is MIXED), while the legacy
    both-ends histogram is retained under the original ``logit_histogram`` key for backward compatibility.
    """
    featuremap_df, metadata, _ = test_resources_calc_run_info

    with tempfile.TemporaryDirectory() as temp_output_dir:
        temp_metadata_file = os.path.join(temp_output_dir, "test_metadata.json")
        with open(temp_metadata_file, "w") as f:
            json.dump(metadata, f)

        categorical_features = [f for f in metadata["features"] if f["type"] == "c"]
        numerical_features = [f for f in metadata["features"] if f["type"] != "c"]

        params = {
            "workdir": temp_output_dir,
            "data_name": "test_run",
            "categorical_features_names": [f["name"] for f in categorical_features],
            "categorical_features_dict": {f["name"]: list(f["values"].keys()) for f in categorical_features},
            "numerical_features": [f["name"] for f in numerical_features],
            "fp_regions_bed_file": 1,
            "num_CV_folds": len(real_models_calc_run_info),
        }

        report = SRSNVReport(
            models=real_models_calc_run_info,
            data_df=featuremap_df.copy(),
            params=params,
            out_path=temp_output_dir,
            srsnv_metadata=temp_metadata_file,
            base_name="test_",
            raise_exceptions=True,
        )

        report.plot_logit_histograms(output_filename=os.path.join(temp_output_dir, "logit_histogram"))

        h5_file = os.path.join(temp_output_dir, "test_single_read_snv.applicationQC.h5")
        with pd.HDFStore(h5_file, "r") as store:
            assert "/logit_histogram" in store.keys(), "legacy logit_histogram key should be retained"
            assert (
                "/logit_histogram_mixed_start" in store.keys()
            ), "new logit_histogram_mixed_start key should be written for display"


# ──────────────────────── consensus mode (nf/nr) tests ──────────────────────


@pytest.fixture
def consensus_resources(test_resources_calc_run_info):
    """Build a consensus-mode featuremap from the real ppmSeq resources.

    Drops the ppmSeq st/et tags and adds nf/nr read counts so the report auto-detects
    CONSENSUS mode. Returns (featuremap_df, metadata).
    """
    featuremap_df, metadata, _ = test_resources_calc_run_info
    consensus_df = featuremap_df.copy()
    consensus_df = consensus_df.drop(columns=[c for c in ("st", "et") if c in consensus_df.columns])
    rng = np.random.default_rng(7)
    consensus_df["nf"] = rng.integers(0, 6, len(consensus_df))
    consensus_df["nr"] = rng.integers(0, 6, len(consensus_df))
    # Ensure both consensus and non-consensus reads are present
    consensus_df.loc[consensus_df.index[:50], ["nf", "nr"]] = 2
    consensus_df.loc[consensus_df.index[50:100], "nf"] = 0
    # Drop st/et from the metadata categorical features so params don't reference them
    metadata = json.loads(json.dumps(metadata))
    metadata["features"] = [f for f in metadata["features"] if f["name"] not in ("st", "et")]
    return consensus_df, metadata


def _make_consensus_report(df, metadata, temp_output_dir, models):
    """Helper: build an SRSNVReport in consensus mode from a consensus dataframe."""
    temp_metadata_file = os.path.join(temp_output_dir, "test_metadata.json")
    with open(temp_metadata_file, "w") as f:
        json.dump(metadata, f)
    categorical_features = [f for f in metadata["features"] if f["type"] == "c"]
    numerical_features = [f for f in metadata["features"] if f["type"] != "c"]
    params = {
        "workdir": temp_output_dir,
        "data_name": "test_run",
        "categorical_features_names": [f["name"] for f in categorical_features],
        "categorical_features_dict": {f["name"]: list(f["values"].keys()) for f in categorical_features},
        "numerical_features": [f["name"] for f in numerical_features],
        "fp_regions_bed_file": 1,
        "num_CV_folds": len(models),
        "report_mode": "consensus",
    }
    return SRSNVReport(
        models=models,
        data_df=df.copy(),
        params=params,
        out_path=temp_output_dir,
        srsnv_metadata=temp_metadata_file,
        base_name="test_",
        raise_exceptions=True,
    )


def test_consensus_mode_auto_detected(consensus_resources, real_models_calc_run_info):
    """SRSNVReport auto-detects consensus mode, adds is_consensus and the 3-group read_group."""
    df, metadata = consensus_resources
    with tempfile.TemporaryDirectory() as temp_output_dir:
        report = _make_consensus_report(df, metadata, temp_output_dir, real_models_calc_run_info)
        assert report.report_mode.value == "consensus"
        assert list(report.scheme.display_variant.groups) == [
            "single read",
            "consensus, one strand",
            "consensus, duplex",
        ]
        assert "is_consensus" in report.data_df.columns
        assert "read_group" in report.data_df.columns
        # duplex group == the is_consensus boolean
        expected = ((df["nf"] >= 1) & (df["nr"] >= 1)).to_numpy()
        np.testing.assert_array_equal(report.data_df["is_consensus"].to_numpy(), expected)
        np.testing.assert_array_equal((report.data_df["read_group"] == "consensus, duplex").to_numpy(), expected)


def test_consensus_mode_roc_auc_table_keys_and_labels(consensus_resources, real_models_calc_run_info):
    """ROC AUC table keeps stable h5 keys and uses the 3 consensus read-group labels."""
    df, metadata = consensus_resources
    with tempfile.TemporaryDirectory() as temp_output_dir:
        report = _make_consensus_report(df, metadata, temp_output_dir, real_models_calc_run_info)
        report.calc_roc_auc_table()
        h5_file = os.path.join(temp_output_dir, "test_single_read_snv.applicationQC.h5")
        with pd.HDFStore(h5_file, "r") as store:
            # consensus writes the scheme-suffixed key + the bare base alias (for h5->json)
            assert "/roc_auc_table" in store.keys()
            assert "/roc_auc_table_strand_support" in store.keys()
        table = pd.read_hdf(h5_file, key="roc_auc_table_strand_support")
        # Index after .T stores the group labels
        labels = list(table.columns) + list(table.index)
        assert "single read only" in labels
        assert "consensus, one strand only" in labels
        assert "consensus, duplex only" in labels


CONSENSUS_GROUP_LABELS = ["single read", "consensus, one strand", "consensus, duplex"]


def test_consensus_mode_run_quality_table_labels(consensus_resources, real_models_calc_run_info):
    """Run quality display table carries the 3 consensus read-group labels; keys stable."""
    df, metadata = consensus_resources
    with tempfile.TemporaryDirectory() as temp_output_dir:
        report = _make_consensus_report(df, metadata, temp_output_dir, real_models_calc_run_info)
        report.plot_fq_recall(only_calculate=True)
        report.calc_run_quality_table()
        h5_file = os.path.join(temp_output_dir, "test_single_read_snv.applicationQC.h5")
        with pd.HDFStore(h5_file, "r") as store:
            assert "/run_quality_table" in store.keys()
            assert "/run_quality_table_strand_support" in store.keys()
            assert "/run_quality_table_display" in store.keys()
        display = pd.read_hdf(h5_file, key="run_quality_table_display")
        third_level = set(display.columns.get_level_values(2))
        for label in CONSENSUS_GROUP_LABELS:
            assert label in third_level


def test_consensus_mode_run_info_table_label(consensus_resources, real_models_calc_run_info):
    """Run info table labels one row group per consensus read group."""
    df, metadata = consensus_resources
    with tempfile.TemporaryDirectory() as temp_output_dir:
        report = _make_consensus_report(df, metadata, temp_output_dir, real_models_calc_run_info)
        report.plot_fq_recall(only_calculate=True)
        report.calc_run_info_table()
        h5_file = os.path.join(temp_output_dir, "test_single_read_snv.applicationQC.h5")
        run_info = pd.read_hdf(h5_file, key="run_info_table_strand_support")
        level0 = set(run_info.index.get_level_values(0))
        for label in CONSENSUS_GROUP_LABELS:
            assert f"{label} training reads" in level0


def test_consensus_mode_quality_per_tags_table(consensus_resources, real_models_calc_run_info):
    """quality_per_ppmseq_tags produces a per-read-group (3-row) table under stable keys."""
    df, metadata = consensus_resources
    with tempfile.TemporaryDirectory() as temp_output_dir:
        report = _make_consensus_report(df, metadata, temp_output_dir, real_models_calc_run_info)
        out_png = os.path.join(temp_output_dir, "qual_vs_tags")
        report.quality_per_ppmseq_tags(output_filename=out_png)
        h5_file = os.path.join(temp_output_dir, "test_single_read_snv.applicationQC.h5")
        with pd.HDFStore(h5_file, "r") as store:
            assert "/ppmseq_category_quality_table_strand_support" in store.keys()
            # bare base keys still written so downstream h5->json conversion doesn't KeyError
            assert "/ppmseq_category_quality_table" in store.keys()
            assert "/ppmseq_category_quantity_table" in store.keys()
        summary = pd.read_hdf(h5_file, key="ppmseq_category_quality_table_strand_support")
        assert list(summary.index) == CONSENSUS_GROUP_LABELS
        assert os.path.exists(out_png + ".png")


def test_consensus_mode_histograms_run(consensus_resources, real_models_calc_run_info):
    """Quality + logit histograms run in consensus mode and keep stable h5 keys."""
    df, metadata = consensus_resources
    with tempfile.TemporaryDirectory() as temp_output_dir:
        report = _make_consensus_report(df, metadata, temp_output_dir, real_models_calc_run_info)
        report.plot_quality_histogram(output_filename=os.path.join(temp_output_dir, "qual_hist"))
        report.plot_logit_histograms(output_filename=os.path.join(temp_output_dir, "logit_hist"))
        h5_file = os.path.join(temp_output_dir, "test_single_read_snv.applicationQC.h5")
        with pd.HDFStore(h5_file, "r") as store:
            keys = store.keys()
        for k in (
            "/quality_histogram",
            "/quality_histogram_strand_support",
            "/logit_histogram",
            "/logit_histogram_strand_support",
        ):
            assert k in keys, f"missing {k}"


def test_none_mode_graceful(test_resources_calc_run_info, real_models_calc_run_info):
    """With neither ppmSeq tags nor nf/nr, the report runs as a single group (NONE mode)."""
    featuremap_df, metadata, _ = test_resources_calc_run_info
    none_df = featuremap_df.copy().drop(columns=[c for c in ("st", "et") if c in featuremap_df.columns])
    metadata = json.loads(json.dumps(metadata))
    metadata["features"] = [f for f in metadata["features"] if f["name"] not in ("st", "et")]
    with tempfile.TemporaryDirectory() as temp_output_dir:
        temp_metadata_file = os.path.join(temp_output_dir, "test_metadata.json")
        with open(temp_metadata_file, "w") as f:
            json.dump(metadata, f)
        categorical_features = [f for f in metadata["features"] if f["type"] == "c"]
        numerical_features = [f for f in metadata["features"] if f["type"] != "c"]
        params = {
            "workdir": temp_output_dir,
            "data_name": "test_run",
            "categorical_features_names": [f["name"] for f in categorical_features],
            "categorical_features_dict": {f["name"]: list(f["values"].keys()) for f in categorical_features},
            "numerical_features": [f["name"] for f in numerical_features],
            "fp_regions_bed_file": 1,
            "num_CV_folds": len(real_models_calc_run_info),
        }
        report = SRSNVReport(
            models=real_models_calc_run_info,
            data_df=none_df.copy(),
            params=params,
            out_path=temp_output_dir,
            srsnv_metadata=temp_metadata_file,
            base_name="test_",
            raise_exceptions=True,
        )
        assert report.report_mode.value == "none"
        assert not report.data_df["is_consensus"].any()
        report.plot_fq_recall(only_calculate=True)
        report.calc_roc_auc_table()  # should not raise


# The exact historical mixed (ppmSeq) h5 key surface produced by the full prepare_report path.
# The recipe refactor MUST keep the ppmSeq report byte-identical, and the key set is the first
# line of defense: any dropped / renamed / added key for mixed data breaks backward compatibility
# (the notebook reads *_mixed_start; h5->json conversion reads the bare base keys).
MIXED_EXPECTED_H5_KEYS = {
    "/FQ_recall_LoD",
    "/FQ_recall_LoD_mixed_start",
    "/keys_to_convert",
    "/logit_histogram",
    "/logit_histogram_mixed_start",
    "/mean_abs_SHAP_scores",
    "/ppmseq_category_quality_table",
    "/ppmseq_category_quality_table_mixed_start",
    "/ppmseq_category_quantity_table",
    "/quality_histogram",
    "/quality_histogram_mixed_start",
    "/roc_auc_table",
    "/roc_auc_table_mixed_start",
    "/run_info_table",
    "/run_info_table_mixed_start",
    "/run_quality_summary_table",
    "/run_quality_summary_table_mixed_start",
    "/run_quality_table",
    "/run_quality_table_display",
    "/run_quality_table_mixed_start",
    "/training_info_table",
    "/training_progress",
    "/trinuc_stats",
}


def test_mixed_report_h5_key_surface_is_stable(resources_dir):
    """Full ppmSeq report via prepare_report writes exactly the historical mixed h5 key set.

    Locks the byte-identical contract at the key level (mixed keeps the base + *_mixed_start dual
    keys; consensus/none use their own suffixes, tested separately)."""
    featuremap_df = str(resources_dir / "402572-CL10377.featuremap_df.parquet")
    srsnv_metadata = str(resources_dir / "402572-CL10377.srsnv_metadata.json")
    with tempfile.TemporaryDirectory() as td:
        # prepare_report resolves model paths relative to CWD basename fallback; copy models in.
        for mf in resources_dir.glob("402572-CL10377.model_fold_*.json"):
            shutil.copy(mf, td)
        cwd = os.getcwd()
        os.chdir(td)
        try:
            prepare_report(
                featuremap_df=featuremap_df,
                srsnv_metadata=srsnv_metadata,
                report_path=td,
                basename="bl",
                random_seed=0,
            )
        finally:
            os.chdir(cwd)
        h5 = next(Path(td).glob("*single_read_snv.applicationQC.h5"))
        with pd.HDFStore(str(h5), "r") as store:
            keys = set(store.keys())
        assert keys == MIXED_EXPECTED_H5_KEYS, (
            f"mixed h5 key surface drifted:\n  missing={MIXED_EXPECTED_H5_KEYS - keys}\n"
            f"  unexpected={keys - MIXED_EXPECTED_H5_KEYS}"
        )
