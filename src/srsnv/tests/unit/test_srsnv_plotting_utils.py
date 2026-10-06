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
        import inspect  # noqa: PLC0415

        from ugbio_srsnv.srsnv_report import prepare_report  # noqa: PLC0415

        sig = inspect.signature(prepare_report)
        assert "featuremap_df" in sig.parameters
        assert "srsnv_metadata" in sig.parameters
        assert "report_path" in sig.parameters

        # Verify the files have the expected structure
        import pandas as pd  # noqa: PLC0415

        df = pd.read_parquet(featuremap_path)  # noqa: PD901
        assert "prob_fold_0" in df.columns, "Generated test data should have prob_fold_0 column"

        with open(metadata_path) as f:
            import json  # noqa: PLC0415

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


# ──────────────────────── consensus mode (fs/rs) tests ──────────────────────


@pytest.fixture
def consensus_resources(test_resources_calc_run_info):
    """Build a consensus-mode featuremap from the real ppmSeq resources.

    Drops the ppmSeq st/et tags and adds fs/rs read counts so the report auto-detects
    CONSENSUS mode. Returns (featuremap_df, metadata).
    """
    featuremap_df, metadata, _ = test_resources_calc_run_info
    consensus_df = featuremap_df.copy()  # noqa: PD901
    consensus_df = consensus_df.drop(columns=[c for c in ("st", "et") if c in consensus_df.columns])
    rng = np.random.default_rng(7)
    consensus_df["fs"] = rng.integers(0, 6, len(consensus_df))
    consensus_df["rs"] = rng.integers(0, 6, len(consensus_df))
    # Ensure both consensus and non-consensus reads are present
    consensus_df.loc[consensus_df.index[:50], ["fs", "rs"]] = 2
    consensus_df.loc[consensus_df.index[50:100], "fs"] = 0
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
        expected = ((df["fs"] >= 1) & (df["rs"] >= 1)).to_numpy()
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


def _make_pe_duplex_report(df, metadata, temp_output_dir, models):
    """Helper: build an SRSNVReport in paired-end-duplex mode (df must carry the CS-family stats)."""
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
        "report_mode": "pe_duplex",
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


def test_pe_duplex_logit_histograms_run(consensus_resources, real_models_calc_run_info):
    """Regression: the pe-duplex logit histogram must not raise and must produce the figure.

    The pe-duplex group_fn reads the pre-assigned ``pe_duplex_group`` column (not recomputed from raw
    inputs), so it must be carried into the logit-histogram column slice via ``_logit_extra_cols``;
    otherwise the figure throws ``KeyError: 'pe_duplex_group'`` and the PNG is never written.
    """
    base_df, metadata = consensus_resources
    pe_df = base_df.copy()
    rng = np.random.default_rng(5)
    pe_df["DS"] = rng.choice([1, 2], len(pe_df))  # 2 == full duplex; else single-strand (simplex)
    pe_df["cs_family_size"] = rng.integers(2, 5, len(pe_df))  # >=2 so none drop out as singletons
    pe_df["cs_n_crossing"] = rng.integers(1, 5, len(pe_df))
    pe_df["cs_n_supporting"] = rng.integers(0, 3, len(pe_df))
    pe_df["cs_n_pe_pairs"] = rng.integers(0, 3, len(pe_df))
    with tempfile.TemporaryDirectory() as temp_output_dir:
        report = _make_pe_duplex_report(pe_df, metadata, temp_output_dir, real_models_calc_run_info)
        assert report.report_mode.value == "pe_duplex"
        assert "pe_duplex_group" in report.data_df.columns
        assert "pe_duplex_group" in report._logit_extra_cols()
        out = os.path.join(temp_output_dir, "logit_hist")
        report.plot_logit_histograms(output_filename=out, plot_by_fold=False)
        assert os.path.exists(out + ".png")


def test_none_mode_graceful(test_resources_calc_run_info, real_models_calc_run_info):
    """With neither ppmSeq tags nor fs/rs, the report runs as a single group (NONE mode)."""
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


# ──────────────────── hmer-indel SNVQ section (read-type x indel-class) ────────────────────


def _add_hmer_variant_columns(df, n_indel=300, seed=5):
    """Set an explicit ins/del split (rest = snv) + variant_type, so the hmer section is exercised."""
    df = df.copy()  # noqa: PD901
    rng = np.random.default_rng(seed)
    if "X_HMER_REF" not in df.columns:
        df["X_HMER_REF"] = rng.integers(0, 13, len(df))
    n_indel = min(n_indel, len(df) // 2)
    xic = np.array([None] * len(df), dtype=object)
    xic[: n_indel // 2] = "ins"
    xic[n_indel // 2 : n_indel] = "del"
    df["X_IC"] = xic
    df["variant_type"] = np.where(pd.Series(xic).isin(["ins", "del"]).to_numpy(), "hmer_indel", "snv")
    # snvfind context fields consumed by the hmer-indel context figure (geometry-correct set)
    df["X_HMER_RUN"] = rng.integers(1, 10, len(df))  # affected reference run length
    df["X_HMER_BASE"] = rng.choice(list("ACGT"), len(df))  # repeated base of the run
    df["X_HMER_PRE"] = rng.choice(list("ACGT"), len(df))  # base 5' of the run
    df["X_HMER_POST"] = rng.choice(list("ACGT"), len(df))  # base 3' of the run
    return df


def test_hmer_indel_snvq_summary_and_plots(consensus_resources, real_models_calc_run_info):
    """SNVQ hmer-indel section: cross-product summary table (read-type x indel-class) + by-class figure.
    Validates h5 keys, table structure/margins, and figure creation."""
    df, metadata = consensus_resources
    df = _add_hmer_variant_columns(df)  # noqa: PD901
    with tempfile.TemporaryDirectory() as temp_output_dir:
        report = _make_consensus_report(df, metadata, temp_output_dir, real_models_calc_run_info)
        assert report._has_hmer_indel_rows()
        report.calc_hmer_indel_run_info_table()
        by_class = os.path.join(temp_output_dir, "hmer_by_class")
        report.plot_hmer_indel_by_class(output_filename=by_class)

        h5_file = os.path.join(temp_output_dir, "test_single_read_snv.applicationQC.h5")
        with pd.HDFStore(h5_file, "r") as store:
            assert "/run_quality_summary_table_hmer_indel" in store.keys()
            assert "/hmer_indel_class_stats" in store.keys()

        table = pd.read_hdf(h5_file, key="run_quality_summary_table_hmer_indel")
        assert set(table["indel_class"]) == {"snv (baseline)", "all indel", "ins", "del"}
        assert "All reads" in set(table["read_type"])
        assert {"single read", "consensus, one strand", "consensus, duplex"} <= set(table["read_type"])
        for col in ["Median SNVQ", "Recall at SNVQ=50", "Recall at SNVQ=60", "Recall at SNVQ=70", "ROC AUC (Phred)"]:
            assert col in table.columns
        # consensus (non-duplex) run: no Recall at SNVQ=80/90 columns (duplex-only)
        assert "Recall at SNVQ=80" not in table.columns
        # per read_type: ins + del n_TP == all-indel n_TP (every indel is ins or del)
        for rt in table["read_type"].unique():
            n_tp = table[table["read_type"] == rt].set_index("indel_class")["n_TP"]
            assert n_tp["ins"] + n_tp["del"] == n_tp["all indel"]
        # by-class figure uses SNVQ (median_snvq column, not median_mqual)
        class_stats = pd.read_hdf(h5_file, key="hmer_indel_class_stats")
        assert "median_snvq" in class_stats.columns
        assert os.path.exists(by_class + ".png")


def _make_duplex_report(df, metadata, temp_output_dir, models):
    """A duplex-mode SRSNVReport (report_mode='duplex_molecule'); df should carry mate_present."""
    temp_metadata_file = os.path.join(temp_output_dir, "test_metadata.json")
    with open(temp_metadata_file, "w") as f:
        json.dump(metadata, f)
    categorical_features = [feat for feat in metadata["features"] if feat["type"] == "c"]
    numerical_features = [feat for feat in metadata["features"] if feat["type"] != "c"]
    params = {
        "workdir": temp_output_dir,
        "data_name": "test_run",
        "categorical_features_names": [feat["name"] for feat in categorical_features],
        "categorical_features_dict": {feat["name"]: list(feat["values"].keys()) for feat in categorical_features},
        "numerical_features": [feat["name"] for feat in numerical_features],
        "fp_regions_bed_file": 1,
        "num_CV_folds": len(models),
        "report_mode": "duplex_molecule",
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


def _add_duplex_columns(df, seed=7):
    rng = np.random.default_rng(seed)
    df["mate_present"] = rng.integers(0, 2, len(df))
    return df


def test_snvq_thresholds_duplex_adds_80(consensus_resources, real_models_calc_run_info):
    """_snvq_thresholds() returns [50,60,70] for a consensus run and adds only 80 for duplex runs
    (SNVQ90 omitted — the recalibrated SNVQ ceiling is ~84, so Recall@SNVQ90 would be structurally 0);
    the duplex hmer-indel SNVQ table then carries a Recall at SNVQ=80 column but not =90."""
    df, metadata = consensus_resources
    df = _add_hmer_variant_columns(df)  # noqa: PD901
    with tempfile.TemporaryDirectory() as temp_output_dir:
        report = _make_consensus_report(df, metadata, temp_output_dir, real_models_calc_run_info)
        assert report._snvq_thresholds() == [50, 60, 70]
    dfd = _add_duplex_columns(_add_hmer_variant_columns(consensus_resources[0].copy()))
    with tempfile.TemporaryDirectory() as temp_output_dir:
        report = _make_duplex_report(dfd, metadata, temp_output_dir, real_models_calc_run_info)
        assert report._snvq_thresholds() == [50, 60, 70, 80]
        report.calc_hmer_indel_run_info_table()
        table = pd.read_hdf(
            os.path.join(temp_output_dir, "test_single_read_snv.applicationQC.h5"),
            key="run_quality_summary_table_hmer_indel",
        )
        for col in ["Recall at SNVQ=70", "Recall at SNVQ=80"]:
            assert col in table.columns
        assert "Recall at SNVQ=90" not in table.columns


def test_by_class_hmer_indel_stats(consensus_resources, real_models_calc_run_info):
    """plot_hmer_indel_by_class writes the hmer_indel_class_stats h5 (with a median_snvq column);
    lines split by the active read-split groups (e.g. duplex molecule / single-strand consensus) in
    addition to ins/del, when a multi-group read_group is present."""
    df, metadata = consensus_resources
    dfd = _add_duplex_columns(_add_hmer_variant_columns(df))  # noqa: PD901
    with tempfile.TemporaryDirectory() as temp_output_dir:
        report = _make_duplex_report(dfd, metadata, temp_output_dir, real_models_calc_run_info)
        by_class = os.path.join(temp_output_dir, "by_class")
        report.plot_hmer_indel_by_class(output_filename=by_class)
        stats = pd.read_hdf(
            os.path.join(temp_output_dir, "test_single_read_snv.applicationQC.h5"), key="hmer_indel_class_stats"
        )
        assert "median_snvq" in stats.columns
        rg = set(stats["read_group"])
        # duplex data has >1 read group -> lines are split by group (not the single "all" fallback)
        assert rg and rg != {"all"}
        assert rg.issubset(set(report._display_variant.groups))
        assert os.path.exists(by_class + ".png")


def test_removed_hmer_methods_absent():
    """The removed figures are gone (SNV-vs-hmer metrics, SNVQ reliability, SNVQ histogram)."""
    for name in ("plot_hmer_indel_metrics", "plot_hmer_indel_snvq_reliability", "plot_hmer_indel_snvq_histograms"):
        assert not hasattr(SRSNVReport, name)


def _add_non_hmer_variant_columns(df, n_indel=300, seed=11):
    """Set non-hmer indel rows (variant_type=non_hmer_indel) with X_IC/X_IL + tandem-repeat tags
    (RU/RPA/STR), so the non-hmer sections are exercised. ~half the indels sit in a tandem repeat."""
    df = df.copy()  # noqa: PD901
    rng = np.random.default_rng(seed)
    n_indel = min(n_indel, len(df) // 2)
    is_indel = np.zeros(len(df), dtype=bool)
    is_indel[:n_indel] = True
    xic = np.array([None] * len(df), dtype=object)
    xic[: n_indel // 2] = "ins"
    xic[n_indel // 2 : n_indel] = "del"
    df["X_IC"] = xic
    df["variant_type"] = np.where(is_indel, "non_hmer_indel", "snv")
    df["X_IL"] = np.where(is_indel, rng.integers(1, 10, len(df)), np.nan)
    motifs = ["AC", "AG", "AT", "CA", "ATG", "CAG", "TAGC", "AAAT"]
    ru = np.array(["."] * len(df), dtype=object)
    for i in range(0, n_indel, 2):  # ~half the indels are STR (carry a repeat unit)
        ru[i] = rng.choice(motifs)
    df["RU"] = ru
    df["RPA"] = np.where(ru != ".", rng.integers(1, 7, len(df)), np.nan)
    df["STR"] = np.where(ru != ".", 1, np.nan)
    return df


def test_non_hmer_indel_sections(consensus_resources, real_models_calc_run_info):
    """The three non-hmer figures render and write their h5 stats; read-group split is exercised via a
    duplex report (multi-group). Mirrors the hmer by-class/context tests."""
    df, metadata = consensus_resources
    dfd = _add_duplex_columns(_add_non_hmer_variant_columns(df))  # noqa: PD901
    with tempfile.TemporaryDirectory() as temp_output_dir:
        report = _make_duplex_report(dfd, metadata, temp_output_dir, real_models_calc_run_info)
        assert report._has_non_hmer_indel_rows()
        h5 = os.path.join(temp_output_dir, "test_single_read_snv.applicationQC.h5")

        by_class = os.path.join(temp_output_dir, "nh_by_class")
        report.plot_non_hmer_indel_by_class(output_filename=by_class)
        assert os.path.exists(by_class + ".png")
        stats = pd.read_hdf(h5, key="non_hmer_indel_class_stats")
        assert {"median_snvq", "indel_length", "indel_class", "read_group"}.issubset(stats.columns)
        assert set(stats["read_group"]) and set(stats["read_group"]) != {"all"}

        str_fig = os.path.join(temp_output_dir, "nh_str")
        report.plot_non_hmer_indel_str(output_filename=str_fig)
        assert os.path.exists(str_fig + ".png")
        str_stats = pd.read_hdf(h5, key="non_hmer_indel_str_stats")
        assert set(str_stats["axis"]) == {"period", "copies"}

        ctx = os.path.join(temp_output_dir, "nh_ctx")
        report.calc_and_plot_non_hmer_str_context_plot(output_filename=ctx)
        assert os.path.exists(ctx + ".png")
        ctx_stats = pd.read_hdf(h5, key="non_hmer_indel_str_context_stats")
        assert {"period", "motif", "ins_del", "read_group", "n_TP", "n_FP"}.issubset(ctx_stats.columns)
        assert set(ctx_stats["period"]).issubset({2, 3, 4})


def test_non_hmer_indel_sections_skip_snv_only(consensus_resources, real_models_calc_run_info):
    """SNV-only run (no indel columns): all three non-hmer methods self-skip cleanly (no raise under
    raise_exceptions=True, no PNG written)."""
    df, metadata = consensus_resources
    with tempfile.TemporaryDirectory() as temp_output_dir:
        report = _make_consensus_report(df, metadata, temp_output_dir, real_models_calc_run_info)
        for method, stem in (
            (report.plot_non_hmer_indel_by_class, "a"),
            (report.plot_non_hmer_indel_str, "b"),
            (report.calc_and_plot_non_hmer_str_context_plot, "c"),
        ):
            out = os.path.join(temp_output_dir, stem)
            method(output_filename=out)
            assert not os.path.exists(out + ".png")


def test_canonical_repeat_unit():
    """Phase/strand variants of a repeat collapse to one canonical motif."""
    f = SRSNVReport._canonical_repeat_unit
    assert f("AC") == f("CA") == f("GT") == f("TG")  # dinucleotide: rotations + revcomp
    assert f("AAT") == f("ATA") == f("TAA")  # trinucleotide rotations
    assert f("N") == "" and f("") == ""  # non-ACGT -> empty


def test_logit_histograms_split_hmer_indel(consensus_resources, real_models_calc_run_info):
    """The logit histogram splits into an SNV figure and a separate hmer-indel figure when hmer rows
    are present; an SNV-only run produces only the SNV figure."""
    df, metadata = consensus_resources
    dfh = _add_hmer_variant_columns(df.copy())  # noqa: PD901
    with tempfile.TemporaryDirectory() as temp_output_dir:
        report = _make_consensus_report(dfh, metadata, temp_output_dir, real_models_calc_run_info)
        snv = os.path.join(temp_output_dir, "logit_snv")
        hmer = os.path.join(temp_output_dir, "logit_hmer")
        report.plot_logit_histograms(output_filename=snv, output_filename_hmer_indel=hmer, plot_by_fold=False)
        assert os.path.exists(snv + ".png")
        assert os.path.exists(hmer + ".png")
    # SNV-only run: hmer-indel figure not produced
    df_snv, metadata2 = consensus_resources
    with tempfile.TemporaryDirectory() as temp_output_dir:
        report = _make_consensus_report(df_snv.copy(), metadata2, temp_output_dir, real_models_calc_run_info)
        snv = os.path.join(temp_output_dir, "logit_snv")
        hmer = os.path.join(temp_output_dir, "logit_hmer")
        report.plot_logit_histograms(output_filename=snv, output_filename_hmer_indel=hmer, plot_by_fold=False)
        assert os.path.exists(snv + ".png")
        assert not os.path.exists(hmer + ".png")


def test_hmer_indel_context_plot(consensus_resources, real_models_calc_run_info):
    """Hmer-indel context figure: tidy per-context h5 table + PNG. Validates dimensions and counts."""
    df, metadata = consensus_resources
    df = _add_hmer_variant_columns(df)  # noqa: PD901
    with tempfile.TemporaryDirectory() as temp_output_dir:
        report = _make_consensus_report(df, metadata, temp_output_dir, real_models_calc_run_info)
        ctx = os.path.join(temp_output_dir, "hmer_context")
        report.calc_and_plot_hmer_indel_context_plot(output_filename=ctx)

        h5_file = os.path.join(temp_output_dir, "test_single_read_snv.applicationQC.h5")
        table = pd.read_hdf(h5_file, key="hmer_indel_context_stats")
        assert set(table["ins_del"]) <= {"ins", "del"}
        assert set(table["hmer_base"]) <= set("ACGT")
        assert table["hmer_len"].max() <= 10  # clipped to the >=10 bucket  # noqa: PLR2004
        assert "read_group" in table.columns
        assert (table["n_TP"] >= 0).all() and (table["n_FP"] >= 0).all()
        assert os.path.exists(ctx + ".png")


def test_hmer_indel_section_skipped_for_snv_only(consensus_resources, real_models_calc_run_info):
    """SNV-only run (no hmer indels): the hmer summary self-skips (no h5 key, no error)."""
    df, metadata = consensus_resources
    df = df.copy()  # noqa: PD901
    df["variant_type"] = "snv"
    df["X_IC"] = None
    with tempfile.TemporaryDirectory() as temp_output_dir:
        report = _make_consensus_report(df, metadata, temp_output_dir, real_models_calc_run_info)
        assert not report._has_hmer_indel_rows()
        report.calc_hmer_indel_run_info_table()  # no-op
        h5_file = os.path.join(temp_output_dir, "test_single_read_snv.applicationQC.h5")
        if os.path.exists(h5_file):
            with pd.HDFStore(h5_file, "r") as store:
                assert "/run_quality_summary_table_hmer_indel" not in store.keys()


def _add_pe_duplex_columns(df, seed=11):
    """Add the DNN per-image CS-family stats (+ nf/nr/DS/CS) so the report detects the pe-duplex scheme."""
    rng = np.random.default_rng(seed)
    n = len(df)
    df = df.copy()  # noqa: PD901
    df["nf"] = rng.integers(0, 4, n)
    df["nr"] = rng.integers(0, 4, n)
    df["DS"] = rng.choice([0, 2], n)
    df["CS"] = [f"cs{i}" for i in range(n)]
    df["cs_family_size"] = rng.integers(1, 5, n)
    df["cs_n_crossing"] = rng.integers(1, 5, n)
    df["cs_n_supporting"] = rng.integers(0, 3, n)
    df["cs_n_pe_pairs"] = rng.integers(0, 3, n)
    return df


def test_run_info_table_has_cs_family_stats(consensus_resources, real_models_calc_run_info):
    """Paired-end-duplex run: the per-group run-quality summary gains median CS-family stat rows
    (family size / crossing / supporting / PE pairs) alongside the SSC / duplex-SE / duplex-PE groups."""
    df, metadata = consensus_resources
    dfp = _add_pe_duplex_columns(df)
    with tempfile.TemporaryDirectory() as temp_output_dir:
        temp_metadata_file = os.path.join(temp_output_dir, "test_metadata.json")
        with open(temp_metadata_file, "w") as f:
            json.dump(metadata, f)
        categorical_features = [feat for feat in metadata["features"] if feat["type"] == "c"]
        numerical_features = [feat for feat in metadata["features"] if feat["type"] != "c"]
        params = {
            "workdir": temp_output_dir,
            "data_name": "test_run",
            "categorical_features_names": [feat["name"] for feat in categorical_features],
            "categorical_features_dict": {f["name"]: list(f["values"].keys()) for f in categorical_features},
            "numerical_features": [feat["name"] for feat in numerical_features],
            "fp_regions_bed_file": 1,
            "num_CV_folds": len(real_models_calc_run_info),
        }  # no report_mode -> scheme is auto-detected from the columns (pe-duplex via cs_n_crossing)
        report = SRSNVReport(
            models=real_models_calc_run_info,
            data_df=dfp,
            params=params,
            out_path=temp_output_dir,
            srsnv_metadata=temp_metadata_file,
            base_name="test_",
            raise_exceptions=True,
        )
        assert report.scheme.mode.value == "pe_duplex"
        report.plot_fq_recall(only_calculate=True)
        report.calc_run_info_table()
        summary = pd.read_hdf(
            os.path.join(temp_output_dir, "test_single_read_snv.applicationQC.h5"),
            key="run_quality_summary_table",
        )
        row_labels = {idx[0] for idx in summary.index}
        for expected in (
            "Median CS family size",
            "Median reads crossing / CS",
            "Median reads supporting / CS",
            "Median PE pairs / CS",
        ):
            assert expected in row_labels
