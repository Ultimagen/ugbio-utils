"""
System test for train_cnv_filtering_model.py

Tests the complete CNV filtering model training pipeline using real data
to ensure the entire process works end-to-end without mocking internal functions.
"""

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ugbio_filtering import train_cnv_filtering_model


@pytest.fixture
def resources_dir():
    """Get path to test resources directory"""
    return Path(__file__).parent.parent / "resources"


@pytest.fixture
def cnv_test_data(resources_dir):
    """Path to CNV test data H5 file"""
    test_file = resources_dir / "cnv.gt.test.h5"
    if not test_file.exists():
        pytest.skip(f"Test data file not found: {test_file}")
    return str(test_file)


def test_train_cnv_filtering_model_complete_pipeline(cnv_test_data, tmpdir):
    """
    Test the complete CNV filtering model training pipeline

    This test runs the entire pipeline without mocking internal functions
    to ensure it produces valid results on real data.
    """

    tmpdir = str(tmpdir)
    # Set up output paths
    output_model = os.path.join(tmpdir, "cnv_model.pkl")
    output_results = os.path.join(tmpdir, "cnv_model.h5")

    # Verify input data exists and has expected structure
    data = pd.read_hdf(cnv_test_data, key="calls")
    print(f"Loaded test data with {len(data)} samples")
    print(f"Columns: {list(data.columns)}")

    # Ensure we have the minimum required columns for the model
    required_features = ["svlen", "jump_alignments", "cnv_source", "roundedcopynumber", "flt"]
    missing_features = [col for col in required_features if col not in data.columns]

    if missing_features:
        pytest.skip(f"Test data missing required features: {missing_features}")

    if "label" not in data.columns:
        pytest.skip("Test data missing 'label' column required for training")

    # Verify we have both positive and negative samples for training
    label_counts = data["label"].value_counts()
    print(f"Label distribution: {label_counts.to_dict()}")

    if len(label_counts) < 2:
        pytest.skip("Test data needs both positive and negative labels for training")

    # Run the training pipeline
    print("\nRunning CNV model training pipeline...")

    # Simulate command line arguments
    import sys

    original_argv = sys.argv
    try:
        sys.argv = [
            "train_cnv_filtering_model.py",
            cnv_test_data,
            output_model,
            "--test-size",
            "0.3",  # Use smaller test size for small dataset
            "--random-state",
            "42",
        ]

        # Run main function
        train_cnv_filtering_model.main()

    finally:
        sys.argv = original_argv

    # Verify outputs were created
    assert os.path.exists(output_model), f"Model file not created: {output_model}"
    assert os.path.exists(output_results), f"Results file not created: {output_results}"

    # Verify model file is valid
    with open(output_model, "rb") as f:
        model = pickle.load(f)

    # Check model has expected structure (should be a dict with xgb and transformer)
    assert isinstance(model, dict), "Model should be a dictionary"
    assert "xgb" in model, "Model should contain 'xgb' key"
    assert "transformer" in model, "Model should contain 'transformer' key"

    # Check that the xgb component has required methods
    xgb_model = model["xgb"]
    assert hasattr(xgb_model, "predict"), "XGB model should have predict method"
    assert hasattr(xgb_model, "predict_proba"), "XGB model should have predict_proba method"

    print(f"✓ Model type: {type(model)}")
    print(f"✓ Model keys: {list(model.keys())}")
    print(f"✓ XGB model type: {type(xgb_model)}")

    # Verify results file is valid - check both keys that are actually saved
    results = pd.read_hdf(output_results, key="optimal_recall_precision")
    assert not results.empty, "Results should not be empty"
    assert "accuracy" in results.columns, "Results should contain accuracy column"

    # Check that results contain reasonable accuracy (> 0.0, <= 1.0)
    accuracy = results["accuracy"].iloc[0]
    assert 0.0 <= accuracy <= 1.0, f"Accuracy should be between 0 and 1, got {accuracy}"

    # Also verify the recall-precision curve data exists
    curve_results = pd.read_hdf(output_results, key="recall_precision_curve")
    assert not curve_results.empty, "Curve results should not be empty"
    assert "precision" in curve_results.columns, "Curve results should contain precision column"
    assert "recall" in curve_results.columns, "Curve results should contain recall column"
    assert "f1" in curve_results.columns, "Curve results should contain f1 column"

    print("✓ Model training completed successfully")
    print(f"✓ Model saved to: {output_model}")
    print(f"✓ Results saved to: {output_results}")
    print(f"✓ Test accuracy: {accuracy:.4f}")


def test_load_and_prepare_data_function(cnv_test_data):
    """
    Test the load_and_prepare_data function specifically
    """

    # Test data loading
    x, y = train_cnv_filtering_model.load_and_prepare_data(cnv_test_data)

    # Verify data structure
    assert isinstance(x, pd.DataFrame), "Features should be a DataFrame"
    assert isinstance(y, pd.Series), "Labels should be a Series"
    assert len(x) == len(y), "Features and labels should have same length"
    assert len(x) > 0, "Should have some data"

    # Check feature columns
    expected_features = ["svlen", "jump_alignments", "cnv_source", "roundedcopynumber", "flt"]
    for feature in expected_features:
        assert feature in x.columns, f"Feature {feature} should be in X"

    # Check labels are binary
    unique_labels = y.unique()
    assert len(unique_labels) <= 2, f"Should have at most 2 unique labels, got {len(unique_labels)}"

    print(f"✓ Data loading successful: {len(x)} samples with {len(x.columns)} features")


def test_create_model_pipeline_function():
    """
    Test the create_model_pipeline function
    """

    model = train_cnv_filtering_model.create_model_pipeline()

    # Verify model structure
    assert hasattr(model, "fit"), "Model should have fit method"
    assert hasattr(model, "predict"), "Model should have predict method"
    assert hasattr(model, "predict_proba"), "Model should have predict_proba method"

    # Check pipeline steps
    assert "preprocessor" in model.named_steps, "Model should have preprocessor step"
    assert "classifier" in model.named_steps, "Model should have classifier step"

    print("✓ Model pipeline creation successful")


def test_train_cnv_filtering_model_consistent_results(cnv_test_data, resources_dir, tmpdir):
    """
    Test that the CNV filtering model produces consistent results

    This test compares the newly generated model results against expected results
    to ensure training produces consistent, reproducible outputs.
    """

    tmpdir = str(tmpdir)
    # Set up output paths
    output_model = os.path.join(tmpdir, "cnv_model.pkl")
    output_results = os.path.join(tmpdir, "cnv_model.h5")

    # Run the training pipeline with fixed parameters for reproducibility
    import sys

    original_argv = sys.argv
    try:
        sys.argv = [
            "train_cnv_filtering_model.py",
            cnv_test_data,
            output_model,
            "--test-size",
            "0.3",
            "--random-state",
            "42",
        ]

        # Run main function
        train_cnv_filtering_model.main()

    finally:
        sys.argv = original_argv

    # Verify outputs were created
    assert os.path.exists(output_model), f"Model file not created: {output_model}"
    assert os.path.exists(output_results), f"Results file not created: {output_results}"

    # Load expected results
    expected_results_file = resources_dir / "expected_cnv_model_results.h5"
    assert expected_results_file.exists(), f"Expected results file not found: {expected_results_file}"

    # Load both results - only compare optimal_recall_precision
    actual_results = pd.read_hdf(output_results, key="optimal_recall_precision")
    expected_results = pd.read_hdf(str(expected_results_file), key="optimal_recall_precision")

    # Verify structure matches
    assert list(actual_results.columns) == list(
        expected_results.columns
    ), f"Column mismatch: actual={list(actual_results.columns)}, expected={list(expected_results.columns)}"

    assert len(actual_results) == len(
        expected_results
    ), f"Row count mismatch: actual={len(actual_results)}, expected={len(expected_results)}"

    # Compare metrics with small epsilon tolerance for floating point precision
    epsilon = 1e-6
    for metric in ["accuracy", "precision", "recall", "f1"]:
        if metric in actual_results.columns:
            actual_values = np.array(actual_results[metric])
            expected_values = np.array(expected_results[metric])

            diff = np.abs(actual_values - expected_values)
            max_diff = diff.max()

            assert max_diff <= epsilon, (
                f"Metric '{metric}' differs by {max_diff} (max allowed: {epsilon}). "
                f"Actual: {actual_values}, Expected: {expected_values}"
            )

    # Compare support (integer values - should be exact)
    if "support" in actual_results.columns:
        actual_support = np.array(actual_results["support"])
        expected_support = np.array(expected_results["support"])
        assert np.array_equal(
            actual_support, expected_support
        ), f"Support values differ: actual={actual_support}, expected={expected_support}"

    print("✓ Model results are consistent with expected values")
    print(f"✓ All metrics within epsilon tolerance: {epsilon}")
