#!/usr/bin/env python3
"""
CNV Filtering Model Training Script

This script trains a machine learning model to filter CNV (Copy Number Variant) calls
based on features extracted from VCF files. The model distinguishes between true
positives (TP) and false positives (FP) CNV calls.

Usage:
    python train_cnv_filtering_model.py <input_h5_file> <output_model_file> [options]

Required Input:
    - H5 file with 'calls' key containing CNV data with a 'label' column
    - The 'label' column should classify variants as 'TP' (true) or 'FP' (false)

Output:
    - PKL file containing trained model and transformer for inference
"""

import argparse
import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder

warnings.filterwarnings("ignore")


class SafeLabelEncoder(BaseEstimator):
    """Label encoder that handles unseen categories gracefully."""

    def __init__(self):
        self.encoders = {}

    def fit(self, x, y=None):
        """Fit the encoder on the data."""
        for col in x.select_dtypes(include=["object"]).columns:
            encoder = LabelEncoder()
            # Add 'unknown' to handle unseen categories
            unique_vals = list(x[col].astype(str).unique())
            if "unknown" not in unique_vals:
                unique_vals.append("unknown")
            encoder.fit(unique_vals)
            self.encoders[col] = encoder
        return self

    def transform(self, x):
        """Transform the data using fitted encoders."""
        result = x.copy()
        for col in self.encoders:
            # Handle unseen categories by mapping them to 'unknown'
            col_values = x[col].astype(str)
            # Map unseen values to 'unknown'
            known_classes = set(self.encoders[col].classes_)
            col_values = col_values.apply(lambda x, kc=known_classes: x if x in kc else "unknown")
            result[col] = self.encoders[col].transform(col_values)
        return result

    def fit_transform(self, x, y=None, **fit_params):
        """Fit and transform the data."""
        return self.fit(x, y).transform(x)

    def set_output(self, *, transform=None):
        """Set the output container format."""
        # This method is required for sklearn compatibility
        # Since we're already working with pandas DataFrames, we don't need to do anything special
        return self


# Feature columns used throughout the CNV filtering model
NUMERIC_FEATURES = ["svlen", "jump_alignments", "roundedcopynumber"]
CATEGORICAL_FEATURES = ["cnv_source", "region_annotations"]
FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def get_active_features(disabled_features=None):
    """
    Get active feature lists based on disabled features

    Args:
        disabled_features: List of features to disable

    Returns:
        Tuple of (active_numeric_features, active_categorical_features, active_feature_cols)
    """
    if disabled_features is None:
        disabled_features = []

    # Validate disabled features
    invalid_features = [f for f in disabled_features if f not in FEATURE_COLS]
    if invalid_features:
        raise ValueError(f"Invalid features to disable: {invalid_features}. Available features: {FEATURE_COLS}")

    # Filter out disabled features
    active_numeric_features = [f for f in NUMERIC_FEATURES if f not in disabled_features]
    active_categorical_features = [f for f in CATEGORICAL_FEATURES if f not in disabled_features]
    active_feature_cols = active_numeric_features + active_categorical_features

    return active_numeric_features, active_categorical_features, active_feature_cols


def process_numeric_features(x):
    """Process all numeric features: convert svlen tuples and scale jump alignments"""
    x = x.copy()
    # Convert svlen tuples to integers by extracting first element
    if "svlen" in x.columns:
        x["svlen"] = x["svlen"].apply(lambda val: val[0])
    # Scale jump alignments
    if "jump_alignments" in x.columns:
        x["jump_alignments"] = x["jump_alignments"]
    return x


def create_preprocessing_pipeline(numeric_features=NUMERIC_FEATURES, categorical_features=CATEGORICAL_FEATURES):
    """
    Create preprocessing pipeline for CNV features

    Args:
        numeric_features: List of numeric features to use (default: NUMERIC_FEATURES)
        categorical_features: List of categorical features to use (default: CATEGORICAL_FEATURES)

    Returns:
        ColumnTransformer: Preprocessing pipeline
    """
    # Create numeric transformer with svlen conversion, NaN filling and jump alignment scaling
    # Tree-based models don't require scaling, so we only handle NaN values and process features
    numeric_transformer = Pipeline(
        [
            ("process_features", FunctionTransformer(process_numeric_features, validate=False)),
            ("impute", SimpleImputer(strategy="constant", fill_value=0)),  # Fill NaN values with zeros
        ]
    )

    # Create categorical transformer
    categorical_transformer = Pipeline([("encode", SafeLabelEncoder())])

    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    # Set output to DataFrame instead of numpy array
    preprocessor.set_output(transform="pandas")

    return preprocessor


def create_model_pipeline(numeric_features=None, categorical_features=None):
    """Create complete model pipeline with preprocessing and classification"""
    preprocessor = create_preprocessing_pipeline(numeric_features, categorical_features)

    # Best parameters found from hyperparameter optimization
    classifier = RandomForestClassifier(
        random_state=42,
        class_weight="balanced",
        n_estimators=100,
        max_depth=5,
        max_features=0.5,
        max_leaf_nodes=10,
        max_samples=0.9,
        min_samples_leaf=3,
        min_samples_split=10,
    )

    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])

    return pipeline


def load_and_prepare_data(h5_file: str, feature_cols=FEATURE_COLS):
    """
    Load data from H5 file and prepare features and labels

    Args:
        h5_file: Path to H5 file with CNV data
        feature_cols: List of feature columns to use (default: FEATURE_COLS)

    Returns:
        Tuple of (features_df, labels_series)
    """
    print(f"Loading data from {h5_file}...")

    # Load data from H5 file
    try:
        data = pd.read_hdf(h5_file, key="calls")
    except Exception as e:
        print(f"Error loading H5 file: {e}")
        sys.exit(1)

    print(f"Loaded {len(data)} records")

    # Check required columns
    missing_features = [col for col in feature_cols if col not in data.columns]

    if missing_features:
        print(f"Error: Missing required features: {missing_features}")
        print(f"Available columns: {list(data.columns)}")
        sys.exit(1)

    if "label" not in data.columns:
        print("Error: 'label' column not found in data")
        print(f"Available columns: {list(data.columns)}")
        sys.exit(1)

    # Prepare features and labels
    x = data[feature_cols].copy()
    y = data["label"].copy()

    # Print label distribution
    print("\nLabel distribution:")
    print(y.value_counts())

    return x, y


def evaluate_model(df: pd.DataFrame, model, preprocessor, data_type: str, feature_cols=FEATURE_COLS) -> tuple:
    """
    Evaluate model performance on data similar to train_models_pipeline

    Args:
        df: DataFrame with features and labels
        model: Trained model (classifier part only)
        preprocessor: Preprocessing pipeline (transformer part)
        data_type: String describing data type for logging (i.e. training or testing data)
        feature_cols: List of feature columns to use (default: FEATURE_COLS)

    Returns:
        Tuple of (accuracy_df, curve_df) similar to variant_filtering_utils.eval_model
    """
    print(f"\n{data_type} Evaluation:")
    print("=" * (len(data_type) + 12))

    # Extract features and labels
    x = df[feature_cols].copy()
    y = df["label"].copy()

    # Transform features
    x_transformed = preprocessor.transform(x)

    # Make predictions
    y_pred = model.predict(x_transformed)
    y_pred_proba = model.predict_proba(x_transformed)

    # Convert both labels and predictions to binary for metrics (TP=1, FP=0)
    y_binary = (y == "TP").astype(int)
    y_pred_binary = (y_pred == "TP").astype(int)

    # Calculate metrics
    accuracy = accuracy_score(y_binary, y_pred_binary)
    precision, recall, f1, _ = precision_recall_fscore_support(y_binary, y_pred_binary, average="binary")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_binary, y_pred_binary))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_binary, y_pred_binary))

    # Create accuracy dataframe (similar to variant_filtering_utils format)
    accuracy_df = pd.DataFrame(
        {
            "group": ["ALL"],
            "accuracy": [accuracy],
            "precision": [precision],
            "recall": [recall],
            "f1": [f1],
            "support": [len(y_binary)],
        }
    )

    # Create curve dataframe with thresholds (simplified version)
    thresholds = np.linspace(0.1, 0.9, 9)
    curve_data = []

    for threshold in thresholds:
        y_pred_thresh = (y_pred_proba[:, 1] >= threshold).astype(int)
        if len(np.unique(y_pred_thresh)) > 1:  # Avoid division by zero
            prec = precision_score(y_binary, y_pred_thresh, zero_division=0)
            rec = recall_score(y_binary, y_pred_thresh, zero_division=0)
            f1_thresh = f1_score(y_binary, y_pred_thresh, zero_division=0)
        else:
            prec = rec = f1_thresh = 0.0

        curve_data.append({"group": "ALL", "precision": prec, "recall": rec, "f1": f1_thresh, "threshold": threshold})

    curve_df = pd.DataFrame(curve_data)

    # Feature importance if available
    if hasattr(model, "feature_importances_"):
        print("\nFeature Importance:")
        # If model has more features than expected, print all
        for name, importance in zip(feature_cols, model.feature_importances_, strict=True):
            print(f"{name}: {importance:.4f}")

    return accuracy_df, curve_df


def save_results(output_file_prefix: str, model, preprocessor, train_results, test_results):
    """
    Save results similar to train_models_pipeline

    Args:
        output_file_prefix: Output file prefix for .pkl and .h5 files
        model: Trained model (classifier)
        preprocessor: Preprocessing pipeline (transformer)
        train_results: Training evaluation results (accuracy_df, curve_df)
        test_results: Test evaluation results (accuracy_df, curve_df)
    """
    print(f"\nSaving results to {output_file_prefix}...")

    # Create results dictionary compatible with existing pipeline
    results_dict = {
        "transformer": preprocessor,
        "xgb": model,  # Use 'xgb' key for compatibility
        "xgb_recall_precision": test_results[0],
        "xgb_recall_precision_curve": test_results[1],
        "xgb_train_recall_precision": train_results[0],
        "xgb_train_recall_precision_curve": train_results[1],
    }

    # Save model and results as pickle
    with open(output_file_prefix + ".pkl", "wb") as file:
        pickle.dump(results_dict, file)

    # Save accuracy metrics to HDF5
    accuracy_dfs = []
    prcdict = {}
    for m_var in ("xgb", "xgb_train"):
        name_optimum = f"{m_var}_recall_precision"
        accuracy_df_per_model = results_dict[name_optimum].copy()
        accuracy_df_per_model["model"] = name_optimum
        accuracy_dfs.append(accuracy_df_per_model)
        prcdict[name_optimum] = results_dict[
            name_optimum.replace("recall_precision", "recall_precision_curve")
        ].set_index("group")

    accuracy_df = pd.concat(accuracy_dfs, ignore_index=True)
    accuracy_df.to_hdf(output_file_prefix + ".h5", key="optimal_recall_precision")

    results_vals = pd.concat(prcdict, names=["model"])
    results_vals = results_vals[["recall", "precision", "f1"]].reset_index()
    results_vals.to_hdf(output_file_prefix + ".h5", key="recall_precision_curve")

    print("Results saved successfully!")
    print(f"Model components: {list(results_dict.keys())}")


def save_model(model, output_file: str):
    """
    Save trained model to pickle file

    Args:
        model: Trained model pipeline
        output_file: Output pickle file path
    """
    print(f"\nSaving model to {output_file}...")

    # Extract components
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    # Create model dict (compatible with filter_variants_pipeline)
    model_data = {
        "xgb": classifier,  # Note: using 'xgb' key for compatibility
        "transformer": preprocessor,
    }

    # Save with pickle
    with open(output_file, "wb") as f:
        pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Model saved successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="Train CNV filtering model", formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__
    )

    parser.add_argument(
        "input_h5", help="Input H5 file with CNV data and labels, should have a key 'calls' with 'label' column"
    )

    parser.add_argument("output_model", help="Output pickle file for trained model")

    parser.add_argument(
        "--test-size", type=float, default=0.25, help="Fraction of data to use for testing (default: 0.25)"
    )

    parser.add_argument("--random-state", type=int, default=42, help="Random state for reproducibility (default: 42)")

    parser.add_argument(
        "--disable",
        action="append",
        default=[],
        help="Disable specific features (can be used multiple times). Available features: " + ", ".join(FEATURE_COLS),
    )

    args = parser.parse_args()

    # Validate arguments
    if not os.path.exists(args.input_h5):
        print(f"Error: Input file {args.input_h5} does not exist")
        sys.exit(1)

    # Create output directory if needed
    output_dir = os.path.dirname(args.output_model)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process disabled features
    try:
        active_numeric_features, active_categorical_features, active_feature_cols = get_active_features(args.disable)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print("CNV Filtering Model Training")
    print("=" * 40)
    print(f"Input H5 file: {args.input_h5}")
    print(f"Output model: {args.output_model}")
    print(f"Test size: {args.test_size}")
    print(f"Random state: {args.random_state}")

    if args.disable:
        print(f"Disabled features: {args.disable}")
    print(f"Active features: {active_feature_cols}")

    # Load and prepare data
    x, y = load_and_prepare_data(args.input_h5, active_feature_cols)

    # Split data
    print(f"\nSplitting data (test_size={args.test_size})...")
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    print(f"Training samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")

    # Train model
    print("\nTraining model...")
    model = create_model_pipeline(active_numeric_features, active_categorical_features)
    model.fit(x_train, y_train)

    # Prepare data for evaluation (similar to train_models_pipeline)
    train_data = pd.concat([x_train, y_train], axis=1)
    test_data = pd.concat([x_test, y_test], axis=1)

    # Extract model components
    classifier = model.named_steps["classifier"]
    preprocessor = model.named_steps["preprocessor"]

    # Evaluate model on both train and test sets
    train_results = evaluate_model(train_data, classifier, preprocessor, "training", active_feature_cols)
    test_results = evaluate_model(test_data, classifier, preprocessor, "test", active_feature_cols)

    # Save results (both .pkl and .h5 files)
    output_prefix = args.output_model.replace(".pkl", "")
    save_results(output_prefix, classifier, preprocessor, train_results, test_results)

    # Save model in the original format for compatibility
    save_model(model, args.output_model)

    # Final summary
    print("\n" + "=" * 50)
    print("Training Complete!")
    print(f"Model saved to: {args.output_model}")
    print(f"Results saved to: {output_prefix}.h5")

    test_acc = accuracy_score(y_test, model.predict(x_test))
    print(f"Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
