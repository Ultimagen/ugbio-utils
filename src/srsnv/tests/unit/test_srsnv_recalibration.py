from pathlib import Path
from types import SimpleNamespace

import xgboost as xgb
from ugbio_srsnv.srsnv_training import (
    FOLD_COL,
    LABEL_COL,
    MQUAL,
    PROB_ORIG,
    PROB_RECAL,
    SRSNVTrainer,
)


def test_recalibration_columns_count(tmp_path):
    """Trainer should append prob_orig, prob_recal and MQUAL columns using real data."""

    # Define paths to the test resources
    resources_dir = Path(__file__).parent.parent / "resources"
    positive_path = resources_dir / "416119_L7402.test.random_sample.featuremap.filtered.sample.parquet"
    negative_path = resources_dir / "416119_L7402.test.raw.featuremap.filtered.sample.parquet"
    stats_positive_path = resources_dir / "416119_L7402.test.random_sample.featuremap.stats_positive.json"
    stats_negative_path = resources_dir / "416119_L7402.test.random_sample.featuremap.stats_negative.json"
    stats_featuremap_path = resources_dir / "416119_L7402.test.raw.featuremap.stats.json"
    interval_list_path = resources_dir / "wgs_calling_regions.without_encode_blacklist.hg38.interval_list.gz"
    model_fold_0_path = resources_dir / "416119_L7402.test.model_fold_0.json"
    model_fold_1_path = resources_dir / "416119_L7402.test.model_fold_1.json"

    # Create args using real file paths
    # Use the exact features from the trained model metadata
    features_str = (
        "REF:ALT:X_PREV1:X_NEXT1:X_PREV2:X_NEXT2:X_PREV3:X_NEXT3:X_HMER_REF:X_HMER_ALT:"
        "BCSQ:BCSQCSS:RL:INDEX:REV:SCST:SCED:SMQ_BEFORE:SMQ_AFTER:tm:rq:st:et:EDIST:HAMDIST:HAMDIST_FILT"
    )

    args = SimpleNamespace(
        positive=str(positive_path),
        negative=str(negative_path),
        stats_positive=str(stats_positive_path),
        stats_negative=str(stats_negative_path),
        stats_featuremap=str(stats_featuremap_path),
        aligned_bases=1000000,
        mean_coverage=30.0,
        training_regions=str(interval_list_path),
        k_folds=2,
        model_params=None,
        output=str(tmp_path),
        basename="test_",
        features=features_str,
        random_seed=42,
        verbose=False,
        max_qual=100.0,
        quality_lut_size=1000,
        metadata=None,
        use_kde_smoothing=False,
    )

    # Create trainer and load the data
    trainer = SRSNVTrainer(args)

    # Load the pre-trained models as XGBClassifier objects
    trainer.models = []
    for fold_idx in range(trainer.k_folds):
        # Create an XGBClassifier and load the pre-trained model
        model = xgb.XGBClassifier()
        # Load the model from file into the classifier
        if fold_idx == 0:
            model.load_model(str(model_fold_0_path))
        else:
            model.load_model(str(model_fold_1_path))
        trainer.models.append(model)

    # Remove existing recalibration columns if they exist to test their creation
    recal_columns_to_remove = [PROB_ORIG, PROB_RECAL, MQUAL, "prob_rescaled", "prob_fold_0", "prob_fold_1", "SNVQ"]
    existing_columns = trainer.data_frame.columns
    columns_to_drop = [col for col in recal_columns_to_remove if col in existing_columns]
    if columns_to_drop:
        trainer.data_frame = trainer.data_frame.drop(columns_to_drop)

    # Get initial columns before training
    initial_cols = set(trainer.data_frame.columns)

    # Since we have pre-trained models, we can skip the actual training
    # and directly call _add_quality_columns to test the recalibration process
    feat_cols = trainer._feature_columns()
    pd_df = trainer.data_frame.to_pandas()

    # Convert categorical columns to categories
    for col in feat_cols:
        if pd_df[col].dtype == object:
            pd_df[col] = pd_df[col].astype("category")

    # Extract metadata after categorical conversion
    trainer._extract_categorical_encodings(pd_df, feat_cols)
    trainer._extract_feature_dtypes(pd_df, feat_cols)

    fold_arr = pd_df[FOLD_COL].to_numpy()
    y_all = pd_df[LABEL_COL].to_numpy()

    # Call _add_quality_columns directly to test recalibration
    trainer._add_quality_columns(pd_df[feat_cols], fold_arr, y_all)  # Check that the recalibration columns were added
    final_cols = set(trainer.data_frame.columns)
    new_cols = final_cols - initial_cols

    # Verify that the required recalibration columns were added
    assert PROB_ORIG in final_cols, f"Missing {PROB_ORIG} column"
    assert PROB_RECAL in final_cols, f"Missing {PROB_RECAL} column"
    assert MQUAL in final_cols, f"Missing {MQUAL} column"

    # Check that the columns contain reasonable values
    trainer_df = trainer.data_frame
    assert trainer_df[PROB_ORIG].min() >= 0.0, "prob_orig should be >= 0"
    assert trainer_df[PROB_ORIG].max() <= 1.0, "prob_orig should be <= 1"
    assert trainer_df[PROB_RECAL].min() >= 0.0, "prob_recal should be >= 0"
    assert trainer_df[PROB_RECAL].max() <= 1.0, "prob_recal should be <= 1"
    assert trainer_df[MQUAL].min() >= 0.0, "MQUAL should be >= 0"

    print(f"Successfully added recalibration columns: {new_cols}")
    print(f"Final dataframe shape: {trainer_df.shape}")
    print(f"PROB_ORIG range: [{trainer_df[PROB_ORIG].min():.4f}, {trainer_df[PROB_ORIG].max():.4f}]")
    print(f"PROB_RECAL range: [{trainer_df[PROB_RECAL].min():.4f}, {trainer_df[PROB_RECAL].max():.4f}]")
    print(f"MQUAL range: [{trainer_df[MQUAL].min():.4f}, {trainer_df[MQUAL].max():.4f}]")


def test_recalibration_columns_kde(tmp_path):
    """Trainer should append prob_orig, prob_recal and MQUAL columns using real data."""

    # Define paths to the test resources
    resources_dir = Path(__file__).parent.parent / "resources"
    positive_path = resources_dir / "416119_L7402.test.random_sample.featuremap.filtered.sample.parquet"
    negative_path = resources_dir / "416119_L7402.test.raw.featuremap.filtered.sample.parquet"
    stats_positive_path = resources_dir / "416119_L7402.test.random_sample.featuremap.stats_positive.json"
    stats_negative_path = resources_dir / "416119_L7402.test.random_sample.featuremap.stats_negative.json"
    stats_featuremap_path = resources_dir / "416119_L7402.test.raw.featuremap.stats.json"
    interval_list_path = resources_dir / "wgs_calling_regions.without_encode_blacklist.hg38.interval_list.gz"
    model_fold_0_path = resources_dir / "416119_L7402.test.model_fold_0.json"
    model_fold_1_path = resources_dir / "416119_L7402.test.model_fold_1.json"

    # Create args using real file paths
    # Use the exact features from the trained model metadata
    features_str = (
        "REF:ALT:X_PREV1:X_NEXT1:X_PREV2:X_NEXT2:X_PREV3:X_NEXT3:X_HMER_REF:X_HMER_ALT:"
        "BCSQ:BCSQCSS:RL:INDEX:REV:SCST:SCED:SMQ_BEFORE:SMQ_AFTER:tm:rq:st:et:EDIST:HAMDIST:HAMDIST_FILT"
    )

    args = SimpleNamespace(
        positive=str(positive_path),
        negative=str(negative_path),
        stats_positive=str(stats_positive_path),
        stats_negative=str(stats_negative_path),
        stats_featuremap=str(stats_featuremap_path),
        aligned_bases=1000000,
        mean_coverage=30.0,
        training_regions=str(interval_list_path),
        k_folds=2,
        model_params=None,
        output=str(tmp_path),
        basename="test_",
        features=features_str,
        random_seed=42,
        verbose=False,
        max_qual=100.0,
        quality_lut_size=1000,
        metadata=None,
        use_kde_smoothing=True,
        use_gpu=False,
        use_float64=False,
    )

    # Create trainer and load the data
    trainer = SRSNVTrainer(args)

    # Load the pre-trained models as XGBClassifier objects
    trainer.models = []
    for fold_idx in range(trainer.k_folds):
        # Create an XGBClassifier and load the pre-trained model
        model = xgb.XGBClassifier()
        # Load the model from file into the classifier
        if fold_idx == 0:
            model.load_model(str(model_fold_0_path))
        else:
            model.load_model(str(model_fold_1_path))
        trainer.models.append(model)

    # Remove existing recalibration columns if they exist to test their creation
    recal_columns_to_remove = [PROB_ORIG, PROB_RECAL, MQUAL, "prob_rescaled", "prob_fold_0", "prob_fold_1", "SNVQ"]
    existing_columns = trainer.data_frame.columns
    columns_to_drop = [col for col in recal_columns_to_remove if col in existing_columns]
    if columns_to_drop:
        trainer.data_frame = trainer.data_frame.drop(columns_to_drop)

    # Get initial columns before training
    initial_cols = set(trainer.data_frame.columns)

    # Since we have pre-trained models, we can skip the actual training
    # and directly call _add_quality_columns to test the recalibration process
    feat_cols = trainer._feature_columns()
    pd_df = trainer.data_frame.to_pandas()

    # Convert categorical columns to categories
    for col in feat_cols:
        if pd_df[col].dtype == object:
            pd_df[col] = pd_df[col].astype("category")

    # Extract metadata after categorical conversion
    trainer._extract_categorical_encodings(pd_df, feat_cols)
    trainer._extract_feature_dtypes(pd_df, feat_cols)

    fold_arr = pd_df[FOLD_COL].to_numpy()
    y_all = pd_df[LABEL_COL].to_numpy()

    # Call _add_quality_columns directly to test recalibration
    trainer._add_quality_columns(pd_df[feat_cols], fold_arr, y_all)  # Check that the recalibration columns were added
    final_cols = set(trainer.data_frame.columns)
    new_cols = final_cols - initial_cols

    # Verify that the required recalibration columns were added
    assert PROB_ORIG in final_cols, f"Missing {PROB_ORIG} column"
    assert PROB_RECAL in final_cols, f"Missing {PROB_RECAL} column"
    assert MQUAL in final_cols, f"Missing {MQUAL} column"

    # Check that the columns contain reasonable values
    trainer_df = trainer.data_frame
    assert trainer_df[PROB_ORIG].min() >= 0.0, "prob_orig should be >= 0"
    assert trainer_df[PROB_ORIG].max() <= 1.0, "prob_orig should be <= 1"
    assert trainer_df[PROB_RECAL].min() >= 0.0, "prob_recal should be >= 0"
    assert trainer_df[PROB_RECAL].max() <= 1.0, "prob_recal should be <= 1"
    assert trainer_df[MQUAL].min() >= 0.0, "MQUAL should be >= 0"

    print(f"Successfully added recalibration columns: {new_cols}")
    print(f"Final dataframe shape: {trainer_df.shape}")
    print(f"PROB_ORIG range: [{trainer_df[PROB_ORIG].min():.4f}, {trainer_df[PROB_ORIG].max():.4f}]")
    print(f"PROB_RECAL range: [{trainer_df[PROB_RECAL].min():.4f}, {trainer_df[PROB_RECAL].max():.4f}]")
    print(f"MQUAL range: [{trainer_df[MQUAL].min():.4f}, {trainer_df[MQUAL].max():.4f}]")
