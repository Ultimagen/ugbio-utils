from pathlib import Path
from types import SimpleNamespace

from ugbio_srsnv.srsnv_training import (
    SRSNVTrainer,
)


def test_recalibration_columns(tmp_path):
    """Trainer should load unified stats file and create SRSNVTrainer instance successfully."""

    # Define paths to the test resources
    resources_dir = Path(__file__).parent.parent / "resources"
    positive_path = resources_dir / "402572-CL10377.random_sample.featuremap.filtered.parquet"
    negative_path = resources_dir / "402572-CL10377.raw.featuremap.filtered.parquet"
    stats_file_path = resources_dir / "402572-CL10377.model_filters_status.funnel.edited.json"
    bed_file_path = resources_dir / "wgs_calling_regions.without_encode_blacklist.hg38.chr1_22.interval_list"

    # Create args using real file paths
    # Use the exact features from the trained model metadata
    features_str = (
        "REF:ALT:X_PREV1:X_NEXT1:X_PREV2:X_NEXT2:X_PREV3:X_NEXT3:X_HMER_REF:X_HMER_ALT:"
        "BCSQ:BCSQCSS:RL:INDEX:REV:SCST:SCED:SMQ_BEFORE:SMQ_AFTER:tm:rq:st:et:EDIST:HAMDIST:HAMDIST_FILT"
    )

    args = SimpleNamespace(
        positive=str(positive_path),
        negative=str(negative_path),
        stats_file=str(stats_file_path),
        mean_coverage=30.0,
        training_regions=str(bed_file_path),
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
    )

    # Create trainer and load the data - this tests the new format stats loading
    trainer = SRSNVTrainer(args)

    # Verify that the trainer was created successfully with the new format
    assert trainer.pos_stats is not None, "Positive stats should be loaded"
    assert trainer.neg_stats is not None, "Negative stats should be loaded"

    # Verify that stats have the expected structure
    assert "filters" in trainer.pos_stats, "Positive stats should have filters"
    assert "filters" in trainer.neg_stats, "Negative stats should have filters"

    # Verify that data was loaded
    assert trainer.data_frame is not None, "Data frame should be loaded"
    assert trainer.data_frame.height > 0, "Data frame should have rows"

    # Verify basic trainer attributes are set correctly
    assert trainer.mean_coverage == 30.0, "Mean coverage should be set"
    assert trainer.k_folds == 2, "K folds should be set"
    assert trainer.seed == 42, "Random seed should be set"

    print("✅ SRSNVTrainer successfully created with unified stats file")
    print(f"✅ Loaded {trainer.data_frame.height} rows of data")
    print(f"✅ Positive stats: {len(trainer.pos_stats['filters'])} filters")
    print(f"✅ Negative stats: {len(trainer.neg_stats['filters'])} filters")
