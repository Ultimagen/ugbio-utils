import argparse
import json
import os
from pathlib import Path

import polars as pl
import pytest
from ugbio_srsnv.srsnv_training import SRSNVTrainer


@pytest.mark.integration
def test_end_to_end_training(tmp_path: Path) -> None:
    """
    Run SRSNVTrainer on the sample inputs and verify that the main outputs are created.
    """
    # ------------------------------------------------------------------ paths
    resources = Path(__file__).parent.parent / "resources"
    pos_file = resources / "402572-CL10377.random_sample.featuremap.filtered.parquet"
    neg_file = resources / "402572-CL10377.raw.featuremap.filtered.parquet"
    stats_file = resources / "402572-CL10377.model_filters_status.funnel.edited.json"

    assert pos_file.is_file(), "positive parquet missing"
    assert neg_file.is_file(), "negative parquet missing"
    assert stats_file.is_file(), "stats file missing"

    # ---------------------------------------------------------------- regions
    # Use the real hg38 calling regions shipped with the repository
    bed_file = resources / "wgs_calling_regions.without_encode_blacklist.hg38.chr1_22.interval_list"
    assert bed_file.is_file(), "Interval list file fixture missing"

    # ---------------------------------------------------------------- args
    # use env-provided directory if defined, otherwise fall back to tmp_path
    out_dir = Path(os.getenv("SRSNV_TEST_OUTPUT_DIR", str(tmp_path)))

    args = argparse.Namespace(
        positive=str(pos_file),
        negative=str(neg_file),
        stats_file=str(stats_file),
        mean_coverage=30.0,
        training_regions=str(bed_file),
        k_folds=2,
        model_params=(
            'n_estimators=2:max_depth=2:enable_categorical=true:eval_metric=["auc","logloss"]'
        ),  # keep test fast
        features="REF:ALT:X_HMER_REF:X_HMER_ALT:X_PREV1:X_NEXT1:X_PREV2:X_NEXT2:X_PREV3:X_NEXT3:BCSQ:BCSQCSS:RL:INDEX:DUP:REV:"
        "SCST:SCED:MAPQ:EDIST:SMQ_BEFORE:SMQ_AFTER:tm:rq:st:et",
        output=str(out_dir),  # override tmp_path when env var is set
        basename="integration_test",
        random_seed=0,
        verbose=True,
        max_qual=100.0,
        quality_lut_size=1000,
        metadata=None,
        use_gpu=False,
        use_float32=False,
        use_kde_smoothing=True,
    )

    # ---------------------------------------------------------------- train
    trainer = SRSNVTrainer(args)
    trainer.run()

    # ---------------------------------------------------------------- assert
    df_out = out_dir / "integration_test.featuremap_df.parquet"
    model_out = out_dir / "integration_test.model_fold_0.json"
    metadata_out = out_dir / "integration_test.srsnv_metadata.json"

    assert df_out.is_file(), "output dataframe not written"
    assert model_out.is_file(), "model JSON not written"
    assert metadata_out.is_file(), "metadata JSON not written"

    # quick sanity: dataframe has required prediction columns
    cols = pl.read_parquet(df_out).columns
    assert "MQUAL" in cols, "prediction column missing"

    # quick sanity: unified metadata format
    with metadata_out.open() as fh:
        meta = json.load(fh)

    assert "chrom_to_model" in meta, "chrom_to_model missing in metadata"
    assert "features" in meta and isinstance(meta["features"], list), "features list missing"

    # new check: every mapping value should be a model filename
    for val in meta["chrom_to_model"].values():
        assert val.endswith(".json"), "chrom_to_model value is not a model filename"

    # ensure categorical feature encoding is present
    x_prev1 = next((f for f in meta["features"] if f["name"] == "X_PREV1"), None)
    assert x_prev1 is not None and x_prev1.get("type") == "c", "X_PREV1 categorical encoding missing"
    assert "values" in x_prev1 and isinstance(x_prev1["values"], dict), "categorical values missing"


@pytest.mark.integration
def test_trainer_initialization(tmp_path: Path) -> None:
    """
    Test that SRSNVTrainer can be initialized without running the full training pipeline.
    """
    resources = Path(__file__).parent.parent / "resources"
    pos_file = resources / "402572-CL10377.random_sample.featuremap.filtered.parquet"
    neg_file = resources / "402572-CL10377.raw.featuremap.filtered.parquet"
    stats_file = resources / "402572-CL10377.model_filters_status.funnel.edited.json"
    bed_file = resources / "wgs_calling_regions.without_encode_blacklist.hg38.chr1_22.interval_list"

    args = argparse.Namespace(
        positive=str(pos_file),
        negative=str(neg_file),
        stats_file=str(stats_file),
        mean_coverage=30.0,
        training_regions=str(bed_file),
        k_folds=2,
        model_params=None,
        output=str(tmp_path),
        basename="test_",
        features="REF:ALT:MAPQ",  # minimal feature set for faster test
        random_seed=42,
        verbose=False,
        max_qual=100.0,
        quality_lut_size=1000,
        metadata=None,
        use_gpu=False,
        use_float32=False,
        use_kde_smoothing=False,
    )

    # Create trainer and load the data
    trainer = SRSNVTrainer(args)

    # Verify that the trainer was created successfully
    assert trainer.pos_stats is not None, "Positive stats should be loaded"
    assert trainer.neg_stats is not None, "Negative stats should be loaded"

    # Verify that all stats have the expected structure
    for stats in [trainer.pos_stats, trainer.neg_stats]:
        assert "filters" in stats, "Stats should have 'filters' key"
        assert isinstance(stats["filters"], list), "Filters should be a list"
        assert len(stats["filters"]) > 0, "Should have at least one filter"

    # Verify that we can find specific filters
    pos_filters = {f["name"] for f in trainer.pos_stats["filters"]}
    neg_filters = {f["name"] for f in trainer.neg_stats["filters"]}

    assert "raw" in pos_filters, "Should have 'raw' filter in positive stats"
    assert "raw" in neg_filters, "Should have 'raw' filter in negative stats"
    assert "mapq_ge_60" in pos_filters, "Should have 'mapq_ge_60' filter in positive stats"
    assert "mapq_ge_60" in neg_filters, "Should have 'mapq_ge_60' filter in negative stats"

    # Test the prior calculation works
    assert hasattr(trainer, "prior_train_error"), "Should have calculated prior_train_error"
    assert 0 < trainer.prior_train_error < 1, "Prior should be a valid probability"
