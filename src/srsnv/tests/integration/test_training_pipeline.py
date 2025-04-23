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
    Run SRSNVTrainer on the sample Parquet inputs and verify that the main
    outputs (dataframe + model JSON) are created.
    """
    # ------------------------------------------------------------------ paths
    resources = Path(__file__).parent.parent / "resources"
    pos_file = resources / "416119_L7402.random_sample.featuremap.filtered.sample.parquet"
    neg_file = resources / "416119_L7402.raw.featuremap.filtered.sample.parquet"
    pos_stats = resources / "416119_L7402.random_sample.featuremap.stats.json"
    neg_stats = resources / "416119_L7402.raw.featuremap.stats.json"

    assert pos_file.is_file(), "positive parquet missing"
    assert neg_file.is_file(), "negative parquet missing"

    # ---------------------------------------------------------------- regions
    # Use the real hg38 calling regions shipped with the repository
    interval_list = resources.parent / "resources" / "wgs_calling_regions.without_encode_blacklist.hg38.interval_list"
    assert interval_list.is_file(), "interval_list fixture missing"

    # ---------------------------------------------------------------- args
    # use env-provided directory if defined, otherwise fall back to tmp_path
    out_dir = Path(os.getenv("SRSNV_TEST_OUTPUT_DIR", str(tmp_path)))

    args = argparse.Namespace(
        positive=str(pos_file),
        negative=str(neg_file),
        stats_positive=str(pos_stats),
        stats_negative=str(neg_stats),
        aligned_bases=134329535408,
        training_regions=str(interval_list),
        k_folds=2,
        model_params="n_estimators=2:max_depth=2:enable_categorical=true",  # keep test fast
        features="REF:ALT:X_HMER_REF:X_HMER_ALT:X_PREV1:X_NEXT1:X_PREV2:X_NEXT2:X_PREV3:X_NEXT3:BCSQ:BCSQCSS:RL:INDEX:DUP:REV:"
        "SCST:SCED:MAPQ:EDIST:SMQ_BEFORE:SMQ_AFTER:tm:rq:st:et",
        output=str(out_dir),  # override tmp_path when env var is set
        basename="unit_test",
        random_seed=0,
        verbose=True,
        max_qual=100.0,
        quality_lut_size=1000,
    )

    # ---------------------------------------------------------------- train
    trainer = SRSNVTrainer(args)
    trainer.run()

    # ---------------------------------------------------------------- assert
    df_out = out_dir / "unit_test.featuremap_df.parquet"
    model_out = out_dir / "unit_test.model_fold_0.json"
    metadata_out = out_dir / "unit_test.srsnv_metadata.json"

    assert df_out.is_file(), "output dataframe not written"
    assert model_out.is_file(), "model JSON not written"
    assert metadata_out.is_file(), "metadata JSON not written"

    # quick sanity: dataframe has required prediction columns
    cols = pl.read_parquet(df_out).columns
    assert "MQUAL" in cols, "prediction column missing"

    # quick sanity: unified metadata format ------------------------------------------------------
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
