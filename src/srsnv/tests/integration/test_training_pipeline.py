import argparse
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

    assert pos_file.is_file(), "positive parquet missing"
    assert neg_file.is_file(), "negative parquet missing"

    # ---------------------------------------------------------------- regions
    # Use the real hg38 calling regions shipped with the repository
    interval_list = resources.parent / "resources" / "wgs_calling_regions.without_encode_blacklist.hg38.interval_list"
    assert interval_list.is_file(), "interval_list fixture missing"

    # ---------------------------------------------------------------- args
    args = argparse.Namespace(
        positive=str(pos_file),
        negative=str(neg_file),
        training_regions=str(interval_list),
        k_folds=2,
        model_params="n_estimators=2:max_depth=2:enable_categorical=true",  # keep test fast
        features="X_HMER_REF:X_HMER_ALT:X_PREV1:X_NEXT1:X_PREV2:X_NEXT2:X_PREV3:X_NEXT3:BCSQ:BCSQCSS:RL:INDEX:DUP:REV:"
        "SCST:SCED:MAPQ:EDIST:SMQ_BEFORE:SMQ_AFTER:tm:rq:st:et",
        output=str(tmp_path),
        basename="unit_test",
        random_seed=0,
        verbose=True,
        max_qual=100.0,
    )

    # ---------------------------------------------------------------- train
    trainer = SRSNVTrainer(args)
    trainer.run()

    # ---------------------------------------------------------------- assert
    df_out = tmp_path / "unit_test.featuremap_df.parquet"
    model_out = tmp_path / "unit_test.model_fold_0.json"
    chrom_map = tmp_path / "unit_test.chrom_to_model.json"

    assert df_out.is_file(), "output dataframe not written"
    assert model_out.is_file(), "model JSON not written"
    assert chrom_map.is_file(), "chromosome→model map not written"

    # quick sanity: dataframe has required prediction columns
    cols = pl.read_parquet(df_out).columns
    assert "raw_qual_val" in cols, "prediction column missing"
    # quick sanity: dataframe has required prediction columns
    cols = pl.read_parquet(df_out).columns
    assert "raw_qual_val" in cols, "prediction column missing"
