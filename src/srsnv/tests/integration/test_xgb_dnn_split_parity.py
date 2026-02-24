import argparse
from pathlib import Path

import polars as pl
import pytest
from ugbio_srsnv.split_manifest import build_single_model_read_hash_manifest, build_split_manifest, save_split_manifest
from ugbio_srsnv.srsnv_training import FOLD_COL, SPLIT_ROLE_COL, SRSNVTrainer


@pytest.mark.integration
def test_srsnv_training_uses_manifest_mapping(monkeypatch, tmp_path: Path) -> None:
    resources = Path(__file__).parent.parent / "resources"
    pos_file = resources / "416119_L7402.test.random_sample.featuremap.filtered.sample.parquet"
    neg_file = resources / "416119_L7402.test.raw.featuremap.filtered.sample.parquet"
    pos_stats = resources / "416119_L7402.test.random_sample.featuremap.stats_positive.json"
    neg_stats = resources / "416119_L7402.test.random_sample.featuremap.stats_negative.json"
    raw_stats = resources / "416119_L7402.test.raw.featuremap.stats.json"
    interval_list = resources / "wgs_calling_regions.without_encode_blacklist.hg38.interval_list.gz"

    manifest = build_split_manifest(
        training_regions=str(interval_list),
        k_folds=2,
        random_seed=123,
        holdout_chromosomes=["chr21", "chr22"],
    )
    manifest_path = tmp_path / "split_manifest.json"
    save_split_manifest(manifest, manifest_path)

    args = argparse.Namespace(
        positive=str(pos_file),
        negative=str(neg_file),
        stats_positive=str(pos_stats),
        stats_negative=str(neg_stats),
        stats_featuremap=str(raw_stats),
        aligned_bases=134329535408,
        mean_coverage=30.0,
        training_regions=str(interval_list),
        k_folds=2,
        split_manifest_in=str(manifest_path),
        split_manifest_out=None,
        holdout_chromosomes=None,
        model_params='n_estimators=2:max_depth=2:enable_categorical=true:eval_metric=["auc","logloss"]',
        features="REF:ALT:X_HMER_REF:X_HMER_ALT:X_PREV1:X_NEXT1:X_PREV2:X_NEXT2:X_PREV3:X_NEXT3:BCSQ:BCSQCSS:RL:INDEX:DUP:REV:SCST:SCED:MAPQ:EDIST:SMQ_BEFORE:SMQ_AFTER:tm:rq:st:et",
        output=str(tmp_path),
        basename="parity_test",
        random_seed=0,
        verbose=False,
        max_qual=100.0,
        quality_lut_size=1000,
        metadata=None,
        use_kde_smoothing=False,
        use_gpu=False,
        use_float32=False,
    )

    def _fake_load_data(_self, _pos, _neg):
        return pl.DataFrame(
            {
                "CHROM": ["chr1", "chr2", "chr21", "chr22"],
                "POS": [1, 2, 3, 4],
                "label": [True, False, True, False],
                "dummy_feature": [0.1, 0.2, 0.3, 0.4],
            }
        )

    monkeypatch.setattr(SRSNVTrainer, "_load_data", _fake_load_data)
    trainer = SRSNVTrainer(args)
    expected = {k: int(v) for k, v in manifest["chrom_to_fold"].items()}
    assert trainer.chrom_to_fold == expected


@pytest.mark.integration
def test_single_model_split_assignment(monkeypatch, tmp_path: Path) -> None:
    resources = Path(__file__).parent.parent / "resources"
    interval_list = resources / "wgs_calling_regions.without_encode_blacklist.hg38.interval_list.gz"
    manifest = build_single_model_read_hash_manifest(
        training_regions=str(interval_list),
        random_seed=123,
        holdout_chromosomes=["chr21", "chr22"],
        val_fraction=0.1,
        hash_key="RN",
    )
    manifest_path = tmp_path / "single_split_manifest.json"
    save_split_manifest(manifest, manifest_path)

    args = argparse.Namespace(
        positive="unused.parquet",
        negative="unused.parquet",
        stats_positive=str(resources / "416119_L7402.test.random_sample.featuremap.stats_positive.json"),
        stats_negative=str(resources / "416119_L7402.test.random_sample.featuremap.stats_negative.json"),
        stats_featuremap=str(resources / "416119_L7402.test.raw.featuremap.stats.json"),
        aligned_bases=134329535408,
        mean_coverage=30.0,
        training_regions=str(interval_list),
        k_folds=3,
        split_manifest_in=str(manifest_path),
        split_manifest_out=None,
        holdout_chromosomes="chr21,chr22",
        single_model_split=True,
        val_fraction=0.1,
        split_hash_key="RN",
        model_params='n_estimators=2:max_depth=2:enable_categorical=true:eval_metric=["auc","logloss"]',
        features="dummy_feature",
        output=str(tmp_path),
        basename="single_mode_parity_test",
        random_seed=0,
        verbose=False,
        max_qual=100.0,
        quality_lut_size=1000,
        metadata=None,
        use_kde_smoothing=False,
        use_gpu=False,
        use_float32=False,
    )

    def _fake_load_data(_self, _pos, _neg):
        return pl.DataFrame(
            {
                "CHROM": ["chr1", "chr1", "chr2", "chr2", "chr21", "chr22"],
                "POS": [1, 2, 3, 4, 5, 6],
                "RN": ["r1", "r2", "r3", "r4", "r5", "r6"],
                "label": [True, False, True, False, True, False],
                "dummy_feature": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            }
        )

    monkeypatch.setattr(SRSNVTrainer, "_load_data", _fake_load_data)
    trainer = SRSNVTrainer(args)
    result_df = trainer.data_frame
    assert result_df.filter(pl.col("CHROM").is_in(["chr21", "chr22"]))[SPLIT_ROLE_COL].to_list() == ["test", "test"]
    assert set(result_df[SPLIT_ROLE_COL].to_list()).issubset({"train", "val", "test"})
    assert result_df.filter(pl.col(SPLIT_ROLE_COL) == "train").height > 0
    test_subset = result_df.filter(pl.col(SPLIT_ROLE_COL) == "test")
    assert test_subset[FOLD_COL].null_count() == test_subset.height
