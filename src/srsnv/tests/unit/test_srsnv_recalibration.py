from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
from ugbio_srsnv.srsnv_training import (
    MQUAL,
    PROB_ORIG,
    PROB_RECAL,
    SRSNVTrainer,
)


class _DummyModel:
    """Minimal drop-in replacement for XGBClassifier used in tests."""

    def predict_proba(self, x):
        # Return a constant probability for class-1
        return np.tile([0.7, 0.3], (len(x), 1))

    def fit(self, *_, **__):
        pass  # no-op


def _mock_load_data(self, *_):
    """Return a small dataframe with two folds and one numeric feature."""
    dummy_df = pl.DataFrame(
        {
            "CHROM": ["chr1", "chr1", "chr2", "chr2"],
            "POS": [1, 2, 3, 4],
            "feat1": [0.1, 0.2, 0.3, 0.4],
            "label": [True, False, True, False],
            "fold_id": [0, 0, 1, 1],
        }
    )
    return dummy_df


def test_recalibration_columns(tmp_path):
    """Trainer should append prob_orig, prob_recal and MQUAL columns."""
    # ---- patch loader -------------------------------------------------
    SRSNVTrainer._load_data = _mock_load_data

    args = SimpleNamespace(
        positive="",
        negative="",
        training_regions=str(tmp_path / "r.interval_list"),
        k_folds=2,
        model_params=None,
        output=str(tmp_path),
        basename="",
        features="",
        random_seed=42,
        verbose=False,
        max_qual=100.0,
    )

    # minimal interval-list file
    Path(args.training_regions).write_text(
        "@SQ\tSN:chr1\tLN:10\n@SQ\tSN:chr2\tLN:10\nchr1\t1\t10\t+\tregion\nchr2\t1\t10\t+\tregion\n"
    )

    trainer = SRSNVTrainer(args)
    trainer.models = [_DummyModel() for _ in range(trainer.k_folds)]
    trainer.train()  # will call _add_quality_columns internally

    cols = set(trainer.data_frame.columns)
    assert {PROB_ORIG, PROB_RECAL, MQUAL}.issubset(cols), "Recalibration columns missing"
