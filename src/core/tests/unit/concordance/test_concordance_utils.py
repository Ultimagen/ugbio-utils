import unittest

import numpy as np
import pandas as pd
import pytest
from ugbio_core.concordance.concordance_utils import (
    _get_empty_recall_precision,
    _get_empty_recall_precision_curve,
    calc_accuracy_metrics,
    calc_recall_precision_curve,
    get_concordance_metrics,
)
from ugbio_core.stats_utils import get_f1


class TestConcordanceUtils(unittest.TestCase):
    def test_calc_accuracy_metrics(self):
        """
        given concordance dataframe with all rows passing filter, gather expected metrics for expected variant type
        """
        concordance_df = pd.DataFrame(
            {
                "classify": ["tp", "tp", "fp", "fn", "tp"],
                "filter": ["PASS"] * 5,
                "tree_score": [1] * 5,
                "hmer_indel_nuc": ["N"] * 5,
                "indel": [True] * 5,
                "hmer_indel_length": [2] * 5,
            }
        )
        accuracy_df = calc_accuracy_metrics(concordance_df, "classify")
        expected = {
            "initial_tp": [3],
            "initial_fp": [1],
            "initial_fn": [1],
            "initial_precision": [0.75],
            "initial_recall": [0.75],
            "initial_f1": [0.75],
            "tp": [3],
            "fp": [1],
            "fn": [1],
            "precision": [0.75],
            "recall": [0.75],
            "f1": [0.75],
        }
        # DataFrame dict contains index->value dictionaries per each column
        expected_indels = {"group": {7: "INDELS"}}
        expected_hmer_indel_lt_4 = {"group": {2: "HMER indel <= 4"}}
        for expected_key, expected_value in expected.items():
            expected_hmer_indel_lt_4[expected_key] = {2: expected_value[0]}
            expected_indels[expected_key] = {7: expected_value[0]}

        self.assertEqual(
            expected_hmer_indel_lt_4,
            accuracy_df[accuracy_df["group"] == "HMER indel <= 4"].to_dict(),
        )
        self.assertEqual(expected_indels, accuracy_df[accuracy_df["group"] == "INDELS"].to_dict())

    def test_calc_accuracy_metrics_with_non_passing_rows(self):
        """
        given concordance dataframe with some rows failing filter, gather expected metrics for expected variant type
        filtered fp should raise precision
        filtered fn should have no effect
        """
        concordance_df = pd.DataFrame(
            {
                "classify": ["tp", "tp", "fp", "fn", "tp", "tn"],
                "filter": ["PASS", "PASS", "SEC", "LOW_SCORE", "PASS", "LOW_SCORE"],
                "tree_score": [1] * 6,
                "hmer_indel_nuc": ["N"] * 6,
                "indel": [False] * 6,
                "hmer_indel_length": [None] * 6,
            }
        )
        accuracy_df = calc_accuracy_metrics(concordance_df, "classify")

        expected = {
            "initial_tp": [3],
            "initial_fp": [2],
            "initial_fn": [1],
            "initial_precision": [0.6],
            "initial_recall": [0.75],
            "initial_f1": [0.66667],
            "tp": [3],
            "fp": [0],
            "fn": [1],
            "precision": [1],
            "recall": [0.75],
            "f1": [0.85714],
        }

        # DataFrame dict contains index->value dictionaries per each column
        expected_snps = {"group": {0: "SNP"}}

        for expected_key, expected_value in expected.items():
            expected_snps[expected_key] = {0: expected_value[0]}

        self.assertEqual(expected_snps, accuracy_df[accuracy_df["group"] == "SNP"].to_dict())

    def test_calc_recall_precision_curve(self):
        """
        given concordance dataframe with all rows passing filter, calc recall/precision curve
        """
        n_tp = 50
        tp_range = (0.5, 1)
        n_fp = 50
        fp_range = (0, 0.49)
        n_fn = 20
        fn_score = -1
        concordance_df = pd.DataFrame(
            {
                "classify": ["tp"] * n_tp + ["fp"] * n_fp + ["fn"] * n_fn,
                "filter": ["PASS"] * (n_tp + n_fp + n_fn),
                "tree_score": np.concatenate(
                    (
                        np.linspace(tp_range[0], tp_range[1], n_tp),
                        np.linspace(fp_range[0], fp_range[1], n_fp),
                        fn_score * np.ones(n_fn),
                    )
                ),
                "hmer_indel_nuc": ["N"] * (n_tp + n_fp + n_fn),
                "indel": [True] * (n_fp + n_tp + n_fn),
                "hmer_indel_length": [2] * (n_fp + n_tp + n_fn),
            }
        )
        accuracy_df = calc_recall_precision_curve(concordance_df, "classify")

        def safemax(lst):
            return max(lst) if len(lst) > 0 else np.nan

        for col in ["precision", "recall", "f1"]:
            accuracy_df[col] = accuracy_df[col].apply(safemax)
        accuracy_df = accuracy_df.drop("predictions", axis=1)
        accuracy_df = accuracy_df.round(5)

        expected = {
            "precision": 1.0,
            "recall": np.round(n_tp / (n_tp + n_fn), 5),
            "f1": np.round(get_f1(n_tp / (n_tp + n_fn), 1.0), 5),
            "threshold": 0.5,
        }
        # DataFrame dict contains index->value dictionaries per each column
        expected_indels = {"group": {7: "INDELS"}}
        expected_hmer_indel_lt_4 = {"group": {2: "HMER indel <= 4"}}
        for expected_key, expected_value in expected.items():
            expected_hmer_indel_lt_4[expected_key] = {2: expected_value}
            expected_indels[expected_key] = {7: expected_value}

        self.assertEqual(
            expected_hmer_indel_lt_4,
            accuracy_df[accuracy_df["group"] == "HMER indel <= 4"].to_dict(),
        )
        self.assertEqual(expected_indels, accuracy_df[accuracy_df["group"] == "INDELS"].to_dict())


def test_get_empty_recall_precision():
    result = _get_empty_recall_precision()
    expected_result = {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "initial_tp": 0,
        "initial_fp": 0,
        "initial_fn": 0,
        "initial_precision": 1.0,
        "initial_recall": 1.0,
        "initial_f1": 1.0,
    }
    assert result == expected_result


def test_get_empty_recall_precision_curve():
    result = _get_empty_recall_precision_curve()
    expected_result = {
        "threshold": 0,
        "predictions": [],
        "precision": [],
        "recall": [],
        "f1": [],
    }
    assert result == expected_result


def test_get_concordance_metrics():
    predictions = np.array([0, 1, 0, 1, 0])
    scores = np.array([0.2, 0.8, 0.4, 0.6, 0.3])
    truth = np.array([0, 1, 1, 0, 1])
    fn_mask = np.array([False, False, True, False, False])

    result = get_concordance_metrics(predictions, scores, truth, fn_mask, return_metrics=True, return_curves=False)

    expected_metrics_df = pd.DataFrame(
        {
            "tp": [1],
            "fp": [1],
            "fn": [2],
            "precision": [0.5],
            "recall": [0.3333333],
            "f1": [0.4],
            "initial_tp": [2],
            "initial_fp": [2],
            "initial_fn": [1],
            "initial_precision": [0.5],
            "initial_recall": [0.6666666666666666],
            "initial_f1": [0.5714285714285714],
        }
    )

    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_frame_equal(result, expected_metrics_df)


def test_get_concordance_metrics_no_return():
    predictions = np.array([0, 1, 0, 1, 0])
    scores = np.array([0.2, 0.8, 0.4, 0.6, 0.3])
    truth = np.array([0, 1, 1, 0, 1])
    fn_mask = np.array([False, False, True, False, False])

    with pytest.raises(AssertionError):
        get_concordance_metrics(predictions, scores, truth, fn_mask, return_metrics=False, return_curves=False)
