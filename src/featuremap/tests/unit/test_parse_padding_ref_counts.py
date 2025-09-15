import pandas as pd
import pytest
from ugbio_featuremap.somatic_pileup_featuremap_inference import parse_padding_ref_counts


def test_parse_padding_ref_counts_basic():
    # Create a DataFrame with ref_counts_pm_2 and nonref_counts_pm_2 columns
    df_counts = pd.DataFrame(
        {
            "ref_counts_pm_2": [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]],
            "nonref_counts_pm_2": [[0, 1, 0, 1, 0], [1, 0, 1, 0, 1]],
            "other_col": [10, 20],
        }
    )
    result = parse_padding_ref_counts(df_counts.copy())
    # Check that new columns are created
    expected_ref_cols = ["ref_m2", "ref_m1", "ref_0", "ref_1", "ref_2"]
    expected_nonref_cols = ["nonref_m2", "nonref_m1", "nonref_0", "nonref_1", "nonref_2"]
    for col in expected_ref_cols + expected_nonref_cols:
        assert col in result.columns
    # Check values
    assert (result.loc[0, expected_ref_cols] == [1, 2, 3, 4, 5]).all()
    assert (result.loc[1, expected_nonref_cols] == [1, 0, 1, 0, 1]).all()


def test_parse_padding_ref_counts_inconsistent_distance():
    df_counts = pd.DataFrame(
        {
            "ref_counts_pm_2": [[1, 2, 3, 4, 5]],
            "nonref_counts_pm_3": [[0, 1, 2, 3, 4, 5, 6]],
        }
    )
    with pytest.raises(ValueError, match="Inconsistent padding distances"):
        parse_padding_ref_counts(df_counts)


def test_parse_padding_ref_counts_no_padding():
    df_counts = pd.DataFrame({"foo": [1, 2]})
    with pytest.raises(ValueError, match="No padding ref count columns"):
        parse_padding_ref_counts(df_counts)
