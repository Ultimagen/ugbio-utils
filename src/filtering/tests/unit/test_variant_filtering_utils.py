import pickle
from os.path import join as pjoin
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import ugbio_filtering.blacklist
from ugbio_filtering import variant_filtering_utils


@pytest.fixture
def resources_dir():
    return Path(__file__).parent.parent / "resources"


def test_blacklist_cg_insertions():
    rows = pd.DataFrame(
        {
            "alleles": [
                ("C", "T"),
                ("CCG", "C"),
                (
                    "G",
                    "GGC",
                ),
            ],
            "filter": ["PASS", "PASS", "PASS"],
        }
    )
    rows = ugbio_filtering.blacklist.blacklist_cg_insertions(rows)
    filters = list(rows)
    assert filters == ["PASS", "CG_NON_HMER_INDEL", "CG_NON_HMER_INDEL"]


def test_merge_blacklist():
    list1 = pd.Series(["PASS", "FAIL", "FAIL"])
    list2 = pd.Series(["PASS", "FAIL1", "PASS"])
    merge_list = list(ugbio_filtering.blacklist.merge_blacklists([list1, list2]))
    assert merge_list == ["PASS;PASS", "FAIL;FAIL1", "FAIL;PASS"]


def test_read_blacklist(resources_dir):
    blacklist = pickle.load(open(pjoin(resources_dir, "blacklist.test.pkl"), "rb"))
    descriptions = [str(x) for x in blacklist]
    assert descriptions == [
        "ILLUMINA_FP: GTR error with 1000 elements",
        "COHORT_FP: HMER indel FP seen in cohort with 1000 elements",
    ]


def test_apply_blacklist(resources_dir):
    test_df = pd.read_hdf(pjoin(resources_dir, "test.df.h5"), key="variants")
    blacklist = pickle.load(open(pjoin(resources_dir, "blacklist.test.pkl"), "rb"))
    blacklists_applied = [x.apply(test_df) for x in blacklist]
    vcs = [x.value_counts() for x in blacklists_applied]
    assert (
        vcs[0]["PASS"] == 6597 and vcs[0]["ILLUMINA_FP"] == 6 and vcs[1]["PASS"] == 6477 and vcs[1]["COHORT_FP"] == 126
    )


def test_validate_data():
    test1 = np.array([[0, 1], [1, 2]])
    variant_filtering_utils._validate_data(test1)
    test2 = np.array([[0, 1], [1, np.nan]])
    with pytest.raises(AssertionError):
        variant_filtering_utils._validate_data(test2)
    test1 = pd.DataFrame([[0, 1], [1, 2]])
    variant_filtering_utils._validate_data(test1)
    test2 = pd.DataFrame([[0, 1], [1, np.nan]])
    with pytest.raises(AssertionError):
        variant_filtering_utils._validate_data(test2)
    test1 = pd.Series([0, 1])
    variant_filtering_utils._validate_data(test1)
    test2 = pd.Series([1, np.nan])
    with pytest.raises(AssertionError):
        variant_filtering_utils._validate_data(test2)
