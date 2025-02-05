from os.path import join as pjoin
from pathlib import Path

import pandas as pd
import pytest
from ugbio_comparison.comparison_utils import (
    _fix_errors,
    bed_file_length,
    close_to_hmer_run,
    vcf2concordance,
)
from ugbio_core.h5_utils import read_hdf
from ugbio_core.vcfbed import vcftools


@pytest.fixture
def resources_dir():
    return Path(__file__).parent.parent / "resources"


def test_fix_errors(resources_dir):
    data = read_hdf(pjoin(resources_dir, "h5_file_unitest.h5"), key="concordance")
    fixed_df = _fix_errors(data)
    assert all(
        fixed_df[((fixed_df["call"] == "TP") & ((fixed_df["base"] == "TP") | (fixed_df["base"].isna())))][
            "gt_ground_truth"
        ].eq(
            fixed_df[(fixed_df["call"] == "TP") & ((fixed_df["base"] == "TP") | (fixed_df["base"].isna()))]["gt_ultima"]
        )
    )

    # (None, TP) (None,FN_CA)
    pd.set_option("display.max_columns", None)
    assert fixed_df[(fixed_df["call"].isna()) & ((fixed_df["base"] == "TP") | (fixed_df["base"] == "FN_CA"))].size == 20
    # (FP_CA,FN_CA), (FP_CA,None)
    temp_df = fixed_df.loc[
        (fixed_df["call"] == "FP_CA") & ((fixed_df["base"] == "FN_CA") | (fixed_df["base"].isna())),
        ["gt_ultima", "gt_ground_truth"],
    ]
    assert all(
        temp_df.apply(
            lambda x: ((x["gt_ultima"][0] == x["gt_ground_truth"][0]) & (x["gt_ultima"][1] != x["gt_ground_truth"][1]))
            | ((x["gt_ultima"][1] == x["gt_ground_truth"][1]) & (x["gt_ultima"][0] != x["gt_ground_truth"][0]))
            | ((x["gt_ultima"][0] == x["gt_ground_truth"][1]) & (x["gt_ultima"][1] != x["gt_ground_truth"][0]))
            | ((x["gt_ultima"][1] == x["gt_ground_truth"][0]) & (x["gt_ultima"][0] != x["gt_ground_truth"][1])),
            axis=1,
        )
    )


class TestVCF2Concordance:
    def test_qual_not_nan(self, resources_dir):
        input_vcf = pjoin(resources_dir, "chr2.vcf.gz")
        concordance_vcf = pjoin(resources_dir, "chr2.conc.vcf.gz")
        result = vcf2concordance(input_vcf, concordance_vcf)
        assert pd.isna(result.query("classify!='fn'").qual).sum() == 0
        assert pd.isna(result.query("classify!='fn'").sor).sum() == 0

    def test_filtered_out_missing(self, resources_dir):
        input_vcf = pjoin(resources_dir, "hg002.vcf.gz")
        concordance_vcf = pjoin(resources_dir, "hg002.conc.vcf.gz")
        result = vcf2concordance(input_vcf, concordance_vcf)
        assert ((result["call"] == "IGN") & (pd.isna(result["base"]))).sum() == 0

    def test_filtered_out_tp_became_fn(self, resources_dir):
        input_vcf = pjoin(resources_dir, "hg002.vcf.gz")
        concordance_vcf = pjoin(resources_dir, "hg002.conc.vcf.gz")
        result = vcf2concordance(input_vcf, concordance_vcf)
        take = result[(result["call"] == "IGN") & (result["base"] == "FN")]
        assert take.shape[0] > 0
        assert (take["classify"] == "fn").all()

    def test_excluded_regions_are_ignored(self, resources_dir):
        input_vcf = pjoin(resources_dir, "hg002.excluded.vcf.gz")
        concordance_vcf = pjoin(resources_dir, "hg002.excluded.conc.vcf.gz")
        result = vcf2concordance(input_vcf, concordance_vcf)
        assert (result["call"] == "OUT").sum() == 0
        assert (result["base"] == "OUT").sum() == 0

    def test_all_ref_never_false_negative(self, resources_dir):
        input_vcf = pjoin(resources_dir, "hg002.allref.vcf.gz")
        concordance_vcf = pjoin(resources_dir, "hg002.allref.conc.vcf.gz")
        result = vcf2concordance(input_vcf, concordance_vcf)
        calls = result[result["gt_ground_truth"] == (0, 0)].classify_gt.value_counts()
        assert "fn" not in calls.index


def test_bed_file_length(resources_dir):
    bed1 = pjoin(resources_dir, "bed1.bed")
    result = bed_file_length(bed1)
    assert result == 3026


def test_close_to_hmer_run(resources_dir):
    input_vcf = vcftools.get_vcf_df(pjoin(resources_dir, "hg19.vcf.gz"))
    runs_file = pjoin(resources_dir, "runs.hg19.bed")
    result = close_to_hmer_run(input_vcf, runs_file)
    assert result["close_to_hmer_run"].sum() == 76
