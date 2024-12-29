import shutil
from collections import Counter
from os.path import basename, exists
from os.path import join as pjoin
from pathlib import Path
from unittest import mock
from unittest.mock import patch

import pandas as pd
import pysam
import pytest
from simppl.simple_pipeline import SimplePipeline
from ugbio_comparison.vcf_pipeline_utils import (
    VcfPipelineUtils,
    _fix_errors,
    bed_file_length,
    close_to_hmer_run,
    vcf2concordance,
)
from ugbio_core.h5_utils import read_hdf
from ugbio_core.vcfbed import vcftools
from ugbio_core.vcfbed.interval_file import IntervalFile


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


def test_transform_hom_calls_to_het_calls(tmpdir, resources_dir):
    input_vcf = pjoin(resources_dir, "dv.input.vcf.gz")
    vpu = VcfPipelineUtils()
    shutil.copyfile(input_vcf, pjoin(tmpdir, basename(input_vcf)))
    expected_output_file = pjoin(tmpdir, basename(input_vcf).replace(".vcf.gz", ".rev.hom.ref.vcf.gz"))
    expected_output_index_file = pjoin(tmpdir, basename(input_vcf).replace(".vcf.gz", ".rev.hom.ref.vcf.gz.tbi"))

    vpu.transform_hom_calls_to_het_calls(pjoin(tmpdir, basename(input_vcf)), expected_output_file)
    assert exists(expected_output_file)
    assert exists(expected_output_index_file)
    input_df = vcftools.get_vcf_df(input_vcf)
    select = (input_df["filter"] != "PASS") & ((input_df["gt"] == (0, 0)) | (input_df["gt"] == (None, None)))
    assert select.sum() > 0
    input_df = vcftools.get_vcf_df(expected_output_file)
    select = (input_df["filter"] != "PASS") & ((input_df["gt"] == (0, 0)) | (input_df["gt"] == (None, None)))
    assert select.sum() == 0


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


class TestVCFEvalRun:
    def test_vcfeval_run_ignore_filter(self, tmp_path, resources_dir):
        ref_genome = pjoin(resources_dir, "sample.fasta")
        sample_calls = pjoin(resources_dir, "sample.sd.vcf.gz")
        truth_calls = pjoin(resources_dir, "gtr.sample.sd.vcf.gz")
        sp = SimplePipeline(0, 100)
        high_conf = IntervalFile(None, pjoin(resources_dir, "highconf.interval_list"))
        VcfPipelineUtils(sp).run_vcfeval_concordance(
            input_file=sample_calls,
            truth_file=truth_calls,
            output_prefix=str(tmp_path / "sample.ignore_filter"),
            ref_genome=ref_genome,
            evaluation_regions=str(high_conf.as_bed_file()),
            comparison_intervals=str(high_conf.as_bed_file()),
            input_sample="sm1",
            truth_sample="HG001",
            ignore_filter=True,
        )
        assert exists(tmp_path / "sample.ignore_filter.vcfeval_concordance.vcf.gz")
        assert exists(tmp_path / "sample.ignore_filter.vcfeval_concordance.vcf.gz.tbi")

        with pysam.VariantFile(str(tmp_path / "sample.ignore_filter.vcfeval_concordance.vcf.gz")) as vcf:
            calls = Counter([x.info["CALL"] for x in vcf])
        assert calls == {"FP": 99, "TP": 1}

    def test_vcfeval_run_use_filter(self, tmp_path, resources_dir):
        ref_genome = pjoin(resources_dir, "sample.fasta")
        sample_calls = pjoin(resources_dir, "sample.sd.vcf.gz")
        truth_calls = pjoin(resources_dir, "gtr.sample.sd.vcf.gz")
        sp = SimplePipeline(0, 100)
        high_conf = IntervalFile(None, pjoin(resources_dir, "highconf.interval_list"))
        VcfPipelineUtils(sp).run_vcfeval_concordance(
            input_file=sample_calls,
            truth_file=truth_calls,
            output_prefix=str(tmp_path / "sample.use_filter"),
            ref_genome=ref_genome,
            evaluation_regions=str(high_conf.as_bed_file()),
            comparison_intervals=str(high_conf.as_bed_file()),
            input_sample="sm1",
            truth_sample="HG001",
            ignore_filter=False,
        )

        assert exists(tmp_path / "sample.use_filter.vcfeval_concordance.vcf.gz")
        assert exists(tmp_path / "sample.use_filter.vcfeval_concordance.vcf.gz.tbi")

        with pysam.VariantFile(str(tmp_path / "sample.use_filter.vcfeval_concordance.vcf.gz")) as vcf:
            calls = Counter([x.info["CALL"] for x in vcf])
        assert calls == {"FP": 91, "TP": 1, "IGN": 8}


@patch("subprocess.call")
def test_intersect_bed_files(mock_subprocess_call, tmp_path, resources_dir):
    bed1 = pjoin(resources_dir, "bed1.bed")
    bed2 = pjoin(resources_dir, "bed2.bed")
    output_path = pjoin(tmp_path, "output.bed")

    # Test with simple pipeline
    sp = SimplePipeline(0, 10)
    VcfPipelineUtils(sp).intersect_bed_files(bed1, bed2, output_path)

    VcfPipelineUtils().intersect_bed_files(bed1, bed2, output_path)
    mock_subprocess_call.assert_called_once_with(
        ["bedtools", "intersect", "-a", bed1, "-b", bed2], stdout=mock.ANY, shell=False
    )
    assert exists(output_path)


def test_bed_file_length(resources_dir):
    bed1 = pjoin(resources_dir, "bed1.bed")
    result = bed_file_length(bed1)
    assert result == 3026


def test_close_to_hmer_run(resources_dir):
    input_vcf = vcftools.get_vcf_df(pjoin(resources_dir, "hg19.vcf.gz"))
    runs_file = pjoin(resources_dir, "runs.hg19.bed")
    result = close_to_hmer_run(input_vcf, runs_file)
    assert result["close_to_hmer_run"].sum() == 76
