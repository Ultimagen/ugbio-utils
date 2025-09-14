import shutil
from collections import Counter
from os.path import basename, exists
from os.path import join as pjoin
from pathlib import Path
from unittest import mock
from unittest.mock import patch

import pysam
import pytest
from simppl.simple_pipeline import SimplePipeline
from ugbio_core.vcf_pipeline_utils import VcfPipelineUtils
from ugbio_core.vcfbed import vcftools
from ugbio_core.vcfbed.interval_file import IntervalFile


@pytest.fixture
def resources_dir():
    return Path(__file__).parent.parent / "resources"


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
