import os.path
import pathlib

import pandas as pd
import pytest
import ugbio_filtering.training_prep as tprep


@pytest.fixture
def resources_dir():
    return pathlib.Path(__file__).parent.parent / "resources"


class TestTrainingPrep:
    def test_calculate_labeled_vcf(self, resources_dir):
        inputs_file = str(pathlib.Path(resources_dir, "input.vcf.gz"))
        vcfeval_output = str(pathlib.Path(resources_dir, "vcfeval_output.vcf.gz"))
        expected_result_file = str(pathlib.Path(resources_dir, "expected_result_calculate_labeled_vcf.h5"))
        expected_results = pd.DataFrame(pd.read_hdf(expected_result_file, key="result"))
        labeled_df = tprep.calculate_labeled_vcf(inputs_file, vcfeval_output, contig="chr20")
        pd.testing.assert_frame_equal(labeled_df, expected_results)

    def test_calculate_labeled_vcf_with_custom_info(self, resources_dir):
        inputs_file = str(pathlib.Path(resources_dir, "input.vcf.gz"))
        vcfeval_output = str(pathlib.Path(resources_dir, "vcfeval_output.vcf.gz"))
        labeled_df = tprep.calculate_labeled_vcf(
            inputs_file, vcfeval_output, contig="chr20", custom_info_fields=["EXOME", "LCR"]
        )
        assert "lcr" in labeled_df.columns
        assert "exome" in labeled_df.columns

    def test_calculate_labels(self, resources_dir):
        joint_vcf_df_file = str(pathlib.Path(resources_dir, "expected_result_calculate_labeled_vcf.h5"))
        joint_vcf_df = pd.DataFrame(pd.read_hdf(joint_vcf_df_file, key="result"))
        labeled_df = tprep.calculate_labels(joint_vcf_df)
        expected_result_file = str(pathlib.Path(resources_dir, "expected_labels.h5"))
        pd.testing.assert_series_equal(labeled_df, pd.read_hdf(expected_result_file, key="labels"))  # type: ignore

    def test_label_with_approximate_gt(self, tmpdir, resources_dir):
        inputs_file = str(pathlib.Path(resources_dir, "006919_no_frd_chr1_1_5000000.vcf.gz"))
        blacklist_file = str(pathlib.Path(resources_dir, "blacklist_chr1_1_5000000.h5"))
        tprep.label_with_approximate_gt(
            inputs_file,
            blacklist_file,
            chromosomes_to_read=["chr1"],
            output_file=str(pathlib.Path(tmpdir, "output.h5")),
        )
        assert os.path.exists(str(pathlib.Path(tmpdir, "output.h5")))
        vc = pd.read_hdf(str(pathlib.Path(tmpdir, "output.h5")), key="chr1")["label"].value_counts()
        assert len(vc) == 2
        assert vc[1] == 8715
        assert vc[0] == 2003

    def test_label_with_approximate_gt_with_custom_columns(self, tmpdir, resources_dir):
        inputs_file = str(pathlib.Path(resources_dir, "006919_no_frd_chr1_1_5000000.vcf.gz"))
        blacklist_file = str(pathlib.Path(resources_dir, "blacklist_chr1_1_5000000.h5"))
        tprep.label_with_approximate_gt(
            inputs_file,
            blacklist_file,
            chromosomes_to_read=["chr1"],
            output_file=str(pathlib.Path(tmpdir, "output.h5")),
            custom_info_fields=["EXOME", "LCR"],
        )
        assert os.path.exists(str(pathlib.Path(tmpdir, "output.h5")))
        vc = pd.read_hdf(str(pathlib.Path(tmpdir, "output.h5")), key="chr1")["label"].value_counts()
        assert len(vc) == 2
        assert vc[1] == 8715
        assert vc[0] == 2003
        test_df = pd.read_hdf(str(pathlib.Path(tmpdir, "output.h5")), key="chr1")
        assert "lcr" in test_df.columns
