from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import ugbio_filtering.training_prep as tprep


@pytest.fixture
def resources_dir():
    return Path(__file__).parent.parent / "resources"


def test_prepare_ground_truth(tmpdir, resources_dir):
    calls_file = str(Path(resources_dir, "input.vcf.gz"))
    vcfeval_output_file = Path(resources_dir, "vcfeval_output.vcf.gz")
    # bypass well-tested run_vcfeval_concordance
    with patch(
        "ugbio_comparison.vcf_comparison_utils.VcfComparisonUtils.run_vcfeval_concordance"
    ) as mock_run_vcfeval_concordance:
        mock_run_vcfeval_concordance.return_value = vcfeval_output_file
        joint_df = pd.read_hdf(str(Path(resources_dir, "labeled_df.h5")), key="df")
        # bypass creating of the joint vcf
        with patch("ugbio_filtering.training_prep.calculate_labeled_vcf") as mock_calculate_labeled_vcf:
            mock_calculate_labeled_vcf.return_value = joint_df
            reference_path = str(Path(resources_dir, "ref_fragment.fa.gz"))
            output_file = str(Path(tmpdir, "output.h5"))
            tprep.prepare_ground_truth(calls_file, "", "", reference_path, output_file, chromosome=["chr21"])
            assert Path(output_file).exists()
            output_df = pd.DataFrame(pd.read_hdf(output_file, key="chr21"))
            expected_df = pd.DataFrame(pd.read_hdf(str(Path(resources_dir, "expected_output.h5")), key="chr21"))
            pd.testing.assert_frame_equal(
                output_df.drop(["spanning_deletion", "multiallelic_group"], axis=1),
                expected_df.drop(["spanning_deletion", "multiallelic_group"], axis=1),
            )
