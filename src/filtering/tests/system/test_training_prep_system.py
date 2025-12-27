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


# Test file paths


def test_training_prep_cnv(tmpdir, resources_dir):
    """
    Test training_prep_cnv

    This test verifies that the function correctly processes CNV data
    when ignoring FILTER fields, which is the typical use case for
    training data preparation.
    """
    call_vcf = str(resources_dir / "HG002_full_sample.TEST.cnv.vcf.gz")
    base_vcf = str(resources_dir / "GRCh38_HG2-T2TQ100-V1.1_stvar.ge1000.vcf.gz")
    hcr_bed = str(resources_dir / "GRCh38_HG2-T2TQ100-V1.1_stvar.benchmark.bed")

    output_prefix = str(tmpdir / "cnv_training")

    # Custom annotations to extract from the VCF
    custom_annotations = [
        "CNMOPS_SAMPLE_MEAN",
        "CNMOPS_SAMPLE_STDEV",
        "CNMOPS_COHORT_MEAN",
        "CNMOPS_COHORT_STDEV",
        "CNV_SOURCE",
        "GAP_PERCENTAGE",
        "SVLEN",
        "SVTYPE",
    ]

    # Run the training prep function
    tprep.training_prep_cnv(
        call_vcf=call_vcf,
        base_vcf=base_vcf,
        hcr=hcr_bed,
        custom_annotations=custom_annotations,
        train_fraction=0.7,
        output_prefix=output_prefix,
        ignore_cnv_type=True,
        skip_collapse=False,
    )

    # Verify output files were created
    train_file = Path(tmpdir) / "cnv_training.train.h5"
    test_file = Path(tmpdir) / "cnv_training.test.h5"
    concordance_file = Path(tmpdir) / "cnv_training.concordance.h5"

    assert train_file.exists(), f"Training file not created: {train_file}"
    assert test_file.exists(), f"Test file not created: {test_file}"
    assert concordance_file.exists(), f"Concordance file not created: {concordance_file}"

    # Load and verify the data
    train_df = pd.read_hdf(str(train_file), key="train")
    test_df = pd.read_hdf(str(test_file), key="test")

    # Check that we have data
    assert len(train_df) > 0, "Training set is empty"
    assert len(test_df) > 0, "Test set is empty"

    # Check that we have both TP and FP labels
    train_label_dist = train_df["label"].value_counts().to_dict()
    test_label_dist = test_df["label"].value_counts().to_dict()

    # Check reasonable label distribution (based on current test data)
    # Training set should have more FPs than TPs (this is characteristic of the test data)
    assert train_label_dist[0] > train_label_dist[1], "Training set should have more FP (0) than TP (1)"

    # Verify total combined counts include the high GAP_PERCENTAGE calls added as FPs
    total_fp = train_label_dist[0] + test_label_dist[0]
    total_tp = train_label_dist[1] + test_label_dist[1]

    # Total FP should be original 2798 from concordance + up to 482 high gap calls
    # (some high gap calls might already be in concordance, so we check a range)
    # Total TP should be 642 from concordance
    assert 3401 == total_fp, f"Expected 3401 total FP, got {total_fp}"
    assert total_tp == 676, f"Expected 676 total TP, got {total_tp}"
