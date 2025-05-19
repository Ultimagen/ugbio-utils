import tempfile
from pathlib import Path

import pysam
import pytest
from ugbio_mrd.mrd_utils import generate_synthetic_signatures, read_signature


@pytest.fixture
def resources_dir():
    return Path(__file__).parent.parent / "resources"


def test_parallel_synthetic_signatures(resources_dir):
    """
    Test that the parallelized version of generate_synthetic_signatures produces
    the same results as the non-parallelized version.
    """
    # Input files
    signature_vcf = str(resources_dir / "mutect_mrd_signature_test.vcf.gz")
    db_vcf = str(
        resources_dir
        / "pancan_pcawg_2020.mutations_hg38_GNOMAD_dbsnp_beds.sorted.Annotated.HMER_LEN.edited.chr19.vcf.gz"
    )

    # Create temporary directories for outputs
    with tempfile.TemporaryDirectory() as output_dir_single, tempfile.TemporaryDirectory() as output_dir_multi:
        # Generate synthetic signatures with a single process
        n_synthetic_signatures = 2
        single_process_signatures = generate_synthetic_signatures(
            signature_vcf=signature_vcf,
            db_vcf=db_vcf,
            n_synthetic_signatures=n_synthetic_signatures,
            output_dir=output_dir_single,
            n_processes=1,
        )

        # Generate synthetic signatures with multiple processes
        multi_process_signatures = generate_synthetic_signatures(
            signature_vcf=signature_vcf,
            db_vcf=db_vcf,
            n_synthetic_signatures=n_synthetic_signatures,
            output_dir=output_dir_multi,
            n_processes=2,
        )

        # Check that the number of output files is the same
        assert len(single_process_signatures) == len(multi_process_signatures) == n_synthetic_signatures

        # Compare the content of each pair of files
        for single_file, multi_file in zip(single_process_signatures, multi_process_signatures):
            # Extract the variants from each file
            single_variants = []
            multi_variants = []

            with pysam.VariantFile(single_file) as single_vcf, pysam.VariantFile(multi_file) as multi_vcf:
                # Check that the headers are the same
                assert str(single_vcf.header) == str(multi_vcf.header)

                # Collect all variants from both files
                for rec in single_vcf:
                    single_variants.append((rec.chrom, rec.pos, rec.ref, rec.alts))

                for rec in multi_vcf:
                    multi_variants.append((rec.chrom, rec.pos, rec.ref, rec.alts))

            # Check that the variants are the same
            assert len(single_variants) == len(multi_variants)
            assert set(single_variants) == set(multi_variants)


def test_synthetic_signatures_deterministic(resources_dir):
    """
    Test that generate_synthetic_signatures produces deterministic results
    when run multiple times with the same inputs.
    """
    # Input files
    signature_vcf = str(resources_dir / "mutect_mrd_signature_test.vcf.gz")
    db_vcf = str(
        resources_dir
        / "pancan_pcawg_2020.mutations_hg38_GNOMAD_dbsnp_beds.sorted.Annotated.HMER_LEN.edited.chr19.vcf.gz"
    )

    # Create temporary directories for outputs
    with tempfile.TemporaryDirectory() as output_dir_1, tempfile.TemporaryDirectory() as output_dir_2:
        # Generate synthetic signatures twice
        n_synthetic_signatures = 2
        signatures_1 = generate_synthetic_signatures(
            signature_vcf=signature_vcf,
            db_vcf=db_vcf,
            n_synthetic_signatures=n_synthetic_signatures,
            output_dir=output_dir_1,
            n_processes=2,
        )

        signatures_2 = generate_synthetic_signatures(
            signature_vcf=signature_vcf,
            db_vcf=db_vcf,
            n_synthetic_signatures=n_synthetic_signatures,
            output_dir=output_dir_2,
            n_processes=2,
        )

        # Check that the number of output files is the same
        assert len(signatures_1) == len(signatures_2) == n_synthetic_signatures

        # Compare the content of each pair of files
        for file_1, file_2 in zip(signatures_1, signatures_2):
            # Extract the variants from each file
            variants_1 = []
            variants_2 = []

            with pysam.VariantFile(file_1) as vcf_1, pysam.VariantFile(file_2) as vcf_2:
                # Check that the headers are the same
                assert str(vcf_1.header) == str(vcf_2.header)

                # Collect all variants from both files
                for rec in vcf_1:
                    variants_1.append((rec.chrom, rec.pos, rec.ref, rec.alts))

                for rec in vcf_2:
                    variants_2.append((rec.chrom, rec.pos, rec.ref, rec.alts))

            # Check that the variants are the same
            assert len(variants_1) == len(variants_2)
            assert set(variants_1) == set(variants_2)


def test_generate_synthetic_signatures(tmpdir, resources_dir):
    """
    Test that the synthetic signatures generated from a database
    have the same trinucleotide substitution context as the input signature.
    """
    # Create temporary directories for outputs
    with tempfile.TemporaryDirectory() as output_dir:
        # Generate synthetic signatures
        n_synthetic_signatures = 1
        synthetic_signature_list = generate_synthetic_signatures(
            signature_vcf=str(resources_dir / "mutect_mrd_signature_test.vcf.gz"),
            db_vcf=str(
                resources_dir
                / "pancan_pcawg_2020.mutations_hg38_GNOMAD_dbsnp_beds.sorted.Annotated.HMER_LEN.edited.chr19.vcf.gz"
            ),
            n_synthetic_signatures=n_synthetic_signatures,
            output_dir=output_dir,
        )
        expected_signature_vcf = str(resources_dir / "synthetic_signature_test.vcf.gz")
        # Read the generated synthetic signature
        signature = read_signature(synthetic_signature_list[0], return_dataframes=True)
        # Read the expected synthetic signature
        expected_signature = read_signature(expected_signature_vcf, return_dataframes=True)
        # Test that motif distribution is the same (0th order)
        assert (
            signature.groupby(["ref", "alt"]).value_counts()
            == expected_signature.groupby(["ref", "alt"]).value_counts()
        ).all()
        # Delete the expected signature index file
        expected_signature_index = expected_signature_vcf + ".csi"
        if Path(expected_signature_index).exists():
            Path(expected_signature_index).unlink()
