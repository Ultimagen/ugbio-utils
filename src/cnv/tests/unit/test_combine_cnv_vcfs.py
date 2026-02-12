"""Unit tests for combine_cnv_vcfs function."""

import os
import tempfile

import pysam
import pytest
from ugbio_cnv.combine_cnmops_cnvpytor_cnv_calls import combine_cnv_vcfs


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def fasta_index(temp_dir):
    """Create a mock FASTA index file."""
    fai_path = os.path.join(temp_dir, "reference.fa.fai")
    with open(fai_path, "w") as f:
        f.write("chr1\t248956422\t52\t60\t61\n")
        f.write("chr2\t242193529\t253404903\t60\t61\n")
        f.write("chr3\t198295559\t500657651\t60\t61\n")
    return fai_path


@pytest.fixture
def cnmops_vcf(temp_dir, fasta_index):
    """Create a mock cn.mops VCF file."""
    vcf_path = os.path.join(temp_dir, "cnmops.vcf.gz")

    # Create header
    header = pysam.VariantHeader()
    header.add_meta("fileformat", value="VCFv4.2")
    header.add_meta("source", value="cn.mops")

    # Add contigs
    header.contigs.add("chr1", length=248956422)
    header.contigs.add("chr2", length=242193529)

    # Add INFO fields
    header.info.add("SVTYPE", number=1, type="String", description="Type of structural variant")
    header.info.add("SVLEN", number=".", type="Integer", description="Length of structural variant")
    header.info.add("CN", number=1, type="Float", description="Copy number")

    # Add sample
    header.add_sample("sample1")

    # Write VCF with records
    with pysam.VariantFile(vcf_path, "w", header=header) as vcf:
        # Add a deletion on chr1
        rec = vcf.new_record(
            contig="chr1",
            start=1000,
            stop=2000,
            alleles=("N", "<DEL>"),
            id="cnmops_del1",
        )
        rec.info["SVTYPE"] = "DEL"
        rec.info["SVLEN"] = -1000
        rec.info["CN"] = 1.0
        vcf.write(rec)

        # Add a duplication on chr2
        rec = vcf.new_record(
            contig="chr2",
            start=5000,
            stop=10000,
            alleles=("N", "<DUP>"),
            id="cnmops_dup1",
        )
        rec.info["SVTYPE"] = "DUP"
        rec.info["SVLEN"] = 5000
        rec.info["CN"] = 3.0
        vcf.write(rec)

    # Index the VCF
    pysam.tabix_index(vcf_path, preset="vcf", force=True)

    return vcf_path


@pytest.fixture
def cnvpytor_vcf(temp_dir, fasta_index):
    """Create a mock CNVpytor VCF file."""
    vcf_path = os.path.join(temp_dir, "cnvpytor.vcf.gz")

    # Create header
    header = pysam.VariantHeader()
    header.add_meta("fileformat", value="VCFv4.2")
    header.add_meta("source", value="CNVpytor")

    # Add contigs (different order to test merging)
    header.contigs.add("chr2", length=242193529)
    header.contigs.add("chr1", length=248956422)

    # Add INFO fields
    header.info.add("SVTYPE", number=1, type="String", description="Type of structural variant")
    header.info.add("SVLEN", number=1, type="Integer", description="Length of structural variant")
    header.info.add("RD", number=1, type="Float", description="Read depth ratio")

    # Add sample
    header.add_sample("sample1")

    # Write VCF with records
    with pysam.VariantFile(vcf_path, "w", header=header) as vcf:
        # Add a deletion on chr1
        rec = vcf.new_record(
            contig="chr1",
            start=3000,
            stop=4000,
            alleles=("N", "<DEL>"),
            id="cnvpytor_del1",
        )
        rec.info["SVTYPE"] = "DEL"
        rec.info["SVLEN"] = -1000
        rec.info["RD"] = 0.5
        vcf.write(rec)

        # Add a duplication on chr2
        rec = vcf.new_record(
            contig="chr2",
            start=15000,
            stop=20000,
            alleles=("N", "<DUP>"),
            id="cnvpytor_dup1",
        )
        rec.info["SVTYPE"] = "DUP"
        rec.info["SVLEN"] = 5000
        rec.info["RD"] = 1.5
        vcf.write(rec)

    # Index the VCF
    pysam.tabix_index(vcf_path, preset="vcf", force=True)

    return vcf_path


def test_combine_cnv_vcfs_basic(temp_dir, cnmops_vcf, cnvpytor_vcf, fasta_index):
    """Test basic functionality of combine_cnv_vcfs."""
    output_vcf = os.path.join(temp_dir, "combined.vcf.gz")

    result = combine_cnv_vcfs(
        cnmops_vcf=[cnmops_vcf],
        cnvpytor_vcf=[cnvpytor_vcf],
        fasta_index=fasta_index,
        output_vcf=output_vcf,
        output_directory=temp_dir,
    )

    # Check that output file was created
    assert os.path.exists(result)
    assert result == output_vcf

    # Check that index was created
    assert os.path.exists(output_vcf + ".tbi") or os.path.exists(output_vcf + ".csi")

    # Open and verify the combined VCF
    with pysam.VariantFile(output_vcf) as vcf:
        # Check header has CNV_SOURCE
        assert "CNV_SOURCE" in vcf.header.info

        # Check contigs are present
        assert "chr1" in vcf.header.contigs
        assert "chr2" in vcf.header.contigs
        assert "chr3" in vcf.header.contigs

        # Check all INFO fields are present
        assert "SVTYPE" in vcf.header.info
        assert "SVLEN" in vcf.header.info
        assert "CN" in vcf.header.info  # from cn.mops
        assert "RD" in vcf.header.info  # from cnvpytor

        # Collect all records
        records = list(vcf)

        # Should have 4 total records (2 from each VCF)
        assert len(records) == 4

        # Check that records are sorted by chromosome and position
        assert records[0].chrom == "chr1"
        assert records[0].start == 1000  # cnmops_del1
        assert records[1].chrom == "chr1"
        assert records[1].start == 3000  # cnvpytor_del1
        assert records[2].chrom == "chr2"
        assert records[2].start == 5000  # cnmops_dup1
        assert records[3].chrom == "chr2"
        assert records[3].start == 15000  # cnvpytor_dup1

        # Check CNV_SOURCE annotation
        # CNV_SOURCE is defined with Number="." so pysam returns tuples
        sources = [rec.info.get("CNV_SOURCE") for rec in records]
        assert ("cn.mops",) in sources
        assert ("cnvpytor",) in sources
        assert sources.count(("cn.mops",)) == 2
        assert sources.count(("cnvpytor",)) == 2


def test_combine_cnv_vcfs_creates_output_directory(temp_dir, cnmops_vcf, cnvpytor_vcf, fasta_index):
    """Test that combine_cnv_vcfs creates output directory if it doesn't exist."""
    new_output_dir = os.path.join(temp_dir, "new_output_dir")
    output_vcf = os.path.join(new_output_dir, "combined.vcf.gz")

    result = combine_cnv_vcfs(
        cnmops_vcf=[cnmops_vcf],
        cnvpytor_vcf=[cnvpytor_vcf],
        fasta_index=fasta_index,
        output_vcf=output_vcf,
        output_directory=new_output_dir,
    )

    assert os.path.exists(result)
    assert os.path.exists(new_output_dir)


def test_combine_cnv_vcfs_preserves_info_fields(temp_dir, cnmops_vcf, cnvpytor_vcf, fasta_index):
    """Test that combine_cnv_vcfs preserves INFO fields from both VCFs."""
    output_vcf = os.path.join(temp_dir, "combined.vcf.gz")

    combine_cnv_vcfs(
        cnmops_vcf=[cnmops_vcf],
        cnvpytor_vcf=[cnvpytor_vcf],
        fasta_index=fasta_index,
        output_vcf=output_vcf,
        output_directory=temp_dir,
    )

    with pysam.VariantFile(output_vcf) as vcf:
        records = list(vcf)

        # Check cn.mops records have CN field
        cnmops_records = [r for r in records if r.info.get("CNV_SOURCE") == "cn.mops"]
        for rec in cnmops_records:
            assert "CN" in rec.info

        # Check cnvpytor records have RD field
        cnvpytor_records = [r for r in records if r.info.get("CNV_SOURCE") == "cnvpytor"]
        for rec in cnvpytor_records:
            assert "RD" in rec.info


def test_combine_cnv_vcfs_empty_lists(temp_dir, fasta_index):
    """Test that combine_cnv_vcfs raises error when both lists are empty."""
    output_vcf = os.path.join(temp_dir, "combined.vcf.gz")

    with pytest.raises(ValueError, match="At least one of cnmops_vcf or cnvpytor_vcf must be non-empty"):
        combine_cnv_vcfs(
            cnmops_vcf=[],
            cnvpytor_vcf=[],
            fasta_index=fasta_index,
            output_vcf=output_vcf,
            output_directory=temp_dir,
        )


def test_combine_cnv_vcfs_only_cnmops(temp_dir, cnmops_vcf, fasta_index):
    """Test combine_cnv_vcfs with only cn.mops VCFs."""
    output_vcf = os.path.join(temp_dir, "combined.vcf.gz")

    result = combine_cnv_vcfs(
        cnmops_vcf=[cnmops_vcf],
        cnvpytor_vcf=[],
        fasta_index=fasta_index,
        output_vcf=output_vcf,
        output_directory=temp_dir,
    )

    assert os.path.exists(result)

    with pysam.VariantFile(output_vcf) as vcf:
        records = list(vcf)
        assert len(records) == 2  # Only cn.mops records
        assert all(rec.info.get("CNV_SOURCE") == ("cn.mops",) for rec in records)


def test_combine_cnv_vcfs_only_cnvpytor(temp_dir, cnvpytor_vcf, fasta_index):
    """Test combine_cnv_vcfs with only CNVpytor VCFs."""
    output_vcf = os.path.join(temp_dir, "combined.vcf.gz")

    result = combine_cnv_vcfs(
        cnmops_vcf=[],
        cnvpytor_vcf=[cnvpytor_vcf],
        fasta_index=fasta_index,
        output_vcf=output_vcf,
        output_directory=temp_dir,
    )

    assert os.path.exists(result)

    with pysam.VariantFile(output_vcf) as vcf:
        records = list(vcf)
        assert len(records) == 2  # Only cnvpytor records
        assert all(rec.info.get("CNV_SOURCE") == ("cnvpytor",) for rec in records)


def test_combine_cnv_vcfs_multiple_files(temp_dir, cnmops_vcf, cnvpytor_vcf, fasta_index):
    """Test combine_cnv_vcfs with multiple VCF files from each caller."""
    output_vcf = os.path.join(temp_dir, "combined.vcf.gz")

    # Use the same VCF twice to simulate multiple input files
    result = combine_cnv_vcfs(
        cnmops_vcf=[cnmops_vcf, cnmops_vcf],
        cnvpytor_vcf=[cnvpytor_vcf, cnvpytor_vcf],
        fasta_index=fasta_index,
        output_vcf=output_vcf,
        output_directory=temp_dir,
    )

    assert os.path.exists(result)

    with pysam.VariantFile(output_vcf) as vcf:
        records = list(vcf)
        # Should have 8 total records (2 from each VCF, duplicated)
        assert len(records) == 8

        cnmops_count = sum(1 for r in records if r.info.get("CNV_SOURCE") == ("cn.mops",))
        cnvpytor_count = sum(1 for r in records if r.info.get("CNV_SOURCE") == ("cnvpytor",))

        assert cnmops_count == 4  # 2 records * 2 files
        assert cnvpytor_count == 4  # 2 records * 2 files


def test_combine_cnv_vcfs_make_ids_unique(temp_dir, cnmops_vcf, cnvpytor_vcf, fasta_index):
    """Test that make_ids_unique parameter preserves original IDs and makes them unique."""
    output_vcf = os.path.join(temp_dir, "combined_unique_ids.vcf.gz")

    result = combine_cnv_vcfs(
        cnmops_vcf=[cnmops_vcf],
        cnvpytor_vcf=[cnvpytor_vcf],
        fasta_index=fasta_index,
        output_vcf=output_vcf,
        output_directory=temp_dir,
        make_ids_unique=True,
    )

    assert os.path.exists(result)

    with pysam.VariantFile(output_vcf) as vcf:
        records = list(vcf)

        # Should have 4 total records
        assert len(records) == 4

        # All records should have IDs
        ids = [rec.id for rec in records]
        assert all(variant_id is not None for variant_id in ids)

        # IDs should be unique
        assert len(set(ids)) == len(ids)

        # Original IDs should be preserved since there are no duplicates
        assert "cnmops_del1" in ids
        assert "cnmops_dup1" in ids
        assert "cnvpytor_del1" in ids
        assert "cnvpytor_dup1" in ids


def test_combine_cnv_vcfs_without_make_ids_unique(temp_dir, cnmops_vcf, cnvpytor_vcf, fasta_index):
    """Test that without make_ids_unique parameter, original IDs are preserved."""
    output_vcf = os.path.join(temp_dir, "combined_original_ids.vcf.gz")

    result = combine_cnv_vcfs(
        cnmops_vcf=[cnmops_vcf],
        cnvpytor_vcf=[cnvpytor_vcf],
        fasta_index=fasta_index,
        output_vcf=output_vcf,
        output_directory=temp_dir,
        make_ids_unique=False,
    )

    assert os.path.exists(result)

    with pysam.VariantFile(output_vcf) as vcf:
        records = list(vcf)

        # Should have 4 total records
        assert len(records) == 4

        # Check that original IDs are preserved
        ids = [rec.id for rec in records]
        assert "cnmops_del1" in ids
        assert "cnmops_dup1" in ids
        assert "cnvpytor_del1" in ids
        assert "cnvpytor_dup1" in ids


def test_combine_cnv_vcfs_unique_ids_multiple_files(temp_dir, cnmops_vcf, cnvpytor_vcf, fasta_index):
    """Test that make_ids_unique adds suffixes when there are duplicate IDs."""
    output_vcf = os.path.join(temp_dir, "combined_unique_multi.vcf.gz")

    # Use the same VCF twice to simulate duplicate records
    result = combine_cnv_vcfs(
        cnmops_vcf=[cnmops_vcf, cnmops_vcf],
        cnvpytor_vcf=[cnvpytor_vcf],
        fasta_index=fasta_index,
        output_vcf=output_vcf,
        output_directory=temp_dir,
        make_ids_unique=True,
    )

    assert os.path.exists(result)

    with pysam.VariantFile(output_vcf) as vcf:
        records = list(vcf)

        # Should have 6 total records (2 * 2 from cnmops + 2 from cnvpytor)
        assert len(records) == 6

        # All IDs should be unique
        ids = [rec.id for rec in records]
        assert len(set(ids)) == len(ids)

        # First occurrence should preserve original ID, duplicates should have suffix
        # cnmops_del1, cnmops_del1_1, cnmops_dup1, cnmops_dup1_1, cnvpytor_del1, cnvpytor_dup1
        assert "cnmops_del1" in ids
        assert "cnmops_del1_1" in ids
        assert "cnmops_dup1" in ids
        assert "cnmops_dup1_1" in ids
        assert "cnvpytor_del1" in ids
        assert "cnvpytor_dup1" in ids
