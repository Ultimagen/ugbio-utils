"""Unit tests for CNV smoothing functions: identify_smoothing_candidates and _group_candidates_transitively."""

import pysam
from ugbio_cnv import combine_cnv_vcf_utils


class TestIdentifySmoothingCandidates:
    """Direct unit tests for identify_smoothing_candidates function."""

    def test_empty_vcf(self, tmp_path):
        """Test that empty VCF returns empty set."""
        vcf_path = tmp_path / "empty.vcf.gz"

        # Create empty VCF with CIPOS in header
        header = pysam.VariantHeader()
        header.add_line("##fileformat=VCFv4.2")
        header.add_line("##contig=<ID=chr1,length=248956422>")
        header.add_line('##INFO=<ID=SVLEN,Number=.,Type=Integer,Description="CNV length">')
        header.add_line('##INFO=<ID=SVTYPE,Number=1,Type=String,Description="CNV type">')
        header.add_line('##INFO=<ID=CIPOS,Number=2,Type=Integer,Description="Confidence interval">')
        header.add_line('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">')
        header.add_line('##FORMAT=<ID=CN,Number=1,Type=Float,Description="Copy number">')
        header.add_sample("test_sample")

        with pysam.VariantFile(str(vcf_path), "w", header=header):
            pass  # Empty file

        pysam.tabix_index(str(vcf_path), preset="vcf", force=True)

        # Call function
        candidates = combine_cnv_vcf_utils.identify_smoothing_candidates(
            str(vcf_path),
            max_gap_absolute=50000,
            gap_scale_fraction=0.05,
            ignore_sv_type=False,
            ignore_filter=True,
            cipos_threshold=50,
        )

        assert candidates == set()

    def test_different_chromosomes(self, tmp_path):
        """Test that CNVs on different chromosomes don't merge."""
        vcf_path = tmp_path / "different_chroms.vcf.gz"

        # Create VCF with CNVs on chr1 and chr2 (close positions but different chroms)
        header = pysam.VariantHeader()
        header.add_line("##fileformat=VCFv4.2")
        header.add_line("##contig=<ID=chr1,length=248956422>")
        header.add_line("##contig=<ID=chr2,length=242193529>")
        header.add_line('##INFO=<ID=SVLEN,Number=.,Type=Integer,Description="CNV length">')
        header.add_line('##INFO=<ID=SVTYPE,Number=1,Type=String,Description="CNV type">')
        header.add_line('##INFO=<ID=CIPOS,Number=2,Type=Integer,Description="Confidence interval">')
        header.add_line('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">')
        header.add_line('##FORMAT=<ID=CN,Number=1,Type=Float,Description="Copy number">')
        header.add_sample("test_sample")

        with pysam.VariantFile(str(vcf_path), "w", header=header) as vcf:
            # CNV on chr1
            record1 = vcf.new_record()
            record1.contig = "chr1"
            record1.pos = 1000
            record1.stop = 2000
            record1.id = "CNV_chr1"
            record1.alleles = ("N", "<DEL>")
            record1.info["SVLEN"] = (1000,)
            record1.info["SVTYPE"] = "DEL"
            record1.info["CIPOS"] = (-100, 100)
            record1.qual = 30.0
            record1.filter.add("PASS")
            record1.samples["test_sample"]["GT"] = (0, 1)
            record1.samples["test_sample"]["CN"] = 1.0
            vcf.write(record1)

            # CNV on chr2 (same position, small gap if they were on same chrom)
            record2 = vcf.new_record()
            record2.contig = "chr2"
            record2.pos = 2500
            record2.stop = 3500
            record2.id = "CNV_chr2"
            record2.alleles = ("N", "<DEL>")
            record2.info["SVLEN"] = (1000,)
            record2.info["SVTYPE"] = "DEL"
            record2.info["CIPOS"] = (-100, 100)
            record2.qual = 30.0
            record2.filter.add("PASS")
            record2.samples["test_sample"]["GT"] = (0, 1)
            record2.samples["test_sample"]["CN"] = 1.0
            vcf.write(record2)

        pysam.tabix_index(str(vcf_path), preset="vcf", force=True)

        # Call function
        candidates = combine_cnv_vcf_utils.identify_smoothing_candidates(
            str(vcf_path),
            max_gap_absolute=50000,
            gap_scale_fraction=0.05,
            ignore_sv_type=False,
            ignore_filter=True,
            cipos_threshold=50,
        )

        # Should be empty (different chromosomes)
        assert candidates == set()

    def test_different_svtype(self, tmp_path):
        """Test that CNVs with different SVTYPE don't merge when ignore_sv_type=False."""
        vcf_path = tmp_path / "different_svtype.vcf.gz"

        # Create VCF with DEL and DUP on same chromosome
        header = pysam.VariantHeader()
        header.add_line("##fileformat=VCFv4.2")
        header.add_line("##contig=<ID=chr1,length=248956422>")
        header.add_line('##INFO=<ID=SVLEN,Number=.,Type=Integer,Description="CNV length">')
        header.add_line('##INFO=<ID=SVTYPE,Number=1,Type=String,Description="CNV type">')
        header.add_line('##INFO=<ID=CIPOS,Number=2,Type=Integer,Description="Confidence interval">')
        header.add_line('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">')
        header.add_line('##FORMAT=<ID=CN,Number=1,Type=Float,Description="Copy number">')
        header.add_sample("test_sample")

        with pysam.VariantFile(str(vcf_path), "w", header=header) as vcf:
            # DEL CNV
            record1 = vcf.new_record()
            record1.contig = "chr1"
            record1.pos = 1000
            record1.stop = 2000
            record1.id = "CNV_DEL"
            record1.alleles = ("N", "<DEL>")
            record1.info["SVLEN"] = (1000,)
            record1.info["SVTYPE"] = "DEL"
            record1.info["CIPOS"] = (-100, 100)
            record1.qual = 30.0
            record1.filter.add("PASS")
            record1.samples["test_sample"]["GT"] = (0, 1)
            record1.samples["test_sample"]["CN"] = 1.0
            vcf.write(record1)

            # DUP CNV (close gap of 500bp)
            record2 = vcf.new_record()
            record2.contig = "chr1"
            record2.pos = 2500
            record2.stop = 3500
            record2.id = "CNV_DUP"
            record2.alleles = ("N", "<DUP>")
            record2.info["SVLEN"] = (1000,)
            record2.info["SVTYPE"] = "DUP"
            record2.info["CIPOS"] = (-100, 100)
            record2.qual = 30.0
            record2.filter.add("PASS")
            record2.samples["test_sample"]["GT"] = (0, 1)
            record2.samples["test_sample"]["CN"] = 3.0
            vcf.write(record2)

        pysam.tabix_index(str(vcf_path), preset="vcf", force=True)

        # Call function with ignore_sv_type=False
        candidates = combine_cnv_vcf_utils.identify_smoothing_candidates(
            str(vcf_path),
            max_gap_absolute=50000,
            gap_scale_fraction=0.05,
            ignore_sv_type=False,
            ignore_filter=True,
            cipos_threshold=50,
        )

        # Should be empty (different SVTYPEs)
        assert candidates == set()

    def test_cipos_filter(self, tmp_path):
        """Test that high-confidence breakpoints (small CIPOS) are filtered out."""
        vcf_path = tmp_path / "high_confidence.vcf.gz"

        # Create VCF with CNVs having small CIPOS (high confidence)
        header = pysam.VariantHeader()
        header.add_line("##fileformat=VCFv4.2")
        header.add_line("##contig=<ID=chr1,length=248956422>")
        header.add_line('##INFO=<ID=SVLEN,Number=.,Type=Integer,Description="CNV length">')
        header.add_line('##INFO=<ID=SVTYPE,Number=1,Type=String,Description="CNV type">')
        header.add_line('##INFO=<ID=CIPOS,Number=2,Type=Integer,Description="Confidence interval">')
        header.add_line('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">')
        header.add_line('##FORMAT=<ID=CN,Number=1,Type=Float,Description="Copy number">')
        header.add_sample("test_sample")

        with pysam.VariantFile(str(vcf_path), "w", header=header) as vcf:
            # CNV with small CIPOS=(-10, 10) -> length = 10 - (-10) - 1 = 19 < 50 threshold
            record1 = vcf.new_record()
            record1.contig = "chr1"
            record1.pos = 1000
            record1.stop = 2000
            record1.id = "CNV_1"
            record1.alleles = ("N", "<DEL>")
            record1.info["SVLEN"] = (1000,)
            record1.info["SVTYPE"] = "DEL"
            record1.info["CIPOS"] = (-10, 10)
            record1.qual = 30.0
            record1.filter.add("PASS")
            record1.samples["test_sample"]["GT"] = (0, 1)
            record1.samples["test_sample"]["CN"] = 1.0
            vcf.write(record1)

            # Another CNV with small CIPOS, close gap
            record2 = vcf.new_record()
            record2.contig = "chr1"
            record2.pos = 2500
            record2.stop = 3500
            record2.id = "CNV_2"
            record2.alleles = ("N", "<DEL>")
            record2.info["SVLEN"] = (1000,)
            record2.info["SVTYPE"] = "DEL"
            record2.info["CIPOS"] = (-10, 10)
            record2.qual = 30.0
            record2.filter.add("PASS")
            record2.samples["test_sample"]["GT"] = (0, 1)
            record2.samples["test_sample"]["CN"] = 1.0
            vcf.write(record2)

        pysam.tabix_index(str(vcf_path), preset="vcf", force=True)

        # Call function with cipos_threshold=50
        candidates = combine_cnv_vcf_utils.identify_smoothing_candidates(
            str(vcf_path),
            max_gap_absolute=50000,
            gap_scale_fraction=0.05,
            ignore_sv_type=False,
            ignore_filter=True,
            cipos_threshold=50,
        )

        # Should be empty (both CNVs filtered due to high-confidence breakpoints)
        assert candidates == set()

    def test_gap_threshold(self, tmp_path):
        """Test that size-scaled gap threshold is applied correctly."""
        vcf_path = tmp_path / "gap_threshold.vcf.gz"

        # Create VCF with pairs of CNVs at different scales
        header = pysam.VariantHeader()
        header.add_line("##fileformat=VCFv4.2")
        header.add_line("##contig=<ID=chr1,length=248956422>")
        header.add_line('##INFO=<ID=SVLEN,Number=.,Type=Integer,Description="CNV length">')
        header.add_line('##INFO=<ID=SVTYPE,Number=1,Type=String,Description="CNV type">')
        header.add_line('##INFO=<ID=CIPOS,Number=2,Type=Integer,Description="Confidence interval">')
        header.add_line('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">')
        header.add_line('##FORMAT=<ID=CN,Number=1,Type=Float,Description="Copy number">')
        header.add_sample("test_sample")

        with pysam.VariantFile(str(vcf_path), "w", header=header) as vcf:
            # Small CNVs (20kb each) with 8kb gap
            # Threshold = min(50000, 0.05 * 20000) = min(50000, 1000) = 1000
            # Gap 8000 > 1000, should NOT merge
            record1 = vcf.new_record()
            record1.contig = "chr1"
            record1.pos = 1000
            record1.stop = 21000
            record1.id = "CNV_small_1"
            record1.alleles = ("N", "<DEL>")
            record1.info["SVLEN"] = (20000,)
            record1.info["SVTYPE"] = "DEL"
            record1.info["CIPOS"] = (-100, 100)
            record1.qual = 30.0
            record1.filter.add("PASS")
            record1.samples["test_sample"]["GT"] = (0, 1)
            record1.samples["test_sample"]["CN"] = 1.0
            vcf.write(record1)

            record2 = vcf.new_record()
            record2.contig = "chr1"
            record2.pos = 29000
            record2.stop = 49000
            record2.id = "CNV_small_2"
            record2.alleles = ("N", "<DEL>")
            record2.info["SVLEN"] = (20000,)
            record2.info["SVTYPE"] = "DEL"
            record2.info["CIPOS"] = (-100, 100)
            record2.qual = 30.0
            record2.filter.add("PASS")
            record2.samples["test_sample"]["GT"] = (0, 1)
            record2.samples["test_sample"]["CN"] = 1.0
            vcf.write(record2)

            # Large CNVs (2Mb each) with 8kb gap
            # Threshold = min(50000, 0.05 * 2000000) = min(50000, 100000) = 50000
            # Gap 8000 <= 50000, SHOULD merge
            record3 = vcf.new_record()
            record3.contig = "chr1"
            record3.pos = 100000
            record3.stop = 2100000
            record3.id = "CNV_large_1"
            record3.alleles = ("N", "<DEL>")
            record3.info["SVLEN"] = (2000000,)
            record3.info["SVTYPE"] = "DEL"
            record3.info["CIPOS"] = (-100, 100)
            record3.qual = 30.0
            record3.filter.add("PASS")
            record3.samples["test_sample"]["GT"] = (0, 1)
            record3.samples["test_sample"]["CN"] = 1.0
            vcf.write(record3)

            record4 = vcf.new_record()
            record4.contig = "chr1"
            record4.pos = 2108000
            record4.stop = 4108000
            record4.id = "CNV_large_2"
            record4.alleles = ("N", "<DEL>")
            record4.info["SVLEN"] = (2000000,)
            record4.info["SVTYPE"] = "DEL"
            record4.info["CIPOS"] = (-100, 100)
            record4.qual = 30.0
            record4.filter.add("PASS")
            record4.samples["test_sample"]["GT"] = (0, 1)
            record4.samples["test_sample"]["CN"] = 1.0
            vcf.write(record4)

        pysam.tabix_index(str(vcf_path), preset="vcf", force=True)

        # Call function
        candidates = combine_cnv_vcf_utils.identify_smoothing_candidates(
            str(vcf_path),
            max_gap_absolute=50000,
            gap_scale_fraction=0.05,
            ignore_sv_type=False,
            ignore_filter=True,
            cipos_threshold=50,
        )

        # Should contain only the large CNV pair
        assert len(candidates) == 1
        assert ("CNV_large_1", "CNV_large_2") in candidates


class TestGroupCandidatesTransitively:
    """Direct unit tests for _group_candidates_transitively function."""

    def test_empty_set(self):
        """Test that empty candidate set returns empty list."""
        groups = combine_cnv_vcf_utils._group_candidates_transitively(set())
        assert groups == []

    def test_single_pair(self):
        """Test that single pair forms one group."""
        candidates = {("A", "B")}
        groups = combine_cnv_vcf_utils._group_candidates_transitively(candidates)
        assert len(groups) == 1
        assert groups[0] == {"A", "B"}

    def test_transitive_closure(self):
        """Test transitive closure: A-B, B-C forms one group {A,B,C}."""
        candidates = {("A", "B"), ("B", "C")}
        groups = combine_cnv_vcf_utils._group_candidates_transitively(candidates)
        assert len(groups) == 1
        assert groups[0] == {"A", "B", "C"}

    def test_separate_components(self):
        """Test that separate components form separate groups."""
        candidates = {("A", "B"), ("C", "D")}
        groups = combine_cnv_vcf_utils._group_candidates_transitively(candidates)
        assert len(groups) == 2
        assert {"A", "B"} in groups
        assert {"C", "D"} in groups

    def test_complex_graph(self):
        """Test complex graph with multiple components."""
        candidates = {
            ("A", "B"),
            ("B", "C"),
            ("C", "D"),  # Group 1: A-B-C-D
            ("E", "F"),  # Group 2: E-F
            ("G", "H"),
            ("H", "I"),  # Group 3: G-H-I
        }
        groups = combine_cnv_vcf_utils._group_candidates_transitively(candidates)
        assert len(groups) == 3
        assert {"A", "B", "C", "D"} in groups
        assert {"E", "F"} in groups
        assert {"G", "H", "I"} in groups
