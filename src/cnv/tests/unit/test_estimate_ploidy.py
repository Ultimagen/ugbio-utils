import functools
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pysam
import pytest
from ugbio_cnv.estimate_ploidy import (
    _call_aneuploidy,
    _call_whole_genome_ploidy_by_baf,
    _compute_ploidy_from_chr_data,
    _determine_karyotype,
    _is_position_excluded,
    _is_standard_biallelic_snp,
    _ploidy_from_sites,
    _sample_genome,
    _sex_label_from_karyotype,
    estimate_allosomes_from_cram,
    estimate_ploidy_from_coverage,
    estimate_ploidy_from_vcf,
    main,
    parse_mosdepth_summary,
    write_haploid_regions_bed,
)
from ugbio_core.vcfbed.bed_writer import parse_intervals_file

PLOIDY_RESOURCES = Path(__file__).parent.parent / "resources" / "ploidy"


class TestDetermineKaryotype:
    def test_xx(self):
        assert _determine_karyotype(1.0, 0.0) == "XX"

    def test_xy(self):
        assert _determine_karyotype(0.5, 0.5) == "XY"

    def test_xxy(self):
        assert _determine_karyotype(1.0, 0.5) == "XXY"

    def test_xyy(self):
        assert _determine_karyotype(0.5, 1.0) == "XYY"

    def test_x0(self):
        assert _determine_karyotype(0.5, 0.0) == "X0"

    def test_undetermined(self):
        assert _determine_karyotype(2.0, 2.0) == "UNDETERMINED"


class TestSexLabelFromKaryotype:
    def test_female(self):
        assert _sex_label_from_karyotype("XX") == "female"
        assert _sex_label_from_karyotype("XXX") == "female"
        assert _sex_label_from_karyotype("X0") == "female"

    def test_male(self):
        assert _sex_label_from_karyotype("XY") == "male"
        assert _sex_label_from_karyotype("XYY") == "male"
        assert _sex_label_from_karyotype("XXY") == "male"

    def test_unknown(self):
        assert _sex_label_from_karyotype("UNDETERMINED") == "unknown"


class TestCallWholeGenomePloidyByBaf:
    def test_no_bins_is_insufficient_data(self):
        assert _call_whole_genome_ploidy_by_baf(None, 0)["label"] == "INSUFFICIENT_DATA"

    def test_diploid_background_is_diploid(self):
        result = _call_whole_genome_ploidy_by_baf(0.097, 1000)  # highest offset among 50 real diploids
        assert result["label"] == "DIPLOID"

    def test_triploid_is_triploid(self):
        result = _call_whole_genome_ploidy_by_baf(1 / 6, 1000)  # hets at 1/3, 2/3
        assert result["label"] == "TRIPLOID"


class TestStandardBiallelicSnp:
    def test_standard_snp(self):
        assert _is_standard_biallelic_snp("A", ("G",)) is True

    @pytest.mark.parametrize("ref,alts", [("A", ("G", "T")), ("A", ("G", "<NON_REF>")), ("A", ("AT",)), ("N", ("G",))])
    def test_nonstandard_sites_are_excluded(self, ref, alts):
        assert _is_standard_biallelic_snp(ref, alts) is False


class TestComputePloidyFromChrData:
    def test_male_hg38(self):
        chr_data = {f"chr{i}": {"coverage": 50.0, "length": 1e8} for i in range(1, 23)}
        chr_data["chrX"] = {"coverage": 25.0, "length": 1e8}
        chr_data["chrY"] = {"coverage": 25.0, "length": 5e7}
        result = _compute_ploidy_from_chr_data(chr_data)
        assert result["karyotype"] == "XY"
        assert result["sex_label"] == "male"
        assert 0.4 < result["x_ratio"] < 0.6
        assert 0.4 < result["y_ratio"] < 0.6

    def test_female_hg38(self):
        chr_data = {f"chr{i}": {"coverage": 50.0, "length": 1e8} for i in range(1, 23)}
        chr_data["chrX"] = {"coverage": 50.0, "length": 1e8}
        chr_data["chrY"] = {"coverage": 0.1, "length": 5e7}
        result = _compute_ploidy_from_chr_data(chr_data)
        assert result["karyotype"] == "XX"
        assert result["sex_label"] == "female"

    def test_male_b37(self):
        chr_data = {str(i): {"coverage": 40.0, "length": 1e8} for i in range(1, 23)}
        chr_data["X"] = {"coverage": 20.0, "length": 1e8}
        chr_data["Y"] = {"coverage": 20.0, "length": 5e7}
        result = _compute_ploidy_from_chr_data(chr_data)
        assert result["karyotype"] == "XY"

    def test_missing_chry_is_undetermined_not_x0(self):
        # b37 pipeline: chrY is outside the calling regions, so the VCF has no chrY records at all
        chr_data = {str(i): {"coverage": 40.0} for i in range(1, 23)}
        chr_data["X"] = {"coverage": 20.0}
        result = _compute_ploidy_from_chr_data(chr_data)
        assert result["karyotype"] == "UNDETERMINED"
        assert any("No usable chrY data" in warning for warning in result["warnings"])

    def test_non_numeric_autosomes_from_header(self):
        chr_data = {
            "scaffold_a": {"coverage": 50.0, "length": 1e8},
            "scaffold_b": {"coverage": 50.0, "length": 1e8},
            "chrX": {"coverage": 25.0, "length": 1e8},
            "chrY": {"coverage": 25.0, "length": 5e7},
        }
        result = _compute_ploidy_from_chr_data(chr_data, sex_chromosomes=("chrX", "chrY"))
        assert result["karyotype"] == "XY"
        assert {entry["flag"] for entry in result["per_chrom"] if entry["chrom"].startswith("scaffold_")} == {""}

    def test_autosomal_baseline_is_median_ignoring_lengths(self):
        chr_data = {
            "chr1": {"coverage": 40.0, "length": 100},
            "chr2": {"coverage": 50.0, "length": 100},
            "chr3": {"coverage": 80.0, "length": 100},
            "chrX": {"coverage": 25.0, "length": 100},
            "chrY": {"coverage": 25.0, "length": 50},
        }
        result = _compute_ploidy_from_chr_data(chr_data, sex_chromosomes=("chrX", "chrY"))
        assert result["auto_mean"] == 50.0  # median; the length-weighted mean would be 56.67
        assert result["warnings"] == []

    def test_autosomal_baseline_is_robust_to_outlier_chromosome(self):
        chr_data = {
            "chr1": {"coverage": 50.0},
            "chr2": {"coverage": 50.0},
            "chr3": {"coverage": 1000.0},
            "chrX": {"coverage": 25.0},
            "chrY": {"coverage": 25.0},
        }
        result = _compute_ploidy_from_chr_data(chr_data, sex_chromosomes=("chrX", "chrY"))
        assert result["auto_mean"] == 50.0
        assert result["karyotype"] == "XY"
        assert result["warnings"] == []

    def test_no_autosomes_returns_undetermined(self):
        result = _compute_ploidy_from_chr_data({"chrX": {"coverage": 25.0}})
        assert result["karyotype"] == "UNDETERMINED"
        assert result["per_chrom"] == []
        assert "No autosomal contigs" in result["warnings"][0]

    def test_zero_coverage_returns_undetermined(self):
        chr_data = {f"chr{i}": {"coverage": 0.0, "length": 1e8} for i in range(1, 23)}
        result = _compute_ploidy_from_chr_data(chr_data)
        assert result["karyotype"] == "UNDETERMINED"
        assert result["per_chrom"] == []
        assert "Autosomal median coverage is 0" in result["warnings"][0]


class TestEstimatePloidyFromCoverage:
    def test_mosdepth_male(self, tmp_path):
        tsv = tmp_path / "summary.txt"
        lines = ["chrom\tlength\tbases\tmean\tmin_cov\tmax_cov\n"]
        for i in range(1, 23):
            lines.append(f"chr{i}_region\t100000000\t5000000000\t50.0\t0\t200\n")
        lines.append("chrX_region\t100000000\t2500000000\t25.0\t0\t100\n")
        lines.append("chrY_region\t50000000\t1250000000\t25.0\t0\t100\n")
        lines.append("chrX_region\t100000000\t2500000000\t25.0\t0\t100\n")
        lines.append("chrY\t50000000\t1250000000\t1.0\t0\t100\n")
        lines.append("chrY_region\t50000000\t1250000000\t25.0\t0\t100\n")
        lines.append("total\t3000000000\t150000000000\t50.0\t0\t200\n")
        tsv.write_text("".join(lines))

        mosdepth_df = parse_mosdepth_summary(tsv)
        result = estimate_ploidy_from_coverage(mosdepth_df)
        assert result["karyotype"] == "XY"
        assert result["sex_label"] == "male"
        assert result["source"] == "mosdepth"
        assert {entry["chrom"] for entry in result["per_chrom"]} == {
            *(f"chr{i}" for i in range(1, 23)),
            "chrX",
            "chrY",
        }
        assert all(entry["mean_cov"] != 1.0 for entry in result["per_chrom"])

    def test_mosdepth_baseline_is_median_of_autosomes(self, tmp_path):
        # Trisomy of the longest chromosome would pull a length-weighted baseline up to ~52x; the median stays 50x
        tsv = tmp_path / "summary.txt"
        lines = ["chrom\tlength\tbases\tmean\tmin_cov\tmax_cov\n", "chr1_region\t250000000\t1\t75.0\t0\t200\n"]
        lines += [f"chr{i}_region\t100000000\t1\t50.0\t0\t200\n" for i in range(2, 23)]
        lines += ["chrX_region\t150000000\t1\t25.0\t0\t100\n", "chrY_region\t50000000\t1\t25.0\t0\t100\n"]
        tsv.write_text("".join(lines))

        result = estimate_ploidy_from_coverage(parse_mosdepth_summary(tsv))

        ploidy = {entry["chrom"]: entry["ploidy"] for entry in result["per_chrom"]}
        assert result["auto_mean"] == 50.0
        assert ploidy["chr1"] == 3.0
        assert all(ploidy[f"chr{i}"] == 2.0 for i in range(2, 23))
        assert result["karyotype"] == "XY"

    def test_mosdepth_requires_region_rows(self):
        summary = pd.DataFrame(
            {
                "chrom": ["chr1", "total"],
                "length": [100, 100],
                "mean": [50.0, 50.0],
            }
        )

        with pytest.raises(ValueError, match=r"no \*_region rows"):
            estimate_ploidy_from_coverage(summary)


class TestEstimatePloidyFromVcf:
    @staticmethod
    def _make_ploidy_vcf(tmp_path, sample_name="SAMPLE"):
        vcf_path = tmp_path / "calls.vcf.gz"
        header = pysam.VariantHeader()
        header.add_sample(sample_name)
        for chrom in ("chr1", "chr2", "chrX", "chrY"):
            header.add_line(f"##contig=<ID={chrom}>")
        header.add_line('##FILTER=<ID=PASS,Description="All filters passed">')
        header.add_line('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">')
        header.add_line('##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Depth">')
        header.add_line('##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allele depths">')
        with pysam.VariantFile(vcf_path, "wz", header=header) as writer:
            chromosome_calls = (("chr1", 50, (0, 1)), ("chr2", 50, (0, 1)), ("chrX", 25, (1, 1)), ("chrY", 25, (1, 1)))
            for chrom, depth, genotype in chromosome_calls:
                for pos in range(1, 21):
                    record = writer.new_record(contig=chrom, start=pos - 1, stop=pos, alleles=("A", "G"))
                    record.samples[sample_name]["GT"] = genotype
                    record.samples[sample_name]["DP"] = depth
                    record.samples[sample_name]["AD"] = (depth // 2, depth // 2)
                    writer.write(record)
        return vcf_path

    @staticmethod
    def _make_full_genome_vcf(tmp_path, sample_name="SAMPLE"):
        vcf_path = tmp_path / "full_genome_calls.vcf.gz"
        header = pysam.VariantHeader()
        header.add_sample(sample_name)
        contigs = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
        for chrom in contigs:
            header.add_line(f"##contig=<ID={chrom},length=100000000>")
        header.add_line('##FILTER=<ID=PASS,Description="All filters passed">')
        header.add_line('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">')
        header.add_line('##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Depth">')
        header.add_line('##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allele depths">')
        with pysam.VariantFile(vcf_path, "wz", header=header) as writer:
            for chrom in contigs:
                if chrom in ("chrX", "chrY"):
                    depth, genotype, ad = 25, (1, 1), (0, 25)
                else:
                    depth, genotype, ad = 50, (0, 1), (25, 25)
                for pos in range(1, 31):
                    record = writer.new_record(contig=chrom, start=pos - 1, stop=pos, alleles=("A", "G"))
                    record.filter.add("PASS")
                    record.samples[sample_name]["GT"] = genotype
                    record.samples[sample_name]["DP"] = depth
                    record.samples[sample_name]["AD"] = ad
                    writer.write(record)
        pysam.tabix_index(str(vcf_path), preset="vcf", force=True)
        return vcf_path

    def test_reads_unfiltered_snps_with_pysam(self, tmp_path):
        vcf_path = self._make_ploidy_vcf(tmp_path)

        coverage_result, baf_result = estimate_ploidy_from_vcf(vcf_path, "SAMPLE")

        assert coverage_result["karyotype"] == "XY"
        assert coverage_result["source"] == "VCF SNP median DP"
        assert baf_result["label"] == "INSUFFICIENT_DATA"

    def test_selects_requested_sample_from_multi_sample_vcf(self, tmp_path):
        vcf_path = tmp_path / "multi_sample.vcf.gz"
        header = pysam.VariantHeader()
        header.add_sample("UNSELECTED")
        header.add_sample("SELECTED")
        for chrom in ("chr1", "chr2", "chrX", "chrY"):
            header.add_line(f"##contig=<ID={chrom}>")
        header.add_line('##FILTER=<ID=PASS,Description="All filters passed">')
        header.add_line('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">')
        header.add_line('##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Depth">')
        header.add_line('##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allele depths">')
        with pysam.VariantFile(vcf_path, "wz", header=header) as writer:
            chromosome_calls = (("chr1", 50), ("chr2", 50), ("chrX", 25), ("chrY", 25))
            for chrom, selected_depth in chromosome_calls:
                for pos in range(1, 21):
                    record = writer.new_record(contig=chrom, start=pos - 1, stop=pos, alleles=("A", "G"))
                    record.filter.add("PASS")
                    record.samples["UNSELECTED"]["GT"] = (0, 0)
                    record.samples["UNSELECTED"]["DP"] = 1
                    record.samples["UNSELECTED"]["AD"] = (1, 0)
                    record.samples["SELECTED"]["GT"] = (0, 1) if chrom[3:].isdigit() else (1, 1)
                    record.samples["SELECTED"]["DP"] = selected_depth
                    record.samples["SELECTED"]["AD"] = (selected_depth // 2, selected_depth // 2)
                    writer.write(record)

        coverage_result, _ = estimate_ploidy_from_vcf(vcf_path, "SELECTED")

        assert coverage_result["karyotype"] == "XY"

    def test_missing_sample_raises(self, tmp_path):
        vcf_path = tmp_path / "empty.vcf.gz"
        header = pysam.VariantHeader()
        header.add_sample("PRESENT")
        with pysam.VariantFile(vcf_path, "wz", header=header):
            pass

        with pytest.raises(ValueError, match="'MISSING' is not present in VCF"):
            estimate_ploidy_from_vcf(vcf_path, "MISSING")

    @staticmethod
    def _make_vcf_with_chry_exclusion_case(tmp_path, sample_name="SAMPLE"):
        """chrY has 21 high-DP SNPs (positions 1-21, meant to be excluded) and 20 low-DP SNPs
        (positions 101-120, meant to be kept). chrX/autosomes are diploid-like (X ratio ~1.0),
        so the karyotype call is driven entirely by whether the high-DP chrY SNPs are excluded.
        """
        vcf_path = tmp_path / "chry_exclusion.vcf.gz"
        header = pysam.VariantHeader()
        header.add_sample(sample_name)
        for chrom in ("chr1", "chr2", "chrX", "chrY"):
            header.add_line(f"##contig=<ID={chrom}>")
        header.add_line('##FILTER=<ID=PASS,Description="All filters passed">')
        header.add_line('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">')
        header.add_line('##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Depth">')
        header.add_line('##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allele depths">')
        with pysam.VariantFile(vcf_path, "wz", header=header) as writer:

            def write_records(chrom, positions, depth):
                for pos in positions:
                    record = writer.new_record(contig=chrom, start=pos - 1, stop=pos, alleles=("A", "G"))
                    record.filter.add("PASS")
                    record.samples[sample_name]["GT"] = (0, 1)
                    record.samples[sample_name]["DP"] = depth
                    record.samples[sample_name]["AD"] = (depth // 2, depth // 2)
                    writer.write(record)

            write_records("chr1", range(1, 21), 40)
            write_records("chr2", range(1, 21), 40)
            write_records("chrX", range(1, 21), 40)
            write_records("chrY", range(1, 22), 20)  # high-DP: should be excluded
            write_records("chrY", range(101, 121), 4)  # low-DP: should be kept
        return vcf_path

    def test_exclude_regions_bed_filters_high_coverage_chry_positions(self, tmp_path):
        vcf_path = self._make_vcf_with_chry_exclusion_case(tmp_path)
        exclude_bed = tmp_path / "exclude.bed"
        exclude_bed.write_text("chrY\t0\t21\n")

        coverage_without_exclusion, _ = estimate_ploidy_from_vcf(vcf_path, "SAMPLE")
        coverage_with_exclusion, _ = estimate_ploidy_from_vcf(vcf_path, "SAMPLE", exclude_regions_bed=exclude_bed)

        assert coverage_without_exclusion["karyotype"] == "XXY"
        assert coverage_with_exclusion["karyotype"] == "XX"


class TestIsPositionExcluded:
    @staticmethod
    def _regions(tmp_path, *lines):
        bed_path = tmp_path / "regions.bed"
        bed_path.write_text("\n".join(lines) + "\n")
        return parse_intervals_file(str(bed_path)).set_index("chromosome")  # as in _iter_vcf_sites

    def test_unsorted_nested_and_adjacent_intervals(self, tmp_path):
        regions = self._regions(
            tmp_path, "chrY\t100\t200", "chrY\t0\t50", "chrY\t50\t60", "chrY\t110\t120", "chrX\t10\t20"
        )
        excluded = [pos for pos in range(1, 202) if _is_position_excluded(regions, "chrY", pos)]
        assert excluded == [*range(1, 61), *range(101, 201)]  # 1-based positions covered by [0, 60) and [100, 200)

    def test_position_inside_interval(self, tmp_path):
        regions = self._regions(tmp_path, "chrY\t100\t200")
        assert _is_position_excluded(regions, "chrY", 101) is True  # 1-based pos 101 -> 0-based 100 (start)
        assert _is_position_excluded(regions, "chrY", 200) is True  # 1-based pos 200 -> 0-based 199 (< end=200)

    def test_position_outside_interval_boundaries(self, tmp_path):
        regions = self._regions(tmp_path, "chrY\t100\t200")
        assert _is_position_excluded(regions, "chrY", 100) is False  # 0-based 99, before start
        assert _is_position_excluded(regions, "chrY", 201) is False  # 0-based 200, == end (exclusive)

    def test_missing_chromosome_returns_false(self, tmp_path):
        regions = self._regions(tmp_path, "chrY\t0\t10")
        assert _is_position_excluded(regions, "chrX", 5) is False

    def test_no_regions_returns_false(self):
        empty = pd.DataFrame({"start": [], "end": []}, index=pd.Index([], name="chromosome"))
        assert _is_position_excluded(empty, "chrY", 5) is False

    def test_position_inside_longer_interval_masked_by_nested_interval(self, tmp_path):
        # Regression: a shorter interval starting later (20, 30) must not hide positions covered
        # only by the earlier, longer interval (10, 100), e.g. 0-based position 95.
        regions = self._regions(tmp_path, "chrY\t10\t100", "chrY\t20\t30")
        assert _is_position_excluded(regions, "chrY", 96) is True  # 1-based 96 -> 0-based 95


class TestEstimatePloidyCli:
    def test_vcf_mode_writes_report(self, tmp_path):
        vcf_path = TestEstimatePloidyFromVcf._make_ploidy_vcf(tmp_path)

        main(["--vcf", str(vcf_path), "--sample-id", "SAMPLE", "--output-dir", str(tmp_path)])

        report = tmp_path / "SAMPLE.ploidy_report.txt"
        report_text = report.read_text()
        assert report.exists()
        assert "Karyotype:" in report_text
        assert "Coverage source:    VCF SNP median DP" in report_text

    def test_vcf_mode_report_matches_reference(self, tmp_path):
        vcf_path = TestEstimatePloidyFromVcf._make_full_genome_vcf(tmp_path, sample_name="SAMPLE")

        main(["--vcf", str(vcf_path), "--sample-id", "SAMPLE", "--output-dir", str(tmp_path)])

        report_path = tmp_path / "SAMPLE.ploidy_report.txt"
        report_text = report_path.read_text()

        expected_lines = [
            "======================================================",
            "  Genome Ploidy Report - SAMPLE",
            "======================================================",
            "",
            "  Karyotype:          XY",
            "  X/Y coverage ratios: X=0.500, Y=0.500",
            "  Whole-genome ploidy: INSUFFICIENT_DATA (no 1 Mb bin with >= 50 het SNPs)",
            "  Autosomal median cov: 50.0x",
            "  Coverage source:    VCF SNP median DP",
            f"  Input:              {vcf_path}",
            "",
            "  Per-chromosome ploidy (relative to autosome median = 2.0):",
            "  --------------------------------------------------------",
        ]
        for i in range(1, 23):
            chrom = f"chr{i}"
            expected_lines.append(f"  {chrom:<6s} ploidy=2.000  cov=50.00x  af_skew=NA   ")
        expected_lines.extend(
            [
                "  chrX   ploidy=1.000  cov=25.00x  .",
                "  chrY   ploidy=1.000  cov=25.00x  .",
                "",
                "  (. = sex chromosome)",
                "",
                "  No autosomal aneuploidy detected.",
                "",
                "",
            ]
        )
        assert report_text == "\n".join(expected_lines)

        # Verify downstream WDL task karyotype extraction logic
        cmd = f"grep -m1 'Karyotype:' {report_path} | awk '{{print $NF}}'"
        karyotype_extracted = subprocess.check_output(cmd, shell=True, text=True).strip()
        assert karyotype_extracted == "XY"

    def test_mosdepth_summary_mode_writes_report(self, tmp_path):
        summary_path = tmp_path / "summary.txt"
        lines = ["chrom\tlength\tbases\tmean\tmin_cov\tmax_cov\n"]
        for i in range(1, 23):
            lines.append(f"chr{i}_region\t100000000\t5000000000\t50.0\t0\t200\n")
        lines.append("chrX_region\t100000000\t2500000000\t25.0\t0\t100\n")
        lines.append("chrY_region\t50000000\t1250000000\t25.0\t0\t100\n")
        summary_path.write_text("".join(lines))

        main(["--mosdepth-summary", str(summary_path), "--sample-id", "SAMPLE", "--output-dir", str(tmp_path)])

        report = tmp_path / "SAMPLE.ploidy_report.txt"
        report_text = report.read_text()
        assert report.exists()
        assert "Karyotype:          XY" in report_text
        assert "Coverage source:    mosdepth" in report_text

    def test_mosdepth_mode_report_matches_reference(self, tmp_path):
        summary_path = tmp_path / "summary.txt"
        lines = ["chrom\tlength\tbases\tmean\tmin_cov\tmax_cov\n"]
        for i in range(1, 23):
            lines.append(f"chr{i}_region\t100000000\t5000000000\t50.0\t0\t200\n")
        lines.append("chrX_region\t100000000\t2500000000\t25.0\t0\t100\n")
        lines.append("chrY_region\t50000000\t1250000000\t25.0\t0\t100\n")
        summary_path.write_text("".join(lines))

        main(["--mosdepth-summary", str(summary_path), "--sample-id", "SAMPLE", "--output-dir", str(tmp_path)])

        report_path = tmp_path / "SAMPLE.ploidy_report.txt"
        report_text = report_path.read_text()

        expected_head = [
            "======================================================",
            "  Genome Ploidy Report - SAMPLE",
            "======================================================",
            "",
            "  Karyotype:          XY",
            "  X/Y coverage ratios: X=0.500, Y=0.500",
            "  Whole-genome ploidy: N/A (CRAM mode, no BAF)",
            "  Autosomal median cov: 50.0x",
            "  Coverage source:    mosdepth",
            f"  Input:              {summary_path}",
        ]
        assert "\n".join(expected_head) in report_text

        # Verify downstream WDL task karyotype extraction logic
        cmd = f"grep -m1 'Karyotype:' {report_path} | awk '{{print $NF}}'"
        karyotype_extracted = subprocess.check_output(cmd, shell=True, text=True).strip()
        assert karyotype_extracted == "XY"

    def test_vcf_and_mosdepth_summary_are_mutually_exclusive(self, tmp_path):
        vcf_path = TestEstimatePloidyFromVcf._make_ploidy_vcf(tmp_path)
        summary_path = tmp_path / "summary.txt"
        summary_path.write_text("chrom\tlength\tbases\tmean\tmin_cov\tmax_cov\n")

        with pytest.raises(SystemExit):
            main(
                [
                    "--vcf",
                    str(vcf_path),
                    "--mosdepth-summary",
                    str(summary_path),
                    "--sample-id",
                    "SAMPLE",
                ]
            )


class TestCallAneuploidy:
    @pytest.mark.parametrize(
        "ploidy, af_skew, expected",
        [
            (2.05, 1.0, False),  # normal
            (2.4, 1.0, False),  # GC-like coverage shift, balanced alleles
            (2.35, 1.6, True),  # mosaic gain confirmed by allelic skew
            (1.0, None, True),  # monosomy: no hets, called on coverage alone
            (4.0, 1.0, True),  # balanced tetrasomy, called on coverage alone
        ],
    )
    def test_calling_rule(self, ploidy, af_skew, expected):
        assert _call_aneuploidy(ploidy, af_skew, 0.15, 0.8, 1.2) is expected


class TestPloidyFromSitesCoverage:
    AUTOSOMES = [f"chr{i}" for i in range(1, 23)]

    @staticmethod
    def _sites(depths_by_chrom: dict[str, list[int]]):
        return [
            (chrom, pos, dp, 0, 0, False)
            for chrom, depths in depths_by_chrom.items()
            for pos, dp in enumerate(depths, start=1)
        ]

    def _ploidy(self, depths_by_chrom):
        coverage_result, _ = _ploidy_from_sites(self._sites(depths_by_chrom))
        return {entry["chrom"]: entry["ploidy"] for entry in coverage_result["per_chrom"]}

    def test_dp_above_three_times_autosomal_median_is_dropped(self):
        depths = {chrom: [40] * 30 for chrom in self.AUTOSOMES}
        depths["chr1"] = [40] * 20 + [500] * 30  # majority of outliers: uncapped median would be 500
        assert self._ploidy(depths)["chr1"] == 2.0

    def test_chromosome_with_too_few_sites_is_skipped(self):
        depths = {chrom: [40] * 30 for chrom in self.AUTOSOMES}
        depths["chr2"] = [40] * 19
        depths["chr3"] = [40] * 20
        ploidy = self._ploidy(depths)
        assert "chr2" not in ploidy
        assert ploidy["chr3"] == 2.0

    def test_baseline_is_robust_to_three_trisomies(self):
        depths = {chrom: [38, 40, 42] * 10 for chrom in self.AUTOSOMES}
        for chrom in ("chr1", "chr2", "chr3"):
            depths[chrom] = [57, 60, 63] * 10
        ploidy = self._ploidy(depths)
        assert all(ploidy[chrom] == pytest.approx(3.0, abs=0.05) for chrom in ("chr1", "chr2", "chr3"))
        assert all(ploidy[chrom] == pytest.approx(2.0, abs=0.05) for chrom in self.AUTOSOMES[3:])


class TestRealDataSites:
    """De-identified site tables of real VCFs (built with resources/ploidy/make_ploidy_sites.py)."""

    @staticmethod
    @functools.cache
    def _run(name):
        sites = pd.read_parquet(PLOIDY_RESOURCES / f"{name}.ploidy_sites.parquet")
        coverage_result, baf_result = _ploidy_from_sites(sites.itertuples(index=False, name=None))
        per_chrom = {entry["chrom"]: entry for entry in coverage_result["per_chrom"]}
        flagged = {chrom for chrom, entry in per_chrom.items() if entry.get("aneuploid")}
        return coverage_result, baf_result, per_chrom, flagged

    @pytest.mark.parametrize(
        "name, karyotype, expected_flagged",
        [
            ("MTB054", "XY", {"chr8"}),  # mosaic chr8 gain
            ("CKT416", "XY", {"chr21"}),  # trisomy 21
            ("PMR903", "XX", set()),  # GC-driven coverage artifacts, balanced alleles
            ("HG002_DS5X", "XY", set()),  # 5x down-sample of a normal male
        ],
    )
    def test_karyotype_and_flagged_chromosomes(self, name, karyotype, expected_flagged):
        coverage_result, baf_result, _, flagged = self._run(name)
        assert coverage_result["karyotype"] == karyotype
        assert flagged == expected_flagged
        assert baf_result["label"] == "DIPLOID"

    @pytest.mark.parametrize("name, chrom", [("MTB054", "chr8"), ("CKT416", "chr21")])
    def test_het_af_agrees_with_coverage(self, name, chrom):
        _, _, per_chrom, _ = self._run(name)
        gain = per_chrom[chrom]["ploidy"] - 2  # fraction of cells carrying one extra copy
        assert per_chrom[chrom]["het_af"] == pytest.approx((1 + gain) / (2 + gain), abs=0.03)

    def test_simulated_triploid_is_triploid(self):
        # Re-draw the het allele counts of a real diploid at AF 1/3 or 2/3, keeping its depths and positions
        sites = pd.read_parquet(PLOIDY_RESOURCES / "MTB054.ploidy_sites.parquet")
        rng = np.random.default_rng(0)
        het = sites["is_het"].to_numpy()
        total = (sites["ref_ad"] + sites["alt_ad"]).to_numpy()[het]
        alt = rng.binomial(total, rng.choice([1 / 3, 2 / 3], len(total)))
        sites.loc[het, "alt_ad"] = alt.astype(np.int16)
        sites.loc[het, "ref_ad"] = (total - alt).astype(np.int16)
        _, baf_result = _ploidy_from_sites(sites.itertuples(index=False, name=None))
        assert baf_result["label"] == "TRIPLOID"
        assert baf_result["genome_het_offset"] == pytest.approx(1 / 6, abs=0.02)

    def test_male_without_chry_calls_is_undetermined(self):
        sites = pd.read_parquet(PLOIDY_RESOURCES / "MTB054.ploidy_sites.parquet")
        no_chry = sites[sites["chrom"] != "chrY"]
        coverage_result, _ = _ploidy_from_sites(no_chry.itertuples(index=False, name=None))
        assert coverage_result["karyotype"] == "UNDETERMINED"
        assert any("No usable chrY data" in warning for warning in coverage_result["warnings"])


SEX_TEST_CONTIGS = ("chr1", "chr7", "chr12", "chrX", "chrY")
SEX_TEST_CONTIG_LENGTH = 20_000
SEX_TEST_READ_LENGTH = 100
SEX_TEST_READS_PER_COPY = 100


def _write_reference(tmp_path: Path, contigs=SEX_TEST_CONTIGS, length=SEX_TEST_CONTIG_LENGTH) -> tuple[Path, dict]:
    rng = np.random.default_rng(0)
    sequences = {contig: "".join(rng.choice(list("ACGT"), length)) for contig in contigs}
    fasta = tmp_path / "ref.fa"
    fasta.write_text("".join(f">{contig}\n{seq}\n" for contig, seq in sequences.items()))
    pysam.faidx(str(fasta))
    return fasta, sequences


def _write_cram(tmp_path: Path, fasta: Path, sequences: dict, reads: list[tuple[str, int, int]], name="sample") -> Path:
    """reads: (contig, start, flag); written sorted, each a perfect 100 bp match."""
    header = {
        "HD": {"VN": "1.6", "SO": "coordinate"},
        "SQ": [{"SN": contig, "LN": len(seq)} for contig, seq in sequences.items()],
    }
    contig_index = {contig: i for i, contig in enumerate(sequences)}
    cram = tmp_path / f"{name}.cram"
    with pysam.AlignmentFile(str(cram), "wc", header=header, reference_filename=str(fasta)) as writer:
        for i, (contig, start, flag) in enumerate(sorted(reads, key=lambda r: (contig_index[r[0]], r[1]))):
            read = pysam.AlignedSegment(writer.header)
            read.query_name = f"read{i}"
            read.flag = flag
            read.reference_id = contig_index[contig]
            read.reference_start = start
            read.mapping_quality = 60
            read.query_sequence = sequences[contig][start : start + SEX_TEST_READ_LENGTH]
            read.query_qualities = pysam.qualitystring_to_array("I" * SEX_TEST_READ_LENGTH)
            if not flag & 0x4:
                read.cigar = [(0, SEX_TEST_READ_LENGTH)]
            writer.write(read)
    pysam.index(str(cram))
    return cram


def _even_reads(contig: str, n_reads: int, start=0, end=SEX_TEST_CONTIG_LENGTH) -> list[tuple[str, int, int]]:
    last_start = end - SEX_TEST_READ_LENGTH
    return [(contig, int(pos), 0) for pos in np.linspace(start, last_start, n_reads)] if n_reads else []


def _karyotype_cram(tmp_path: Path, x_copies: int, y_copies: int, extra_reads=()) -> tuple[Path, Path]:
    fasta, sequences = _write_reference(tmp_path)
    copies = {"chr1": 2, "chr7": 2, "chr12": 2, "chrX": x_copies, "chrY": y_copies}
    reads = [read for contig, n in copies.items() for read in _even_reads(contig, n * SEX_TEST_READS_PER_COPY)]
    return _write_cram(tmp_path, fasta, sequences, reads + list(extra_reads)), fasta


def _regions_bed(tmp_path: Path, contigs=SEX_TEST_CONTIGS, name="regions.bed") -> Path:
    bed = tmp_path / name
    bed.write_text("".join(f"{contig}\t0\t{SEX_TEST_CONTIG_LENGTH}\n" for contig in contigs))
    return bed


class TestSampleGenome:
    @staticmethod
    def _sample(tmp_path, regions_text, suffix=".bed", exclude_text=None, bp_per_chrom=10_000_000):
        regions = tmp_path / f"regions{suffix}"
        regions.write_text(regions_text)
        exclude = None
        if exclude_text is not None:
            exclude = tmp_path / "exclude dir" / "exclude.bed"
            exclude.parent.mkdir()
            exclude.write_text(exclude_text)
        chromosomes = {"x", "y", "1"}
        tiles = _sample_genome(regions, exclude, chromosomes, bp_per_chrom, tmp_path)
        return list(tiles.itertuples(index=False, name=None))

    def test_keeps_only_requested_chromosomes(self, tmp_path):
        tiles = self._sample(tmp_path, "chr1\t0\t100\nchr2\t0\t100\nX\t0\t100\nchrY\t0\t100\n")
        assert tiles == [("chr1", 0, 100, 0), ("X", 0, 100, 0), ("chrY", 0, 100, 0)]

    def test_interval_list_is_read_as_one_based(self, tmp_path):
        header = "@HD\tVN:1.6\n@SQ\tSN:chrX\tLN:100\n"
        tiles = self._sample(tmp_path, header + "chrX\t11\t20\t+\t.\n", suffix=".interval_list")
        assert tiles == [("chrX", 10, 20, 0)]

    def test_exclude_regions_are_subtracted(self, tmp_path):
        tiles = self._sample(tmp_path, "chrY\t0\t100\n", exclude_text="chrY\t40\t60\n")
        assert tiles == [("chrY", 0, 40, 0), ("chrY", 60, 100, 0)]

    def test_bins_are_1mb_and_evenly_thinned(self, tmp_path):
        tiles = self._sample(tmp_path, "chrX\t0\t10000000\n", bp_per_chrom=4_000_000)
        assert tiles == [("chrX", s, s + 1_000_000, s // 1_000_000) for s in (0, 3_000_000, 6_000_000, 9_000_000)]

    def test_bins_hold_1mb_callable_across_fragments(self, tmp_path):
        regions = "chrX\t0\t600000\nchrX\t2000000\t2700000\nchrX\t5000000\t5100000\n"
        tiles = self._sample(tmp_path, regions)
        assert tiles == [
            ("chrX", 0, 600_000, 0),
            ("chrX", 2_000_000, 2_400_000, 0),
            ("chrX", 2_400_000, 2_700_000, 1),
            ("chrX", 5_000_000, 5_100_000, 1),
        ]

    def test_no_regions_on_requested_chromosomes_raises(self, tmp_path):
        with pytest.raises(ValueError, match="has no regions"):
            self._sample(tmp_path, "chr2\t0\t100\n")


class TestEstimateAllosomesFromCram:
    @pytest.mark.parametrize(
        "x_copies,y_copies,karyotype",
        [(1, 1, "XY"), (2, 0, "XX"), (2, 1, "XXY"), (1, 0, "X0"), (3, 0, "XXX"), (1, 2, "XYY")],
    )
    def test_karyotypes(self, tmp_path, x_copies, y_copies, karyotype):
        cram, fasta = _karyotype_cram(tmp_path, x_copies, y_copies)
        result = estimate_allosomes_from_cram(cram, fasta, _regions_bed(tmp_path))
        assert result["karyotype"] == karyotype
        assert result["x_ratio"] == pytest.approx(x_copies / 2, abs=0.02)
        assert result["y_ratio"] == pytest.approx(y_copies / 2, abs=0.02)
        assert result["source"] == "cram-sex-only"

    def test_no_exclude_bed_warns(self, tmp_path):
        cram, fasta = _karyotype_cram(tmp_path, 1, 1)
        result = estimate_allosomes_from_cram(cram, fasta, _regions_bed(tmp_path))
        assert any("No chrY exclude regions given" in warning for warning in result["warnings"])

    def test_exclude_bed_drops_inflated_chry_region(self, tmp_path):
        inflated = _even_reads("chrY", 400, start=0, end=5_000)
        cram, fasta = _karyotype_cram(tmp_path, 1, 1, extra_reads=inflated)
        exclude = tmp_path / "exclude.bed"
        exclude.write_text("chrY\t0\t5000\n")
        result = estimate_allosomes_from_cram(cram, fasta, _regions_bed(tmp_path), exclude_regions_bed=exclude)
        assert result["karyotype"] == "XY"
        assert result["y_ratio"] == pytest.approx(0.5, abs=0.05)
        assert not any("No chrY exclude regions given" in warning for warning in result["warnings"])

    def test_outlier_bin_does_not_inflate_chromosome(self, tmp_path, monkeypatch):
        # chrY split into 4 bins; one carries 20x the expected depth (like the chrY:56.7 Mb satellite)
        monkeypatch.setattr("ugbio_cnv.estimate_ploidy._TILE_BP", 5_000)
        inflated = _even_reads("chrY", 2_000, start=0, end=5_000)
        cram, fasta = _karyotype_cram(tmp_path, 1, 1, extra_reads=inflated)
        regions = _regions_bed(tmp_path, contigs=("chr1", "chr7", "chr12", "chrX"))
        regions.write_text(
            regions.read_text() + "".join(f"chrY\t{start}\t{start + 5_000}\n" for start in range(0, 20_000, 5_000))
        )
        exclude = tmp_path / "exclude.bed"
        exclude.write_text("chrY\t20000\t30000\n")
        result = estimate_allosomes_from_cram(cram, fasta, regions, exclude_regions_bed=exclude)
        assert result["karyotype"] == "XY"
        assert result["y_ratio"] == pytest.approx(0.5, abs=0.05)

    def test_short_fragments_weigh_by_length(self, tmp_path, monkeypatch):
        # chrY: two 4 kb intervals at 1 copy and many 100 bp slivers at 2 copies, like the hg38 chrY minus exclude BED
        monkeypatch.setattr("ugbio_cnv.estimate_ploidy._TILE_BP", 2_000)
        long_y = [(0, 4_000), (4_000, 8_000)]
        slivers = [(start, start + 100) for start in range(8_000, 20_000, 400)]
        y_reads = _even_reads("chrY", 40, start=0, end=8_000) + [
            read for start, end in slivers for read in _even_reads("chrY", 1, start=start, end=end)
        ]
        fasta, sequences = _write_reference(tmp_path)
        copies = {"chr1": 2, "chr7": 2, "chr12": 2, "chrX": 1}
        reads = [read for contig, n in copies.items() for read in _even_reads(contig, n * SEX_TEST_READS_PER_COPY)]
        cram = _write_cram(tmp_path, fasta, sequences, reads + y_reads)
        regions = _regions_bed(tmp_path, contigs=("chr1", "chr7", "chr12", "chrX"))
        regions.write_text(regions.read_text() + "".join(f"chrY\t{s}\t{e}\n" for s, e in long_y + slivers))
        result = estimate_allosomes_from_cram(cram, fasta, regions, bp_per_chrom=10_000)
        assert result["karyotype"] == "XY"
        assert result["y_ratio"] == pytest.approx(0.5, abs=0.05)

    def test_missing_chry_regions_is_undetermined(self, tmp_path):
        cram, fasta = _karyotype_cram(tmp_path, 1, 1)
        regions = _regions_bed(tmp_path, contigs=("chr1", "chr7", "chr12", "chrX"))
        result = estimate_allosomes_from_cram(cram, fasta, regions)
        assert result["karyotype"] == "UNDETERMINED"
        assert any("No usable chrY data" in warning for warning in result["warnings"])

    def test_autosome_without_regions_raises(self, tmp_path):
        cram, fasta = _karyotype_cram(tmp_path, 1, 1)
        with pytest.raises(ValueError, match="autosome.*chr5"):
            estimate_allosomes_from_cram(cram, fasta, _regions_bed(tmp_path), autosomes=("chr1", "chr7", "chr5"))

    def test_regions_contig_absent_from_reference_raises(self, tmp_path):
        cram, fasta = _karyotype_cram(tmp_path, 1, 1)
        regions = tmp_path / "regions.bed"
        regions.write_text(_regions_bed(tmp_path, name="base.bed").read_text() + "chr22\t0\t1000\n")
        with pytest.raises(ValueError, match="chr22.*fai"):
            estimate_allosomes_from_cram(cram, fasta, regions, autosomes=("chr1", "chr7", "chr22"))


class TestSexOnlyCli:
    @staticmethod
    def _args(cram, fasta, regions, tmp_path):
        return [
            "--cram",
            str(cram),
            "--sex-only",
            "--reference",
            str(fasta),
            "--regions-bed",
            str(regions),
            "--sample-id",
            "SAMPLE",
            "--output-dir",
            str(tmp_path),
        ]

    def test_report_matches_reference(self, tmp_path):
        cram, fasta = _karyotype_cram(tmp_path, 1, 1)
        exclude = tmp_path / "exclude.bed"
        exclude.write_text("chrY\t20000\t30000\n")  # suppresses the no-exclude warning only
        args = self._args(cram, fasta, _regions_bed(tmp_path), tmp_path)
        main([*args, "--exclude-regions-bed", str(exclude)])

        report_path = tmp_path / "SAMPLE.ploidy_report.txt"
        expected_lines = [
            "======================================================",
            "  Genome Ploidy Report - SAMPLE",
            "======================================================",
            "",
            "  Karyotype:          XY",
            "  X/Y coverage ratios: X=0.500, Y=0.500",
            "  Whole-genome ploidy: N/A (CRAM mode, no BAF)",
            "  Autosomal median cov: 1.0x",
            "  Coverage source:    cram-sex-only",
            f"  Input:              {cram}",
            "",
            "  Per-chromosome ploidy (relative to autosome median = 2.0):",
            "  --------------------------------------------------------",
            "  chr1   ploidy=2.000  cov=1.00x   ",
            "  chr7   ploidy=2.000  cov=1.00x   ",
            "  chr12  ploidy=2.000  cov=1.00x   ",
            "  chrX   ploidy=1.000  cov=0.50x  .",
            "  chrY   ploidy=1.000  cov=0.50x  .",
            "",
            "  (. = sex chromosome)",
            "",
            "  Autosomal aneuploidy not assessed in sex-only mode.",
            "",
            "",
        ]
        assert report_path.read_text() == "\n".join(expected_lines)
        cmd = f"grep -m1 'Karyotype:' {report_path} | awk '{{print $NF}}'"
        karyotype = subprocess.check_output(cmd, shell=True, text=True).strip()
        assert karyotype == "XY"

    def test_interval_list_regions(self, tmp_path):
        cram, fasta = _karyotype_cram(tmp_path, 2, 0)
        interval_list = tmp_path / "regions.interval_list"
        interval_list.write_text(
            "@HD\tVN:1.6\n"
            + "".join(f"@SQ\tSN:{contig}\tLN:{SEX_TEST_CONTIG_LENGTH}\n" for contig in SEX_TEST_CONTIGS)
            + "".join(f"{contig}\t1\t{SEX_TEST_CONTIG_LENGTH}\t+\t.\n" for contig in SEX_TEST_CONTIGS)
        )
        main(self._args(cram, fasta, interval_list, tmp_path))
        assert "Karyotype:          XX" in (tmp_path / "SAMPLE.ploidy_report.txt").read_text()
        assert not (tmp_path / "SAMPLE.haploid_regions.bed").exists()

    def test_par_bed_writes_haploid_regions(self, tmp_path):
        cram, fasta = _karyotype_cram(tmp_path, 1, 1)
        par_bed = tmp_path / "par.bed"
        par_bed.write_text("chrX\t0\t1000\nchrY\t0\t1000\n")
        main([*self._args(cram, fasta, _regions_bed(tmp_path), tmp_path), "--par-bed", str(par_bed)])
        assert (tmp_path / "SAMPLE.haploid_regions.bed").read_text() == (
            f"chrX\t1000\t{SEX_TEST_CONTIG_LENGTH}\nchrY\t1000\t{SEX_TEST_CONTIG_LENGTH}\n"
        )

    def test_par_bed_requires_sex_only(self, tmp_path):
        with pytest.raises(SystemExit):
            main(["--mosdepth-summary", "summary.txt", "--sample-id", "S", "--par-bed", "par.bed"])

    @pytest.mark.parametrize(
        "drop,extra",
        [
            ("--sex-only", []),
            ("--cram", []),
            ("--reference", []),
            ("--regions-bed", []),
            (None, ["--vcf", "x.vcf.gz"]),
            (None, ["--mosdepth-summary", "summary.txt"]),
        ],
    )
    def test_invalid_combinations_exit(self, tmp_path, drop, extra):
        args = self._args("s.cram", "ref.fa", "regions.bed", tmp_path)
        if drop == "--sex-only":
            args.remove("--sex-only")
        elif drop:
            i = args.index(drop)
            del args[i : i + 2]
        with pytest.raises(SystemExit):
            main(args + extra)


class TestWriteHaploidRegionsBed:
    @staticmethod
    def _write(tmp_path, karyotype, contig_lengths=(("chr1", 2000), ("chrX", 1000), ("chrY", 500)), pars=None):
        fasta = tmp_path / "ref.fa"
        Path(f"{fasta}.fai").write_text("".join(f"{name}\t{length}\t0\t60\t61\n" for name, length in contig_lengths))
        par_bed = tmp_path / "par.bed"
        par_bed.write_text(pars if pars is not None else "chrX\t10\t100\nchrX\t900\t950\nchrY\t10\t100\n")
        return write_haploid_regions_bed("SAMPLE", karyotype, fasta, par_bed, tmp_path).read_text()

    @pytest.mark.parametrize(
        "karyotype,expected",
        [
            ("XY", "chrX\t0\t10\nchrX\t100\t900\nchrX\t950\t1000\nchrY\t0\t10\nchrY\t100\t500\n"),
            ("XYY", "chrX\t0\t10\nchrX\t100\t900\nchrX\t950\t1000\n"),
            ("XXY", "chrY\t0\t10\nchrY\t100\t500\n"),
            ("XXXY", "chrY\t0\t10\nchrY\t100\t500\n"),
            ("X0", ""),
            ("XX", ""),
            ("XXX", ""),
            ("UNDETERMINED", ""),
        ],
    )
    def test_haploid_regions_per_karyotype(self, tmp_path, karyotype, expected):
        assert self._write(tmp_path, karyotype) == expected

    def test_bare_contig_names(self, tmp_path):
        bed = self._write(tmp_path, "XXY", contig_lengths=(("Y", 500),), pars="Y\t10\t100\n")
        assert bed == "Y\t0\t10\nY\t100\t500\n"

    def test_haploid_contig_missing_from_reference_contributes_nothing(self, tmp_path):
        assert self._write(tmp_path, "XXY", contig_lengths=(("chrX", 1000),)) == ""

    def test_par_bed_header_lines_are_skipped(self, tmp_path):
        pars = "#chrom\tstart\tend\ntrack name=par\nbrowser position chrY\nchrY\t10\t100\n"
        assert self._write(tmp_path, "XXY", pars=pars) == "chrY\t0\t10\nchrY\t100\t500\n"

    def test_haploid_contig_without_par_fails(self, tmp_path):
        with pytest.raises(ValueError, match="chrY"):
            self._write(tmp_path, "XY", pars="chrX\t10\t100\n")
