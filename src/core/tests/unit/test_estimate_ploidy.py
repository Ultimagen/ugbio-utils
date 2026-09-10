import random

import pysam
import pytest
from ugbio_core.estimate_ploidy import (
    _autosome_number,
    _classify_baf,
    _compute_ploidy_from_chr_data,
    _detect_chr_prefix,
    _determine_karyotype,
    _is_standard_biallelic_snp,
    _sex_label_from_karyotype,
    _update_reservoir,
    estimate_ploidy_from_coverage,
    estimate_ploidy_from_vcf,
    main,
    parse_mosdepth_summary,
)


class TestDetectChrPrefix:
    def test_hg38_contigs(self):
        assert _detect_chr_prefix(["chr1", "chr2", "chrX"]) is True

    def test_b37_contigs(self):
        assert _detect_chr_prefix(["1", "2", "X"]) is False

    def test_mixed_contigs(self):
        assert _detect_chr_prefix(["chr1", "chrUn_gl000220"]) is True


class TestAutosomeNumber:
    def test_hg38(self):
        assert _autosome_number("chr1", has_chr=True) == 1
        assert _autosome_number("chr22", has_chr=True) == 22
        assert _autosome_number("chrX", has_chr=True) is None

    def test_b37(self):
        assert _autosome_number("1", has_chr=False) == 1
        assert _autosome_number("X", has_chr=False) is None

    def test_non_contig(self):
        assert _autosome_number("chrM", has_chr=True) is None
        assert _autosome_number("GL000220.1", has_chr=False) is None


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


class TestClassifyBaf:
    def test_insufficient_data(self):
        result = _classify_baf([0.5] * 10)
        assert result["label"] == "INSUFFICIENT_DATA"

    def test_diploid(self):
        result = _classify_baf([0.5] * 100)
        assert result["label"] == "DIPLOID"

    def test_triploid_signal(self):
        baf = [0.33] * 60 + [0.67] * 40
        result = _classify_baf(baf)
        assert result["label"] in ("TRIPLOID", "LIKELY_DIPLOID", "INCONCLUSIVE")


class TestUpdateReservoir:
    def test_bounds_sample_size(self):
        reservoir = []
        seen_count = 0

        for value in range(100):
            seen_count = _update_reservoir(reservoir, value, seen_count, sample_count=10, rng=random.Random(42))

        assert seen_count == 100
        assert len(reservoir) == 10


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
        result = _compute_ploidy_from_chr_data(chr_data, has_chr=True)
        assert result["karyotype"] == "XY"
        assert result["sex_label"] == "male"
        assert 0.4 < result["x_ratio"] < 0.6
        assert 0.4 < result["y_ratio"] < 0.6

    def test_female_hg38(self):
        chr_data = {f"chr{i}": {"coverage": 50.0, "length": 1e8} for i in range(1, 23)}
        chr_data["chrX"] = {"coverage": 50.0, "length": 1e8}
        chr_data["chrY"] = {"coverage": 0.1, "length": 5e7}
        result = _compute_ploidy_from_chr_data(chr_data, has_chr=True)
        assert result["karyotype"] == "XX"
        assert result["sex_label"] == "female"

    def test_male_b37(self):
        chr_data = {str(i): {"coverage": 40.0, "length": 1e8} for i in range(1, 23)}
        chr_data["X"] = {"coverage": 20.0, "length": 1e8}
        chr_data["Y"] = {"coverage": 20.0, "length": 5e7}
        result = _compute_ploidy_from_chr_data(chr_data, has_chr=False)
        assert result["karyotype"] == "XY"

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

    def test_autosomal_baseline_uses_length_weighted_coverage(self):
        chr_data = {
            "chr1": {"coverage": 40.0, "length": 200},
            "chr2": {"coverage": 80.0, "length": 100},
            "chrX": {"coverage": 20.0, "length": 100},
            "chrY": {"coverage": 20.0, "length": 50},
        }
        result = _compute_ploidy_from_chr_data(chr_data, sex_chromosomes=("chrX", "chrY"))
        assert result["auto_mean"] == 53.33
        assert result["warnings"] == []

    def test_autosomal_baseline_falls_back_to_unweighted_median_without_lengths(self):
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
        assert "using unweighted median" in result["warnings"][0]

    def test_no_autosomes_raises(self):
        with pytest.raises(ValueError, match="No autosomal contigs"):
            _compute_ploidy_from_chr_data({"chrX": {"coverage": 25.0}}, has_chr=True)

    def test_zero_coverage_raises(self):
        chr_data = {f"chr{i}": {"coverage": 0.0, "length": 1e8} for i in range(1, 23)}
        with pytest.raises(ValueError, match="Autosomal median coverage is 0"):
            _compute_ploidy_from_chr_data(chr_data, has_chr=True)


class TestEstimatePloidyFromCoverage:
    def test_mosdepth_male(self, tmp_path):
        tsv = tmp_path / "summary.txt"
        lines = ["chrom\tlength\tbases\tmean\tmin_cov\tmax_cov\n"]
        for i in range(1, 23):
            lines.append(f"chr{i}\t100000000\t5000000000\t50.0\t0\t200\n")
        lines.append("chrX\t100000000\t2500000000\t25.0\t0\t100\n")
        lines.append("chrY\t50000000\t1250000000\t25.0\t0\t100\n")
        lines.append("total\t3000000000\t150000000000\t50.0\t0\t200\n")
        tsv.write_text("".join(lines))

        mosdepth_df = parse_mosdepth_summary(tsv)
        result = estimate_ploidy_from_coverage(mosdepth_df)
        assert result["karyotype"] == "XY"
        assert result["sex_label"] == "male"
        assert result["source"] == "mosdepth"


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

    def test_reads_unfiltered_snps_with_pysam(self, tmp_path):
        vcf_path = self._make_ploidy_vcf(tmp_path)

        coverage_result, baf_result = estimate_ploidy_from_vcf(vcf_path, "SAMPLE")

        assert coverage_result["karyotype"] == "XY"
        assert coverage_result["source"] == "VCF SNP median DP"
        assert baf_result["label"] == "INSUFFICIENT_DATA"

    def test_estimate_ploidy_from_vcf_does_not_mutate_global_random_state(self, tmp_path):
        vcf_path = self._make_ploidy_vcf(tmp_path)

        random.seed(123)
        expected = random.random()
        random.seed(123)
        estimate_ploidy_from_vcf(vcf_path, "SAMPLE")

        assert random.random() == expected

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


class TestEstimatePloidyCli:
    def test_vcf_mode_writes_report(self, tmp_path):
        vcf_path = TestEstimatePloidyFromVcf._make_ploidy_vcf(tmp_path)

        main(["--vcf", str(vcf_path), "--sample-id", "SAMPLE", "--output-dir", str(tmp_path)])

        report = tmp_path / "SAMPLE.ploidy_report.txt"
        report_text = report.read_text()
        assert report.exists()
        assert "Karyotype:" in report_text
        assert "Coverage source:    VCF SNP median DP" in report_text

    def test_mosdepth_summary_mode_writes_report(self, tmp_path):
        summary_path = tmp_path / "summary.txt"
        lines = ["chrom\tlength\tbases\tmean\tmin_cov\tmax_cov\n"]
        for i in range(1, 23):
            lines.append(f"chr{i}\t100000000\t5000000000\t50.0\t0\t200\n")
        lines.append("chrX\t100000000\t2500000000\t25.0\t0\t100\n")
        lines.append("chrY\t50000000\t1250000000\t25.0\t0\t100\n")
        summary_path.write_text("".join(lines))

        main(["--mosdepth-summary", str(summary_path), "--sample-id", "SAMPLE", "--output-dir", str(tmp_path)])

        report = tmp_path / "SAMPLE.ploidy_report.txt"
        report_text = report.read_text()
        assert report.exists()
        assert "Karyotype:          XY" in report_text
        assert "Coverage source:    mosdepth" in report_text

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
