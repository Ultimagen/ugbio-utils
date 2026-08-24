import pandas as pd
import pytest

from ugbio_core.estimate_ploidy import (
    _autosome_number,
    _classify_baf,
    _compute_ploidy_from_chr_data,
    _detect_chr_prefix,
    _determine_karyotype,
    _sex_label_from_karyotype,
    estimate_ploidy_from_coverage,
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


class TestComputePloidyFromChrData:
    def test_male_hg38(self):
        chr_data = {f"chr{i}": {"mean": 50.0, "length": 1e8} for i in range(1, 23)}
        chr_data["chrX"] = {"mean": 25.0, "length": 1e8}
        chr_data["chrY"] = {"mean": 25.0, "length": 5e7}
        result = _compute_ploidy_from_chr_data(chr_data, has_chr=True)
        assert result["karyotype"] == "XY"
        assert result["sex_label"] == "male"
        assert 0.4 < result["x_ratio"] < 0.6
        assert 0.4 < result["y_ratio"] < 0.6

    def test_female_hg38(self):
        chr_data = {f"chr{i}": {"mean": 50.0, "length": 1e8} for i in range(1, 23)}
        chr_data["chrX"] = {"mean": 50.0, "length": 1e8}
        chr_data["chrY"] = {"mean": 0.1, "length": 5e7}
        result = _compute_ploidy_from_chr_data(chr_data, has_chr=True)
        assert result["karyotype"] == "XX"
        assert result["sex_label"] == "female"

    def test_male_b37(self):
        chr_data = {str(i): {"mean": 40.0, "length": 1e8} for i in range(1, 23)}
        chr_data["X"] = {"mean": 20.0, "length": 1e8}
        chr_data["Y"] = {"mean": 20.0, "length": 5e7}
        result = _compute_ploidy_from_chr_data(chr_data, has_chr=False)
        assert result["karyotype"] == "XY"

    def test_no_autosomes_raises(self):
        with pytest.raises(ValueError, match="No autosomal contigs"):
            _compute_ploidy_from_chr_data({"chrX": {"mean": 25.0}}, has_chr=True)

    def test_zero_coverage_raises(self):
        chr_data = {f"chr{i}": {"mean": 0.0, "length": 1e8} for i in range(1, 23)}
        with pytest.raises(ValueError, match="Autosomal mean coverage is 0"):
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
