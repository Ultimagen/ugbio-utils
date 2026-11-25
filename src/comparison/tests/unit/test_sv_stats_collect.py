from io import StringIO

import dill as pickle
import pandas as pd
import pytest
from ugbio_comparison import sv_stats_collect
from ugbio_comparison.sv_stats_collect import (
    collect_size_type_histograms,
    collect_sv_stats,
    concordance_with_gt,
    concordance_with_gt_roc,
)

VCF_COLUMNS = ["chrom", "pos", "id", "ref", "alt", "qual", "filter", "info"]
MOCK_VCF_DATA = """
    #CHROM  POS     ID      REF     ALT     QUAL    FILTER  INFO
    1       1000    .       A       <DEL>   .       PASS    SVLEN=-500;SVTYPE=DEL
    1       2000    .       A       <INS>   .       PASS    SVLEN=300;SVTYPE=INS
    1       3000    .       A       <DEL>   .       PASS    SVLEN=-1000;SVTYPE=DEL
    1       4000    .       A       <INS>   .       PASS    SVLEN=700;SVTYPE=INS
    1       5000    .       A       <DUP>   .       .       SVLEN=2000;SVTYPE=DUP
    """


def read_mock_vcf(vcf_data: str) -> pd.DataFrame:
    vcf_df = pd.read_csv(StringIO(vcf_data), sep=r"\s+", comment="#", names=VCF_COLUMNS)
    vcf_df["svlen"] = vcf_df["info"].str.extract(r"SVLEN=(-?\d+)").astype(float)
    vcf_df["svtype"] = vcf_df["info"].str.extract(r"SVTYPE=([A-Z]+)")
    return vcf_df


def mock_precision_recall_curve(gt, predictions, fn_mask, pos_label, min_class_counts_to_output):
    # precisions, recalls, f1 scores, thresholds
    return [0.8, 0.9], [0.7, 0.8], [0.75, 0.85], [0.5, 0.6]


@pytest.fixture
def mock_vcf(monkeypatch):
    def _mock(vcf_data: str = MOCK_VCF_DATA):
        monkeypatch.setattr(sv_stats_collect.vcftools, "get_vcf_df", lambda *args, **kwargs: read_mock_vcf(vcf_data))

    return _mock


def test_collect_size_type_histograms(mock_vcf):
    mock_vcf()
    result = collect_size_type_histograms("mock_path")

    pd.testing.assert_series_equal(
        result["type_counts"], pd.Series({"DEL": 2, "INS": 2, "DUP": 1}), check_dtype=False, check_names=False
    )
    expected_length_counts = [0, 0, 1, 2, 1, 1, 0, 0, 0, 0, 0]
    assert result["length_counts"].tolist() == expected_length_counts
    assert result["length_counts"].index.tolist() == sv_stats_collect.SVLABELS

    expected_length_by_type_counts = pd.DataFrame(0, index=["DEL", "DUP", "INS"], columns=sv_stats_collect.SVLABELS)
    expected_length_by_type_counts.loc["INS", "300-500"] = 1
    expected_length_by_type_counts.loc["DEL", "0.5-1k"] = 1
    expected_length_by_type_counts.loc["INS", "0.5-1k"] = 1
    expected_length_by_type_counts.loc["DEL", "1k-2k"] = 1
    expected_length_by_type_counts.loc["DUP", "2k-3k"] = 1
    pd.testing.assert_frame_equal(
        result["length_by_type_counts"], expected_length_by_type_counts, check_dtype=False, check_names=False
    )


def test_collect_size_type_histograms_filters_non_pass(mock_vcf):
    mock_vcf(MOCK_VCF_DATA.replace("PASS    SVLEN=300", "LowQual SVLEN=300"))
    result = collect_size_type_histograms("mock_path")
    assert result["type_counts"]["INS"] == 1
    result = collect_size_type_histograms("mock_path", ignore_filter=True)
    assert result["type_counts"]["INS"] == 2


def test_collect_size_type_histograms_empty_vcf(monkeypatch):
    empty_vcf = pd.DataFrame(columns=[*VCF_COLUMNS, "svlen", "svtype"])
    monkeypatch.setattr(sv_stats_collect.vcftools, "get_vcf_df", lambda *args, **kwargs: empty_vcf)
    result = collect_size_type_histograms("mock_path")

    assert result["type_counts"].empty
    assert (result["length_counts"] == 0).all()
    assert (result["length_by_type_counts"] == 0).all().all()


def test_concordance_with_gt():
    df_base = pd.DataFrame({"label": ["TP", "TP", "FN", "FN"]})
    df_calls = pd.DataFrame({"label": ["TP", "TP", "FP", "FP"]})
    result = concordance_with_gt(df_base, df_calls)

    assert result["TP_base"] == 2
    assert result["TP_calls"] == 2
    assert result["FN"] == 2
    assert result["FP"] == 2
    assert result["Precision"] == 0.5
    assert result["Recall"] == 0.5
    assert result["F1"] == 0.5


def test_concordance_with_gt_roc(monkeypatch):
    df_base = pd.DataFrame({"label": ["TP", "FN", "FN"], "qual": [0.9, 0.8, 0.7]})
    df_calls = pd.DataFrame({"label": ["TP", "FP", "FP"], "qual": [0.95, 0.6, 0.5]})
    monkeypatch.setattr(sv_stats_collect.stats_utils, "precision_recall_curve", mock_precision_recall_curve)

    result = concordance_with_gt_roc(df_base, df_calls)
    assert result["precision"] == [0.8, 0.9]
    assert result["recall"] == [0.7, 0.8]
    assert result["thresholds"] == [0.5, 0.6]


def test_collect_sv_stats_without_concordance(mock_vcf):
    mock_vcf()
    sv_stats, concordance_stats, fp_stats = collect_sv_stats("mock_path")

    assert set(sv_stats) == {"type_counts", "length_counts", "length_by_type_counts"}
    assert concordance_stats == {}
    assert isinstance(fp_stats, pd.Series)
    assert fp_stats.empty


CONCORDANCE_VCF = """
    #CHROM  POS     ID      REF     ALT     QUAL    FILTER  INFO
    1       1000    .       A       <DEL>   10       PASS    SVLEN=-500;SVTYPE=DEL
    1       2000    .       A       <INS>   11       PASS    SVLEN=300;SVTYPE=INS
    """


@pytest.fixture
def mock_concordance(mock_vcf, monkeypatch):
    mock_vcf(CONCORDANCE_VCF)
    tables = {
        "base": pd.DataFrame(
            {"label": ["TP", "FN"], "svlen_int": [-500, 300], "svtype": ["DEL", "INS"], "qual": [10, 11]}
        ),
        "calls": pd.DataFrame(
            {"label": ["TP", "FP"], "svlen_int": [-500, 300], "svtype": ["DEL", "INS"], "qual": [10, 11]}
        ),
    }
    monkeypatch.setattr(sv_stats_collect.pd, "read_hdf", lambda filepath, key: tables[key].copy())
    monkeypatch.setattr(sv_stats_collect.stats_utils, "precision_recall_curve", mock_precision_recall_curve)


@pytest.mark.usefixtures("mock_concordance")
def test_collect_sv_stats_with_concordance():
    sv_stats, concordance_stats, fp_stats = collect_sv_stats("mock_path", "mock_concordance.h5")

    assert set(sv_stats) == {"type_counts", "length_counts", "length_by_type_counts"}
    assert concordance_stats["ALL_concordance"]["TP_base"] == 1
    assert concordance_stats["ALL_concordance"]["TP_calls"] == 1
    assert concordance_stats["ALL_concordance"]["FN"] == 1
    assert concordance_stats["ALL_concordance"]["FP"] == 1
    assert concordance_stats["ALL_roc"]["precision"] == [0.8, 0.9]
    assert concordance_stats["ALL_roc"]["recall"] == [0.7, 0.8]
    assert concordance_stats["ALL_roc"]["thresholds"] == [0.5, 0.6]
    assert isinstance(fp_stats, pd.Series)
    assert fp_stats.to_numpy()[0] == 1


@pytest.mark.usefixtures("mock_concordance")
def test_run_writes_pickle(tmp_path):
    output_file = tmp_path / "sv_stats.pkl"
    sv_stats_collect.run(["collect_sv_stats", "mock.vcf", str(output_file), "--concordance_h5", "mock.h5"])
    with open(output_file, "rb") as stats_file:
        results = pickle.load(stats_file)

    assert set(results) == {"concordance", "fp_stats", "type_counts", "length_counts", "length_by_type_counts"}
    concordance = results["concordance"]
    assert concordance.index.names == ["SV type", "SV length"]
    assert ("ALL", "") in concordance.index
    assert ("DEL", "300-500") in concordance.index
    assert concordance.loc[("ALL", ""), "TP_base"] == 1
    assert concordance.loc[("ALL", ""), "precision roc"] == [0.8, 0.9]


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("ALL_concordance", ("ALL", "")),
        ("DEL_roc", ("DEL", "")),
        ("INS_0.5-1k_concordance", ("INS", "0.5-1k")),
        ("ALL_>1M_concordance", ("ALL", ">1M")),
    ],
)
def test_split_stat_name(name, expected):
    assert sv_stats_collect._split_stat_name(name) == expected
