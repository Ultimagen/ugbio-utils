from os.path import join as pjoin
from pathlib import Path

import numpy as np
import pandas as pd
import pysam
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal
from ugbio_ppmseq.ppmSeq_utils import (
    PpmseqAdapterVersions,
    PpmseqStrandVcfAnnotator,
    add_strand_ratios_and_categories_to_featuremap,
    collect_statistics,
    plot_ppmseq_strand_ratio,
    plot_strand_ratio_category,
    plot_strand_ratio_category_concordnace,
    plot_trimmer_histogram,
    ppmseq_qc_analysis,
    read_ppmseq_trimmer_histogram,
    read_trimmer_failure_codes_ppmseq,
)

inputs_dir = Path(__file__).parent.parent / "resources"
input_histogram_legacy_v5_csv = (
    inputs_dir
    / "130713_UGAv3-51.trimming.A_hmer_5.T_hmer_5.A_hmer_3.T_hmer_3.native_adapter_with_leading_C.histogram.csv"
)
parsed_histogram_legacy_v5_parquet = inputs_dir / "130713_UGAv3-51.parsed_histogram.parquet"
sorter_stats_legacy_v5_csv = inputs_dir / "130713-UGAv3-51.sorter_stats.csv"
collected_stats_legacy_v5_h5 = inputs_dir / "130713-UGAv3-51.stats.h5"
input_histogram_legacy_v5_start_csv = inputs_dir / "130715_UGAv3-132.trimming.A_hmer.T_hmer.histogram.csv"
parsed_histogram_legacy_v5_start_parquet = inputs_dir / "130715_UGAv3-132.parsed_histogram.parquet"
sorter_stats_legacy_v5_start_csv = inputs_dir / "130715-UGAv3-51.sorter_stats.csv"
collected_stats_legacy_v5_start_h5 = inputs_dir / "130715-UGAv3-51.stats.h5"
input_featuremap_legacy_v5 = inputs_dir / "333_CRCs_39_legacy_v5.featuremap.single_substitutions.subsample.vcf.gz"
expected_output_featuremap_legacy_v5 = (
    inputs_dir / "333_CRCs_39_legacy_v5.featuremap.single_substitutions.subsample.with_strand_ratios.vcf.gz"
)
sorter_stats_csv_ppmseq_v1 = inputs_dir / "037239-CgD1502_Cord_Blood-Z0032-CTCTGTATTGCAGAT.csv"
sorter_stats_json_ppmseq_v1 = inputs_dir / "037239-CgD1502_Cord_Blood-Z0032-CTCTGTATTGCAGAT.json"
trimmer_failure_codes_csv_ppmseq_v1 = inputs_dir / "037239-CgD1502_Cord_Blood-Z0032-CTCTGTATTGCAGAT.failure_codes.csv"
trimmer_histogram_ppmseq_v1_csv = inputs_dir / (
    "037239-CgD1502_Cord_Blood-Z0032-CTCTGTATTGCAGAT."
    "Start_loop.Start_loop.End_loop.End_loop.native_adapter.histogram.csv"
)
parsed_histogram_parquet_ppmseq_v1 = (
    inputs_dir / "037239-CgD1502_Cord_Blood-Z0032-CTCTGTATTGCAGAT.parsed_histogram.parquet"
)
subdir = inputs_dir / "401057001"
sorter_stats_csv_ppmseq_v1_401057001 = subdir / "401057001-Lb_2772-Z0016-CATCCTGTGCGCATGAT.csv"
sorter_stats_json_ppmseq_v1_401057001 = subdir / "401057001-Lb_2772-Z0016-CATCCTGTGCGCATGAT.json"
trimmer_failure_codes_csv_ppmseq_v1_401057001 = (
    subdir / "401057001-Lb_2772-Z0016-CATCCTGTGCGCATGAT_trimmer-failure_codes.csv"
)
trimmer_histogram_ppmseq_v1_401057001 = subdir / (
    "Z0016-Start_loop_name.Start_loop_pattern_fw.End_loop_name.End_loop_pattern_fw.native_adapter_length"
    ".histogram.csv"
)

trimmer_failure_codes_csv_ppmseq_v1_incl_failed_rsq = (
    inputs_dir / "412884-L6860-Z0293-CATGTGAGCGGTGAT_trimmer-failure_codes.csv"
)


def _compare_vcfs(vcf_file1, vcf_file2):
    def extract_header(vcf_file):
        with pysam.VariantFile(vcf_file) as vcf:
            return vcf.header

    def extract_records(vcf_file):
        records = []
        with pysam.VariantFile(vcf_file) as vcf:
            for record in vcf:
                records.append(record)
        return records

    def compare_headers(header1, header2):
        header1_lines = str(header1).split("\n")
        header2_lines = str(header2).split("\n")

        diff = set(header1_lines).symmetric_difference(set(header2_lines))
        return diff

    def compare_records(records1, records2):
        records1_set = {str(record) for record in records1}
        records2_set = {str(record) for record in records2}

        diff = records1_set.symmetric_difference(records2_set)
        return diff

    header1 = extract_header(vcf_file1)
    header2 = extract_header(vcf_file2)

    header_diff = compare_headers(header1, header2)
    assert not header_diff, "Differences found in headers"

    records1 = extract_records(vcf_file1)
    records2 = extract_records(vcf_file2)

    records_diff = compare_records(records1, records2)
    assert not records_diff, "Differences found in records"


def test_read_trimmer_failure_codes_ppmseq(tmpdir):
    df_trimmer_failure_codes, df_metrics = read_trimmer_failure_codes_ppmseq(
        trimmer_failure_codes_csv_ppmseq_v1_incl_failed_rsq
    )
    assert (
        df_trimmer_failure_codes["total_read_count"].to_numpy()[0] == 1098118853
    )  # make sure the RSQ failed reads are subtracted
    assert np.allclose(
        df_metrics.loc["PCT_failed_adapter_dimers", "value"], 0.06888, atol=1e-3
    )  # make sure normalization is correct


def test_read_ppmseq_v1_trimmer_histogram(tmpdir):
    tmp_out_path = pjoin(tmpdir, "tmp_out.parquet")
    df_trimmer_histogram = read_ppmseq_trimmer_histogram(
        PpmseqAdapterVersions.V1,
        trimmer_histogram_ppmseq_v1_csv,
        output_filename=tmp_out_path,
        legacy_histogram_column_names=True,
    )
    df_trimmer_histogram_from_parquet = pd.read_parquet(tmp_out_path)
    df_trimmer_histogram_expected = pd.read_parquet(parsed_histogram_parquet_ppmseq_v1)
    assert_frame_equal(
        df_trimmer_histogram,
        df_trimmer_histogram_expected,
    )
    assert_frame_equal(
        df_trimmer_histogram_from_parquet,
        df_trimmer_histogram_expected,
    )


def test_read_ppmseq_legacy_v5_trimmer_histogram(tmpdir):
    tmp_out_path = pjoin(tmpdir, "tmp_out.parquet")
    df_trimmer_histogram = read_ppmseq_trimmer_histogram(
        PpmseqAdapterVersions.LEGACY_V5,
        input_histogram_legacy_v5_csv,
        output_filename=tmp_out_path,
        legacy_histogram_column_names=True,
    )
    df_trimmer_histogram_from_parquet = pd.read_parquet(tmp_out_path)
    df_trimmer_histogram_expected = pd.read_parquet(parsed_histogram_legacy_v5_parquet)
    assert_frame_equal(
        df_trimmer_histogram,
        df_trimmer_histogram_expected,
    )
    assert_frame_equal(
        df_trimmer_histogram_from_parquet,
        df_trimmer_histogram_expected,
    )


def test_read_ppmseq_legacy_v5_start_trimmer_histogram(tmpdir):
    tmp_out_path = pjoin(tmpdir, "tmp_out.parquet")

    df_trimmer_histogram = read_ppmseq_trimmer_histogram(
        PpmseqAdapterVersions.LEGACY_V5_START,
        input_histogram_legacy_v5_start_csv,
        output_filename=tmp_out_path,
        legacy_histogram_column_names=True,
    )
    df_trimmer_histogram_from_parquet = pd.read_parquet(tmp_out_path)
    df_trimmer_histogram_expected = pd.read_parquet(parsed_histogram_legacy_v5_start_parquet)
    assert_frame_equal(
        df_trimmer_histogram,
        df_trimmer_histogram_expected,
    )
    assert_frame_equal(
        df_trimmer_histogram_from_parquet,
        df_trimmer_histogram_expected,
    )


def test_plot_trimmer_histogram(tmpdir):
    tmp_out_path = pjoin(tmpdir, "tmp_out.png")
    df_trimmer_histogram = read_ppmseq_trimmer_histogram(
        PpmseqAdapterVersions.LEGACY_V5_START,
        input_histogram_legacy_v5_start_csv,
        output_filename=tmp_out_path,
        legacy_histogram_column_names=True,
    )
    plot_trimmer_histogram(
        PpmseqAdapterVersions.LEGACY_V5_START,
        df_trimmer_histogram,
        output_filename=tmp_out_path,
        legacy_histogram_column_names=True,
    )
    df_trimmer_histogram = read_ppmseq_trimmer_histogram(
        PpmseqAdapterVersions.LEGACY_V5,
        input_histogram_legacy_v5_csv,
        output_filename=tmp_out_path,
        legacy_histogram_column_names=True,
    )
    plot_trimmer_histogram(
        PpmseqAdapterVersions.LEGACY_V5,
        df_trimmer_histogram,
        output_filename=tmp_out_path,
        legacy_histogram_column_names=True,
    )


def test_collect_statistics(tmpdir):
    tmp_out_path = pjoin(tmpdir, "tmp_out.h5")
    collect_statistics(
        PpmseqAdapterVersions.LEGACY_V5,
        trimmer_histogram_csv=input_histogram_legacy_v5_csv,
        sorter_stats_csv=sorter_stats_legacy_v5_csv,
        output_filename=tmp_out_path,
        legacy_histogram_column_names=True,
    )

    f1 = collected_stats_legacy_v5_h5
    f2 = tmp_out_path
    with pd.HDFStore(f1) as fh1, pd.HDFStore(f2) as fh2:
        assert sorted(fh1.keys()) == sorted(fh2.keys())
        keys = fh1.keys()
    for k in keys:
        # below - removing one changed stat in the output shortlist to keep backward compatibility in the test
        x1 = pd.read_hdf(f1, k).drop("MIXED_read_mean_coverage", errors="ignore")
        x2 = pd.read_hdf(f2, k).drop("PCT_MIXED_start_tag", errors="ignore")
        if isinstance(x1, pd.Series):
            assert_series_equal(x1, x2)
        else:
            assert_frame_equal(x1, x2)

    collect_statistics(
        PpmseqAdapterVersions.LEGACY_V5_START,
        trimmer_histogram_csv=input_histogram_legacy_v5_start_csv,
        sorter_stats_csv=sorter_stats_legacy_v5_csv,
        output_filename=tmp_out_path,
        legacy_histogram_column_names=True,
    )
    f1 = collected_stats_legacy_v5_start_h5
    f2 = tmp_out_path
    with pd.HDFStore(f1) as fh1, pd.HDFStore(f2) as fh2:
        assert fh1.keys() == fh2.keys()
        keys = fh1.keys()
    for k in keys:
        x1 = pd.read_hdf(f1, k)
        x2 = pd.read_hdf(f2, k)
        if isinstance(x1, pd.Series):
            assert_series_equal(x1, x2)
        else:
            assert_frame_equal(x1, x2)


def test_plot_ppmseq_ratio(tmpdir):
    tmp_out_path = pjoin(tmpdir, "tmp_out.png")
    df_trimmer_histogram_legacy_v5_expected = pd.read_parquet(parsed_histogram_legacy_v5_parquet)
    plot_ppmseq_strand_ratio(
        PpmseqAdapterVersions.LEGACY_V5,
        df_trimmer_histogram_legacy_v5_expected,
        output_filename=tmp_out_path,
        title="test",
    )
    df_trimmer_histogram_legacy_v5_start_expected = pd.read_parquet(parsed_histogram_legacy_v5_start_parquet)
    plot_ppmseq_strand_ratio(
        PpmseqAdapterVersions.LEGACY_V5_START,
        df_trimmer_histogram_legacy_v5_start_expected,
        output_filename=tmp_out_path,
        title="test",
    )


def test_plot_strand_ratio_category(tmpdir):
    tmp_out_path = pjoin(tmpdir, "tmp_out.png")
    df_trimmer_histogram_legacy_v5_expected = pd.read_parquet(parsed_histogram_legacy_v5_parquet)
    plot_strand_ratio_category(
        PpmseqAdapterVersions.LEGACY_V5,
        df_trimmer_histogram_legacy_v5_expected,
        title="test",
        output_filename=tmp_out_path,
        ax=None,
    )
    df_trimmer_histogram_legacy_v5_start_expected = pd.read_parquet(parsed_histogram_legacy_v5_start_parquet)
    plot_strand_ratio_category(
        PpmseqAdapterVersions.LEGACY_V5_START,
        df_trimmer_histogram_legacy_v5_start_expected,
        title="test",
        output_filename=tmp_out_path,
        ax=None,
    )


def test_add_strand_ratios_and_categories_to_featuremap(tmpdir):
    tmp_out_path = pjoin(tmpdir, "tmp_out.vcf.gz")
    add_strand_ratios_and_categories_to_featuremap(
        PpmseqAdapterVersions.LEGACY_V5,
        input_featuremap_vcf=input_featuremap_legacy_v5,
        output_featuremap_vcf=tmp_out_path,
    )
    _compare_vcfs(
        expected_output_featuremap_legacy_v5,
        tmp_out_path,
    )


def test_plot_strand_ratio_category_concordnace(tmpdir):
    tmp_out_path = pjoin(tmpdir, "tmp_out.png")
    df_trimmer_histogram_legacy_v5_expected = pd.read_parquet(parsed_histogram_legacy_v5_parquet)
    plot_strand_ratio_category_concordnace(
        PpmseqAdapterVersions.LEGACY_V5,
        df_trimmer_histogram_legacy_v5_expected,
        title="test",
        output_filename=tmp_out_path,
        axs=None,
    )
    df_trimmer_histogram_legacy_v5_start_expected = pd.read_parquet(parsed_histogram_legacy_v5_start_parquet)
    with pytest.raises(ValueError):
        plot_strand_ratio_category_concordnace(
            PpmseqAdapterVersions.LEGACY_V5_START,
            df_trimmer_histogram_legacy_v5_start_expected,
            title="test",
            output_filename=tmp_out_path,
            axs=None,
        )


def test_ppmseq_analysis_legacy_v5(tmpdir):
    ppmseq_qc_analysis(
        PpmseqAdapterVersions.LEGACY_V5,
        trimmer_histogram_csv=[input_histogram_legacy_v5_csv],
        sorter_stats_csv=sorter_stats_legacy_v5_csv,
        output_path=tmpdir,
        output_basename="TEST_legacy_v5",
        collect_statistics_kwargs={},
        legacy_histogram_column_names=True,
    )

    ppmseq_qc_analysis(
        PpmseqAdapterVersions.LEGACY_V5_START,
        trimmer_histogram_csv=[input_histogram_legacy_v5_start_csv],
        sorter_stats_csv=sorter_stats_legacy_v5_start_csv,
        output_path=tmpdir,
        output_basename="TEST_legacy_v5_start",
        collect_statistics_kwargs={},
        legacy_histogram_column_names=True,
    )


def test_ppmseq_analysis_v1(tmpdir):
    ppmseq_qc_analysis(
        PpmseqAdapterVersions.V1,
        trimmer_histogram_csv=[trimmer_histogram_ppmseq_v1_401057001],
        sorter_stats_csv=sorter_stats_csv_ppmseq_v1_401057001,
        trimmer_failure_codes_csv=trimmer_failure_codes_csv_ppmseq_v1_401057001,
        output_path=tmpdir,
        output_basename="TEST_v1",
    )


def test_pickle_an_annotator(tmpdir):
    import pickle

    annotator = PpmseqStrandVcfAnnotator(adapter_version="legacy_v5_start")
    with open(pjoin(tmpdir, "annotators_pickle"), "wb") as f:
        pickle.dump(annotator, f)
