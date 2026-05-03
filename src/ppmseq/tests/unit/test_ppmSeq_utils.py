import pickle
from os.path import join as pjoin
from pathlib import Path

import numpy as np
import pandas as pd
import pysam
import pytest
from ugbio_ppmseq.ppmSeq_utils import (
    ET_TAG,
    SR_TAG,
    ST_TAG,
    TM_TAG,
    PpmseqAdapterVersions,
    PpmseqCategories,
    PpmseqStrandVcfAnnotator,
    add_strand_ratios_and_categories_to_featuremap,
    collect_statistics,
    get_strand_ratio_category_concordance,
    group_trimmer_histogram_by_strand_ratio_category,
    plot_read_length_by_st,
    plot_read_length_overall,
    plot_sr_by_et,
    plot_sr_histogram,
    plot_strand_ratio_category,
    plot_strand_ratio_category_concordnace,
    ppmseq_qc_analysis,
    read_tags_from_subsampled_sam,
    read_trimmer_failure_codes_ppmseq,
)

inputs_dir = Path(__file__).parent.parent / "resources"
subsampled_sam = inputs_dir / "ppmseq_sr_tag" / "Z0263_sample.sam.gz"
input_featuremap_legacy_v5 = inputs_dir / "333_CRCs_39_legacy_v5.featuremap.single_substitutions.subsample.vcf.gz"
expected_output_featuremap_legacy_v5 = (
    inputs_dir / "333_CRCs_39_legacy_v5.featuremap.single_substitutions.subsample.with_strand_ratios.vcf.gz"
)
trimmer_failure_codes_csv_ppmseq_v1_incl_failed_rsq = (
    inputs_dir / "412884-L6860-Z0293-CATGTGAGCGGTGAT_trimmer-failure_codes.csv"
)
sorter_stats_csv_legacy_v5 = inputs_dir / "130713-UGAv3-51.sorter_stats.csv"


def _compare_vcfs(vcf_file1, vcf_file2):
    def extract_header(vcf_file):
        with pysam.VariantFile(vcf_file) as vcf:
            return vcf.header

    def extract_records(vcf_file):
        with pysam.VariantFile(vcf_file) as vcf:
            return list(vcf)

    def compare_headers(h1, h2):
        return set(str(h1).split("\n")).symmetric_difference(set(str(h2).split("\n")))

    def compare_records(r1, r2):
        return {str(r) for r in r1}.symmetric_difference({str(r) for r in r2})

    assert not compare_headers(extract_header(vcf_file1), extract_header(vcf_file2))
    assert not compare_records(extract_records(vcf_file1), extract_records(vcf_file2))


def test_read_trimmer_failure_codes_ppmseq():
    df_trimmer_failure_codes, df_metrics = read_trimmer_failure_codes_ppmseq(
        trimmer_failure_codes_csv_ppmseq_v1_incl_failed_rsq
    )
    # sanity: RSQ-failed reads are excluded from the read total before normalization.
    assert df_trimmer_failure_codes["total_read_count"].to_numpy()[0] == 1098118853
    # Regression: normalization of the per-category PCT columns (adapter-dimer fraction
    # is a critical QC metric per the review).
    assert np.isclose(df_metrics.loc["PCT_failed_adapter_dimers", "value"], 0.068886, atol=1e-4)


def test_read_tags_from_subsampled_sam():
    df_reads = read_tags_from_subsampled_sam(str(subsampled_sam))
    assert len(df_reads) > 0
    assert set(df_reads.columns) >= {
        SR_TAG,
        ST_TAG,
        ET_TAG,
        TM_TAG,
        "count",
        "strand_ratio_category_start",
        "strand_ratio_category_end",
    }
    # Every ppmSeq read produced by the current pipeline carries an st tag; the reader
    # raises KeyError on anything that doesn't, so here we only see valid categories.
    assert df_reads[ST_TAG].isin(["PLUS", "MINUS", "MIXED", "UNDETERMINED"]).all()
    # sr is nominally in [0, 1]; calibration can produce a small out-of-range tail.
    assert df_reads[SR_TAG].between(-1, 2).all()
    assert df_reads[SR_TAG].between(0, 1).mean() > 0.99


def test_read_tags_raises_on_missing_st_tag(tmp_path):
    # Build a 2-record SAM where one record is missing both sr and st.
    sam_content = (
        "@HD\tVN:1.6\n"
        "@SQ\tSN:chr1\tLN:100\n"
        "@RG\tID:test\n"
        # good read: has sr, st
        "r1\t4\t*\t0\t0\t*\t*\t0\t0\tAAAA\t!!!!\tRG:Z:test\tsr:f:0.5\tst:Z:MIXED\n"
        # bad read: no sr, no st
        "r2\t4\t*\t0\t0\t*\t*\t0\t0\tAAAA\t!!!!\tRG:Z:test\n"
    )
    sam_file = tmp_path / "bad.sam"
    sam_file.write_text(sam_content)
    with pytest.raises(KeyError):
        read_tags_from_subsampled_sam(str(sam_file))


def test_group_by_strand_ratio_category():
    df_reads = read_tags_from_subsampled_sam(str(subsampled_sam))
    grouped = group_trimmer_histogram_by_strand_ratio_category(PpmseqAdapterVersions.V1, df_reads)
    assert "strand_ratio_category_start" in grouped.columns
    assert "strand_ratio_category_end" in grouped.columns
    # Counts across category columns equal the total number of reads.
    assert int(grouped["strand_ratio_category_start"].sum()) == len(df_reads)
    # PLUS/MINUS/MIXED should each contain a non-trivial share of reads.
    for cat in [PpmseqCategories.PLUS.value, PpmseqCategories.MINUS.value, PpmseqCategories.MIXED.value]:
        assert grouped.loc[cat, "strand_ratio_category_start"] > 0


def test_get_concordance_sums_to_one():
    df_reads = read_tags_from_subsampled_sam(str(subsampled_sam))
    concordance, _concordance_no_unreached, consensus = get_strand_ratio_category_concordance(
        PpmseqAdapterVersions.V1, df_reads
    )
    # Both concordance and consensus series normalize to 1 across all cells.
    assert concordance.sum() == pytest.approx(1.0, abs=1e-6)
    assert consensus.sum() == pytest.approx(1.0, abs=1e-6)
    assert set(consensus.index) >= {"MIXED", "MINUS", "PLUS"}


def test_plot_sr_histogram(tmp_path):
    df_reads = read_tags_from_subsampled_sam(str(subsampled_sam))
    plot_sr_histogram(df_reads, title="test", output_filename=str(tmp_path / "sr.png"))
    assert (tmp_path / "sr.png").exists()


def test_plot_sr_by_et(tmp_path):
    df_reads = read_tags_from_subsampled_sam(str(subsampled_sam))
    plot_sr_by_et(df_reads, title="test", output_filename=str(tmp_path / "sr_by_et.png"))
    assert (tmp_path / "sr_by_et.png").exists()


def test_plot_strand_ratio_category(tmp_path):
    df_reads = read_tags_from_subsampled_sam(str(subsampled_sam))
    plot_strand_ratio_category(
        PpmseqAdapterVersions.V1,
        df_reads,
        title="test",
        output_filename=str(tmp_path / "cat.png"),
    )
    assert (tmp_path / "cat.png").exists()


def test_plot_read_length_overall(tmp_path):
    df_reads = read_tags_from_subsampled_sam(str(subsampled_sam))
    plot_read_length_overall(df_reads, title="test", output_filename=str(tmp_path / "rl.png"))
    assert (tmp_path / "rl.png").exists()


def test_plot_read_length_by_st(tmp_path):
    df_reads = read_tags_from_subsampled_sam(str(subsampled_sam))
    plot_read_length_by_st(df_reads, title="test", output_filename=str(tmp_path / "rl_by_st.png"))
    assert (tmp_path / "rl_by_st.png").exists()


def test_plot_strand_ratio_category_concordance(tmp_path):
    df_reads = read_tags_from_subsampled_sam(str(subsampled_sam))
    plot_strand_ratio_category_concordnace(
        PpmseqAdapterVersions.V1,
        df_reads,
        title="test",
        output_filename=str(tmp_path / "concord.png"),
    )
    assert (tmp_path / "concord.png").exists()


def test_collect_statistics(tmp_path):
    out = tmp_path / "stats.h5"
    collect_statistics(
        PpmseqAdapterVersions.V1,
        subsampled_sam=str(subsampled_sam),
        output_filename=str(out),
    )
    with pd.HDFStore(str(out)) as store:
        keys = set(store.keys())
        shortlist = store["stats_shortlist"]
        concordance = store["strand_ratio_category_concordance"]
    assert "/subsampled_reads" in keys
    assert "/strand_ratio_category_counts" in keys
    assert "/strand_ratio_category_concordance" in keys
    # The consensus percentages are part of the shortlist and the per-strand-ratio
    # category totals should match the number of reads we loaded.
    assert "PCT_MIXED_both_tags" in shortlist.index
    assert concordance.sum() == pytest.approx(1.0, abs=1e-6)


def test_collect_statistics_with_sorter_stats(tmp_path):
    """Sorter stats must be stored under /sorter_stats but must NOT be concat'd into
    /stats_shortlist — the report shows them in a dedicated section."""
    out = tmp_path / "stats_with_sorter.h5"
    collect_statistics(
        PpmseqAdapterVersions.V1,
        subsampled_sam=str(subsampled_sam),
        sorter_stats_csv=str(sorter_stats_csv_legacy_v5),
        output_filename=str(out),
    )
    with pd.HDFStore(str(out)) as store:
        sorter_stats = store["sorter_stats"]
        shortlist = store["stats_shortlist"]
    assert len(sorter_stats) > 0
    assert "PCT_MIXED_both_tags" in shortlist.index
    # Sorter metric names should NOT leak into the mixed-reads shortlist.
    shortlist_indices = set(shortlist.index)
    for sorter_metric in sorter_stats.index:
        assert (
            sorter_metric not in shortlist_indices
        ), f"sorter metric {sorter_metric!r} should not appear in stats_shortlist"


def test_ppmseq_qc_analysis(tmp_path):
    ppmseq_qc_analysis(
        PpmseqAdapterVersions.V1,
        subsampled_sam=str(subsampled_sam),
        output_path=str(tmp_path),
        output_basename="TEST_v1",
        generate_report=False,
    )
    assert (tmp_path / "TEST_v1.ppmSeq.applicationQC.h5").exists()


def test_add_strand_ratios_and_categories_to_featuremap(tmp_path):
    tmp_out_path = pjoin(tmp_path, "tmp_out.vcf.gz")
    add_strand_ratios_and_categories_to_featuremap(
        PpmseqAdapterVersions.LEGACY_V5,
        input_featuremap_vcf=input_featuremap_legacy_v5,
        output_featuremap_vcf=tmp_out_path,
    )
    _compare_vcfs(expected_output_featuremap_legacy_v5, tmp_out_path)


def test_pickle_an_annotator(tmp_path):
    annotator = PpmseqStrandVcfAnnotator(adapter_version="legacy_v5")
    with open(pjoin(tmp_path, "annotators_pickle"), "wb") as f:
        pickle.dump(annotator, f)
