import pickle
from os.path import join as pjoin
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pysam
import pytest
from ugbio_ppmseq.ppmSeq_utils import (
    ET_TAG,
    SORTER_STATS_KEYS_TO_SHOW,
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
    has_sr_tag,
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
subsampled_sam_no_sr = inputs_dir / "ppmseq_sr_tag" / "Z0263_sample_no_sr.sam.gz"
input_featuremap_legacy_v5 = inputs_dir / "333_CRCs_39_legacy_v5.featuremap.single_substitutions.subsample.vcf.gz"
expected_output_featuremap_legacy_v5 = (
    inputs_dir / "333_CRCs_39_legacy_v5.featuremap.single_substitutions.subsample.with_strand_ratios.vcf.gz"
)
trimmer_failure_codes_csv_ppmseq_v1_incl_failed_rsq = (
    inputs_dir / "412884-L6860-Z0293-CATGTGAGCGGTGAT_trimmer-failure_codes.csv"
)
sorter_stats_csv = inputs_dir / "ppmseq_sr_tag" / "Z0263_sorter_stats.csv"


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


def test_read_tags_from_subsampled_sam_no_sr():
    """Fixture identical to Z0263_sample.sam.gz but with every sr tag stripped — the reader
    must still succeed, leave sr as NaN, and has_sr_tag() must return False."""
    df_reads = read_tags_from_subsampled_sam(str(subsampled_sam_no_sr))
    assert len(df_reads) > 0
    assert df_reads[SR_TAG].isna().all()
    assert not has_sr_tag(df_reads)
    # The non-sr tags should still be populated exactly like the source fixture.
    assert df_reads[ST_TAG].isin(["PLUS", "MINUS", "MIXED", "UNDETERMINED"]).all()


def test_has_sr_tag_true_on_full_fixture():
    df_reads = read_tags_from_subsampled_sam(str(subsampled_sam))
    assert has_sr_tag(df_reads)


def test_read_tags_drops_unmatched_reads(tmp_path):
    """Reads with RG=unmatched (Trimmer failures routed to the unmatched read group) must
    be filtered out before any QC is computed. They never received ppmSeq tag calls so
    leaving them in would either raise KeyError on st or skew Section 1 denominators."""
    sam_content = (
        "@HD\tVN:1.6\n"
        "@SQ\tSN:chr1\tLN:100\n"
        "@RG\tID:Z0263\n"
        "@RG\tID:unmatched\n"
        # Matched read with full tags.
        "r1\t4\t*\t0\t0\t*\t*\t0\t0\tAAAA\t!!!!\tRG:Z:Z0263\tsr:f:0.5\tst:Z:MIXED\n"
        # Unmatched read — no ppmSeq tags. Must be skipped, not raise KeyError on st.
        "r2\t4\t*\t0\t0\t*\t*\t0\t0\tAAAA\t!!!!\tRG:Z:unmatched\n"
    )
    sam_file = tmp_path / "mixed.sam"
    sam_file.write_text(sam_content)
    df_reads = read_tags_from_subsampled_sam(str(sam_file))
    assert len(df_reads) == 1
    assert df_reads.iloc[0][ST_TAG] == "MIXED"


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


def _category_bar_xticks(ax):
    """Return the x-axis tick labels actually rendered on a seaborn barplot."""
    return [t.get_text() for t in ax.get_xticklabels()]


def test_plot_strand_ratio_category_sr_present_drops_start_undetermined(tmp_path):
    """When sr is known to be present on every read, the sr cascade only emits
    MIXED/PLUS/MINUS on the start axis, so the start bars for UNDETERMINED should
    be suppressed. We force a row into the fixture where st=UNDETERMINED and
    verify the `sr_present=True` plot drops it."""
    df_reads = read_tags_from_subsampled_sam(str(subsampled_sam))
    # Inject a visible UNDETERMINED start row so the suppression is observable.
    df_reads.loc[df_reads.index[0], "strand_ratio_category_start"] = PpmseqCategories.UNDETERMINED.value

    fig, (ax_off, ax_on) = plt.subplots(2, 1, figsize=(12, 6))
    plot_strand_ratio_category(PpmseqAdapterVersions.V1, df_reads, ax=ax_off, sr_present=False)
    plot_strand_ratio_category(PpmseqAdapterVersions.V1, df_reads, ax=ax_on, sr_present=True)
    fig.savefig(tmp_path / "cat_sr.png")

    # sr_present=False: UNDETERMINED still appears somewhere in the rendered bars.
    # sr_present=True: the Start-tag series for UNDETERMINED is masked, so seaborn only
    # draws the end-tag bar on the UNDETERMINED slot. We inspect each hue container directly.
    def _find_container(ax, hue_substring):
        legend = ax.get_legend()
        texts = [t.get_text() for t in legend.get_texts()] if legend is not None else []
        for cont, lbl in zip(ax.containers, texts, strict=False):
            if hue_substring.lower() in lbl.lower():
                return cont
        return None

    start_off = _find_container(ax_off, "Start")
    start_on = _find_container(ax_on, "Start")
    assert start_off is not None and start_on is not None
    # When sr is not claimed to be present, UNDETERMINED on start should be one bar
    # (possibly small but > 0 since we injected one row above).
    assert len(start_off) > len(start_on), (
        f"sr_present=True should drop at least one start-tag bar " f"(got {len(start_off)} off vs {len(start_on)} on)"
    )


def test_plot_strand_ratio_category_concordance_sr_present_drops_start_undetermined(tmp_path):
    """Same idea for the concordance heatmap: the start-axis (row index) UNDETERMINED
    row is dropped when sr_present=True, while the end-axis (column index) keeps it."""
    df_reads = read_tags_from_subsampled_sam(str(subsampled_sam))
    # Make sure at least one read has st=UNDETERMINED so the base plot actually has a row
    # to drop.
    df_reads.loc[df_reads.index[0], "strand_ratio_category_start"] = PpmseqCategories.UNDETERMINED.value

    fig, (ax_off_a, ax_off_b, ax_on_a, ax_on_b) = plt.subplots(4, 1, figsize=(10, 20))
    plot_strand_ratio_category_concordnace(
        PpmseqAdapterVersions.V1, df_reads, axs=[ax_off_a, ax_off_b], sr_present=False
    )
    plot_strand_ratio_category_concordnace(PpmseqAdapterVersions.V1, df_reads, axs=[ax_on_a, ax_on_b], sr_present=True)
    fig.savefig(tmp_path / "concord_sr.png")
    # Each heatmap axes has y-tick labels that are the row categories.
    off_rows = [t.get_text() for t in ax_off_a.get_yticklabels()]
    on_rows = [t.get_text() for t in ax_on_a.get_yticklabels()]
    assert PpmseqCategories.UNDETERMINED.value in off_rows
    assert PpmseqCategories.UNDETERMINED.value not in on_rows
    # Columns (end-tag axis) still include UNDETERMINED regardless.
    on_cols = [t.get_text() for t in ax_on_a.get_xticklabels()]
    assert PpmseqCategories.UNDETERMINED.value in on_cols


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
    /stats_shortlist — the report shows them in a dedicated section. Only the allow-listed
    keys from SORTER_STATS_KEYS_TO_SHOW should appear, in that order."""
    out = tmp_path / "stats_with_sorter.h5"
    collect_statistics(
        PpmseqAdapterVersions.V1,
        subsampled_sam=str(subsampled_sam),
        sorter_stats_csv=str(sorter_stats_csv),
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
    # All rows in sorter_stats are whitelisted keys; any row that isn't in the allow-list
    # must have been dropped.
    assert set(sorter_stats.index).issubset(set(SORTER_STATS_KEYS_TO_SHOW))
    # Rows appear in the order defined in SORTER_STATS_KEYS_TO_SHOW (filtered to those
    # present in the CSV).
    expected_order = [k for k in SORTER_STATS_KEYS_TO_SHOW if k in sorter_stats.index]
    assert list(sorter_stats.index) == expected_order


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
