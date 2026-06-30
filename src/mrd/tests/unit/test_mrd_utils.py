import json
from os.path import join as pjoin
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from ugbio_mrd.mrd_utils import (
    calc_tumor_fraction_denominator_ratio,
    generate_synthetic_signatures,
    read_intersection_dataframes,
    read_signature,
)

intersection_file_basename = "MRD_test_subsample.MRD_test_subsample_annotated_AF_vcf_gz_mrd_quality_snvs.intersection"


@pytest.fixture
def resources_dir():
    return Path(__file__).parent.parent / "resources"


def _assert_read_signature(signature, expected_signature, expected_columns=None, possibly_null_columns=None):
    expected_columns = expected_columns or [
        "ref",
        "alt",
        "id",
        "qual",
        "af",
    ]
    possibly_null_columns = possibly_null_columns or [
        "id",
        "qual",
    ]
    for c in expected_columns:
        assert c in signature.columns
        if c not in possibly_null_columns:
            assert not signature[c].isna().all()
            assert (signature[c] == expected_signature[c]).all() or np.allclose(signature[c], expected_signature[c])


def test_read_signature_ug_mutect(tmpdir, resources_dir):
    signature = read_signature(pjoin(resources_dir, "mutect_mrd_signature_test.vcf.gz"), return_dataframes=True)
    signature_no_sample_name = read_signature(
        pjoin(resources_dir, "mutect_mrd_signature_test.no_sample_name.vcf.gz"),
        return_dataframes=True,
        tumor_sample="_10_FFPE",
    )  # make sure we can read the dataframe even if the sample name could not be deduced from the header
    expected_signature = pd.read_hdf(pjoin(resources_dir, "mutect_mrd_signature_test.expected_output.h5"))
    _assert_read_signature(
        signature,
        expected_signature,
        expected_columns=[
            "ref",
            "alt",
            "id",
            "qual",
            "af",
            "depth_tumor_sample",
            "cycle_skip_status",
            "gc_content",
            "left_motif",
            "right_motif",
            "mutation_type",
        ],
    )
    _assert_read_signature(
        signature_no_sample_name,
        expected_signature,
        expected_columns=[
            "ref",
            "alt",
            "id",
            "qual",
            "af",
            "depth_tumor_sample",
            "cycle_skip_status",
            "gc_content",
            "left_motif",
            "right_motif",
            "mutation_type",
        ],
        possibly_null_columns=["id", "qual", "depth_tumor_sample", "af"],
    )


def test_read_signature_ug_dv(tmpdir, resources_dir):
    signature = read_signature(pjoin(resources_dir, "dv_mrd_signature_test.vcf.gz"), return_dataframes=True)
    expected_signature = pd.read_hdf(pjoin(resources_dir, "dv_mrd_signature_test.expected_output.h5"))
    _assert_read_signature(
        signature,
        expected_signature,
        expected_columns=[
            "ref",
            "alt",
            "id",
            "qual",
            "af",
            "depth_tumor_sample",
            "cycle_skip_status",
            "gc_content",
            "left_motif",
            "right_motif",
            "mutation_type",
        ],
    )


def test_read_signature_external(resources_dir):
    signature = read_signature(pjoin(resources_dir, "external_somatic_signature.vcf.gz"), return_dataframes=True)
    expected_signature = pd.read_hdf(pjoin(resources_dir, "external_somatic_signature.expected_output.h5"))

    _assert_read_signature(signature, expected_signature)


def test_read_signature_pileup_featuremap(resources_dir):
    signature = read_signature(
        pjoin(resources_dir, "featuremap_pileup_mrd_signature_test.vcf.gz"), return_dataframes=True
    )
    expected_signature = pd.read_hdf(pjoin(resources_dir, "featuremap_pileup_mrd_signature_test.expected_output.h5"))

    _assert_read_signature(signature, expected_signature)


def test_read_intersection_dataframes(tmpdir, resources_dir):
    parsed_intersection_dataframe = read_intersection_dataframes(
        pjoin(resources_dir, f"{intersection_file_basename}.expected_output.parquet"),
        return_dataframes=True,
    )
    parsed_intersection_dataframe_expected = pd.read_parquet(
        pjoin(resources_dir, f"{intersection_file_basename}.parsed.expected_output.parquet")
    )
    parsed_intersection_dataframe2 = read_intersection_dataframes(
        [pjoin(resources_dir, f"{intersection_file_basename}.expected_output.parquet")],
        return_dataframes=True,
    )
    assert_frame_equal(
        parsed_intersection_dataframe.reset_index(),
        parsed_intersection_dataframe_expected,
    )
    assert_frame_equal(
        parsed_intersection_dataframe2.reset_index(),
        parsed_intersection_dataframe_expected,
    )


def test_read_intersection_dataframes_skips_empty_parquets(tmpdir, resources_dir):
    valid_parquet = pjoin(resources_dir, f"{intersection_file_basename}.expected_output.parquet")
    empty_parquet = str(tmpdir / "sample.sig2.db_control.intersection.parquet")
    open(empty_parquet, "w").close()

    result = read_intersection_dataframes([valid_parquet, empty_parquet], return_dataframes=True)
    assert not result.empty
    assert len(result) > 0


def test_read_intersection_dataframes_all_empty(tmpdir):
    empty1 = str(tmpdir / "sample.sig1.control.intersection.parquet")
    empty2 = str(tmpdir / "sample.sig2.db_control.intersection.parquet")
    open(empty1, "w").close()
    open(empty2, "w").close()

    result = read_intersection_dataframes([empty1, empty2], return_dataframes=True)
    assert result.empty


def test_generate_synthetic_signatures(tmpdir, resources_dir):
    signature_file = pjoin(resources_dir, "mutect_mrd_signature_test.vcf.gz")
    db_file = pjoin(
        resources_dir,
        "pancan_pcawg_2020.mutations_hg38_GNOMAD_dbsnp_beds.sorted.Annotated.HMER_LEN.edited.chr19.vcf.gz",
    )
    synthetic_signature_list = generate_synthetic_signatures(
        signature_vcf=signature_file, db_vcf=db_file, n_synthetic_signatures=1, output_dir=tmpdir
    )
    signature = read_signature(synthetic_signature_list[0], return_dataframes=True)
    expected_signature = read_signature(pjoin(resources_dir, "synthetic_signature_test.vcf.gz"), return_dataframes=True)
    # test that motif distribution is the same (0th order)
    assert (
        signature.groupby(["ref", "alt"]).value_counts() == expected_signature.groupby(["ref", "alt"]).value_counts()
    ).all()


def test_calc_tumor_fraction_denominator_ratio(tmpdir, resources_dir):
    """Test calc_tumor_fraction_denominator_ratio function with real parquet file."""
    featuremap_file = pjoin(resources_dir, "416119_L7402.featuremap_df.10k.parquet")
    metadata_file = pjoin(resources_dir, "416119_L7402.srsnv_metadata.json")

    # Test with a simple query that uses columns from the parquet file
    read_filter_query = "filt>0 and snvq>60"

    # Run the function
    denom_ratio, filt_ratio, read_filter_non_filt = calc_tumor_fraction_denominator_ratio(
        featuremap_df_file=featuremap_file,
        srsnv_metadata_json=metadata_file,
        read_filter_query=read_filter_query,
    )

    # Verify return values are floats
    assert isinstance(denom_ratio, float | np.floating)
    assert isinstance(filt_ratio, float | np.floating)
    assert isinstance(read_filter_non_filt, float | np.floating)

    # Verify values
    assert np.isclose(filt_ratio, 0.7012110035188417)
    assert np.isclose(read_filter_non_filt, 0.658410138248848)
    assert np.isclose(denom_ratio, 0.461684433768454)

    # Verify the calculation: denom_ratio = filt_ratio * read_filter_non_filt
    assert np.isclose(denom_ratio, filt_ratio * read_filter_non_filt)


def test_calc_tumor_fraction_denominator_ratio_rows_metadata(tmpdir, resources_dir):
    """Test calc_tumor_fraction_denominator_ratio supports metadata with rows counts."""
    featuremap_file = pjoin(resources_dir, "416119_L7402.featuremap_df.10k.parquet")
    metadata_file = pjoin(resources_dir, "416119_L7402.srsnv_metadata.json")
    read_filter_query = "filt>0 and snvq>60"

    with open(metadata_file) as f:
        metadata = json.load(f)

    for filter_step in metadata["filtering_stats"]["positive"]["filters"]:
        if "funnel" in filter_step:
            filter_step["rows"] = filter_step.pop("funnel")

    metadata_rows_file = tmpdir / "416119_L7402.srsnv_metadata.rows.json"
    with open(metadata_rows_file, "w") as f:
        json.dump(metadata, f)

    denom_ratio, filt_ratio, read_filter_non_filt = calc_tumor_fraction_denominator_ratio(
        featuremap_df_file=featuremap_file,
        srsnv_metadata_json=str(metadata_rows_file),
        read_filter_query=read_filter_query,
    )

    assert isinstance(denom_ratio, float | np.floating)
    assert isinstance(filt_ratio, float | np.floating)
    assert isinstance(read_filter_non_filt, float | np.floating)

    assert np.isclose(filt_ratio, 0.7012110035188417)
    assert np.isclose(read_filter_non_filt, 0.658410138248848)
    assert np.isclose(denom_ratio, 0.461684433768454)


def test_read_and_filter_features_parquet_noise_filter(tmp_path):
    """
    Verify the noisy loci filter flags/removes the correct loci.

    Scenario (read_filter_query = "filt>0 and snvq>60 and mapq>=60"):
    - Locus A (pos=100): 1 HQ + 2 LQ reads  -> NOISY (n_lq=2>=1)
    - Locus B (pos=200): 5 HQ + 1 LQ reads  -> NOISY (n_lq=1>=1; no exemption)
    - Locus C (pos=300): 3 HQ + 0 LQ reads  -> CLEAN (n_lq=0<1)
    """
    from ugbio_mrd.mrd_utils import read_and_filter_features_parquet

    read_filter_query = "filt>0 and snvq>60 and mapq>=60"
    loci = [(100, 1, 2), (200, 5, 1), (300, 3, 0)]
    rows = []
    counter = 0
    for pos, n_hq, n_lq in loci:
        for snvq in [80] * n_hq + [40] * n_lq:
            rows.append(
                {
                    "chrom": "chr1",
                    "pos": pos,
                    "signature": "sig1",
                    "signature_type": "matched",
                    "ref": "A",
                    "alt": "T",
                    "filt": 1,
                    "snvq": snvq,
                    "mapq": 60,
                    "ad_a": 100,
                    "ad_c": 5,
                    "ad_g": 3,
                    "ad_t": 10,
                    "dp": 118,
                    "dp_filt": 100,
                    "mi": None,
                    "rn": f"read_{pos}_{counter}",
                    "ad_del": 2,
                    "ad_ins": 1,
                }
            )
            counter += 1

    df_features_raw = pd.DataFrame(rows)
    parquet_path = str(tmp_path / "test_features.parquet")
    df_features_raw.to_parquet(parquet_path, engine="fastparquet", index=False)

    df_features, df_features_filt, _ = read_and_filter_features_parquet(
        parquet_path,
        read_filter_query,
        thresh_noise_lq_reads=1,
    )

    assert "locus_filter_noise" in df_features.columns
    assert "n_lq_reads_per_locus" in df_features.columns

    pos_idx = df_features.index.get_level_values("pos")

    assert df_features.loc[pos_idx == 100, "locus_filter_noise"].all(), "Locus A should be noisy"
    assert (df_features.loc[pos_idx == 100, "n_lq_reads_per_locus"] == 2).all()
    assert df_features.loc[pos_idx == 200, "locus_filter_noise"].all(), "Locus B should be noisy (no exemption)"
    assert not df_features.loc[pos_idx == 300, "locus_filter_noise"].any(), "Locus C should be clean"

    filt_pos = df_features_filt.index.get_level_values("pos")
    assert 100 not in filt_pos, "Locus A should be removed from df_features_filt"
    assert 200 not in filt_pos, "Locus B should be removed from df_features_filt (noisy, no exemption)"
    assert 300 in filt_pos, "Locus C HQ reads should remain"


def test_read_and_filter_features_parquet_noise_filter_disabled(tmp_path):
    """When thresh_noise_lq_reads=None the noise filter columns must not appear."""
    from ugbio_mrd.mrd_utils import read_and_filter_features_parquet

    read_filter_query = "filt>0 and snvq>60 and mapq>=60"
    df_features_raw = pd.DataFrame(
        [
            {
                "chrom": "chr1",
                "pos": 100,
                "signature": "sig1",
                "signature_type": "matched",
                "ref": "A",
                "alt": "T",
                "filt": 1,
                "snvq": 40,
                "mapq": 60,
                "ad_a": 100,
                "ad_c": 5,
                "ad_g": 3,
                "ad_t": 10,
                "dp": 118,
                "dp_filt": 100,
                "mi": None,
                "rn": "read_0",
                "ad_del": 2,
                "ad_ins": 1,
            }
        ]
    )
    parquet_path = str(tmp_path / "test_features_disabled.parquet")
    df_features_raw.to_parquet(parquet_path, engine="fastparquet", index=False)

    df_features, _, _ = read_and_filter_features_parquet(parquet_path, read_filter_query, thresh_noise_lq_reads=None)
    assert "locus_filter_noise" not in df_features.columns
    assert "n_lq_reads_per_locus" not in df_features.columns


# ─────────────────────────────────────────────────────────────────────────────
# apply_multi_read_locus_filter
# ─────────────────────────────────────────────────────────────────────────────


def _make_multi_read_test_data():
    """
    Build minimal df_features_filt, df_tf, df_signatures_filt for multi-read
    locus filter tests.

    Loci:
      (chr1, 100) — 5 HQ reads  → "hot" locus (should be filtered at low TF)
      (chr1, 200) — 1 HQ read
      (chr1, 300) — 1 HQ read
    TF (matched_ctdna_vaf) = 0.0001
    mean_coverage = 1000  → lambda = 0.1 per locus
    signature_size = 3
    P(X >= 5 | Poisson(0.1)) * 3 ≈ 1.9e-7 * 3 ≈ 5.7e-7  << threshold 0.01
    P(X >= 1 | Poisson(0.1)) * 3 ≈ 0.095 * 3 ≈ 0.28      > threshold 0.01
    """
    records = []
    for locus_pos, n_reads in [(100, 5), (200, 1), (300, 1)]:
        for i in range(n_reads):
            records.append(  # noqa: PERF401
                {
                    "chrom": "chr1",
                    "pos": locus_pos,
                    "signature": "sig1",
                    "signature_type": "matched",
                }
            )
    df_features_filt = pd.DataFrame(records).set_index(["chrom", "pos"])

    df_tf = pd.DataFrame(
        [{"ctdna_vaf": 0.0001, "supporting_reads": 7, "corrected_coverage": 1000.0}],
        index=pd.MultiIndex.from_tuples([("matched", "sig1")], names=["signature_type", "signature"]),
    )

    df_signatures_filt = pd.DataFrame(
        [{"signature": "sig1", "signature_type": "matched", "coverage": 1000.0} for _ in range(3)]
    )
    return df_features_filt, df_tf, df_signatures_filt


def test_apply_multi_read_locus_filter_removes_hot_locus():
    """Loci with unexpectedly many reads (Bonferroni Poisson p < threshold) must be removed."""
    from ugbio_mrd.mrd_utils import apply_multi_read_locus_filter

    df_features_filt, df_tf, df_signatures_filt = _make_multi_read_test_data()
    df_out, info = apply_multi_read_locus_filter(df_features_filt, df_tf, df_signatures_filt, 0.01)

    # Hot locus (chr1, 100) with 5 reads should be removed
    assert info["n_filtered_loci"] == 1
    assert info["n_filtered_reads"] == 5
    assert len(df_out) == 2  # 7 total - 5 removed = 2

    # Remaining rows must only be from clean loci
    remaining_positions = df_out.index.get_level_values("pos").tolist()
    assert 100 not in remaining_positions
    assert 200 in remaining_positions
    assert 300 in remaining_positions

    # Metadata
    assert info["poisson_lambda"] > 0
    assert info["min_bonferroni_pval"] < 0.01
    assert info["max_reads_per_locus"] == 1  # post-filter max: hot locus removed, remaining have 1 read each


def test_apply_multi_read_locus_filter_no_outliers():
    """When no loci exceed the Bonferroni threshold, df_features_filt must be unchanged."""
    from ugbio_mrd.mrd_utils import apply_multi_read_locus_filter

    df_features_filt, df_tf, df_signatures_filt = _make_multi_read_test_data()
    # Use a very strict threshold so that even the hot locus is NOT filtered
    df_out, info = apply_multi_read_locus_filter(
        df_features_filt, df_tf, df_signatures_filt, thresh_multi_read_pvalue=1e-20
    )

    assert info["n_filtered_loci"] == 0
    assert info["n_filtered_reads"] == 0
    assert len(df_out) == len(df_features_filt)


def test_apply_multi_read_matched_filter_preserves_control_rows():
    """Control-signature rows at the same locus must not be removed."""
    from ugbio_mrd.mrd_utils import apply_multi_read_locus_filter

    df_features_filt, df_tf, df_signatures_filt = _make_multi_read_test_data()

    # Add a control row at the hot locus
    control_row = pd.DataFrame(
        [{"chrom": "chr1", "pos": 100, "signature": "ctrl1", "signature_type": "control"}]
    ).set_index(["chrom", "pos"])
    df_features_with_ctrl = pd.concat([df_features_filt, control_row])

    df_out, info = apply_multi_read_locus_filter(df_features_with_ctrl, df_tf, df_signatures_filt, 0.01)

    # Matched reads at locus 100 must be removed, but the control row must survive
    assert info["n_filtered_loci"] == 1
    remaining = df_out.reset_index()
    ctrl_rows = remaining[(remaining["pos"] == 100) & (remaining["signature_type"] == "control")]
    assert len(ctrl_rows) == 1


def test_apply_multi_read_locus_filter_zero_vaf():
    """When matched_ctdna_vaf is 0, filter must be skipped gracefully."""
    from ugbio_mrd.mrd_utils import apply_multi_read_locus_filter

    df_features_filt, df_tf, df_signatures_filt = _make_multi_read_test_data()
    df_tf_zero = df_tf.copy()
    df_tf_zero["ctdna_vaf"] = 0.0

    df_out, info = apply_multi_read_locus_filter(df_features_filt, df_tf_zero, df_signatures_filt, 0.01)
    assert info["n_filtered_loci"] == 0
    assert len(df_out) == len(df_features_filt)


def test_apply_multi_read_locus_filter_no_matched_key():
    """When there is no 'matched' key in df_tf, filter must return original df unchanged."""
    from ugbio_mrd.mrd_utils import apply_multi_read_locus_filter

    df_features_filt, _df_tf, df_signatures_filt = _make_multi_read_test_data()
    df_tf_no_matched = pd.DataFrame(
        [{"ctdna_vaf": 0.001}],
        index=pd.MultiIndex.from_tuples([("control", "ctrl1")], names=["signature_type", "signature"]),
    )

    df_out, info = apply_multi_read_locus_filter(df_features_filt, df_tf_no_matched, df_signatures_filt, 0.01)
    assert info["n_filtered_loci"] == 0
    assert len(df_out) == len(df_features_filt)


def test_apply_multi_read_locus_filter_skips_db_control_rows():
    """Synthetic (db_control) rows at the same outlier locus must not be removed."""
    from ugbio_mrd.mrd_utils import apply_multi_read_locus_filter

    df_features_filt, df_tf, df_signatures_filt = _make_multi_read_test_data()

    # Add a db_control row at the hot locus (pos=100)
    db_ctrl_row = pd.DataFrame(
        [{"chrom": "chr1", "pos": 100, "signature": "syn0", "signature_type": "db_control"}]
    ).set_index(["chrom", "pos"])
    df_features_with_db_ctrl = pd.concat([df_features_filt, db_ctrl_row])

    df_out, info = apply_multi_read_locus_filter(df_features_with_db_ctrl, df_tf, df_signatures_filt, 0.01)

    # Matched reads at locus 100 must be removed
    assert info["n_filtered_loci"] == 1
    remaining = df_out.reset_index()
    # db_control row at the hot locus must survive (only matched rows are filtered by matched-filter)
    db_ctrl_rows = remaining[(remaining["pos"] == 100) & (remaining["signature_type"] == "db_control")]
    assert len(db_ctrl_rows) == 1
    # All matched reads at the hot locus must be gone
    matched_hot = remaining[(remaining["pos"] == 100) & (remaining["signature_type"] == "matched")]
    assert len(matched_hot) == 0


def _make_ctrl_filter_test_data():
    """
    Build df_features_filt, df_tf, df_signatures_filt for control/db_control filter tests.

    Signature loci: (chr1, 100), (chr1, 200), (chr1, 300) — all 1 matched read each.
    matched TF (ctdna_vaf) = 1e-4, corrected_coverage = 1000.

    Control data at same loci (1 cohort control sample + 2 synthetic controls):
      - ctrl0 (cohort):  (chr1, 100) = 10 reads (HOT), (chr1, 200) = 1, (chr1, 300) = 1
        corrected_coverage = 30_000 (large signature), ctdna_vaf ≈ 12/30000 = 4e-4
        → λ_ctrl0 = 4e-4 × 1000 = 0.4  →  P(X≥10 | Poi(0.4)) × 3 ≈ tiny  → OUTLIER
      - syn0 (db_ctrl):  (chr1, 100-300) = 1 read each.  ctdna_vaf = 3/3000 = 1e-3
      - syn1 (db_ctrl):  (chr1, 100-200) = 1 read, (chr1, 300) = 0.  ctdna_vaf = 2/3000
    """
    matched_records = []
    for pos in [100, 200, 300]:
        matched_records.append({"chrom": "chr1", "pos": pos, "signature": "sig1", "signature_type": "matched"})
    ctrl_records = []
    # Cohort control: pos=100 is hot (10 reads), others have 1 read
    for pos, n_reads in [(100, 10), (200, 1), (300, 1)]:
        for _ in range(n_reads):
            ctrl_records.append({"chrom": "chr1", "pos": pos, "signature": "ctrl0", "signature_type": "control"})
    # Two synthetic controls: each 1 read at pos 100, 200 and 0 at 300
    for syn_sig, reads_per_pos in [("syn0", {100: 1, 200: 1, 300: 1}), ("syn1", {100: 1, 200: 1, 300: 0})]:
        for pos, n_reads in reads_per_pos.items():
            for _ in range(n_reads):
                ctrl_records.append({"chrom": "chr1", "pos": pos, "signature": syn_sig, "signature_type": "db_control"})

    all_records = matched_records + ctrl_records
    df_features_filt = pd.DataFrame(all_records).set_index(["chrom", "pos"])

    # df_tf: matched + cohort control + 2 db_control entries.
    # ctrl0 has corrected_coverage=30_000 (large signature) → very low ctdna_vaf → hot locus is a clear outlier.
    index = pd.MultiIndex.from_tuples(
        [("matched", "sig1"), ("control", "ctrl0"), ("db_control", "syn0"), ("db_control", "syn1")],
        names=["signature_type", "signature"],
    )
    df_tf = pd.DataFrame(
        {
            "ctdna_vaf": [1e-4, 12 / 30_000, 3 / 3000, 2 / 3000],
            "supporting_reads": [3, 12, 3, 2],
            "corrected_coverage": [1000.0, 30_000.0, 1000.0, 1000.0],
        },
        index=index,
    )

    df_signatures_filt = pd.DataFrame(
        [{"signature": "sig1", "signature_type": "matched", "coverage": 1000.0} for _ in range(3)]
    )
    return df_features_filt, df_tf, df_signatures_filt


def test_apply_multi_read_locus_filter_removes_outlier_control_loci():
    """Cohort control loci with unexpectedly many reads must be removed."""
    from ugbio_mrd.mrd_utils import apply_multi_read_locus_filter

    df_features_filt, df_tf, df_signatures_filt = _make_ctrl_filter_test_data()
    df_out, info = apply_multi_read_locus_filter(df_features_filt, df_tf, df_signatures_filt, 0.01)

    # The hot cohort control locus (chr1, 100) should be removed
    assert info["n_filtered_control_loci"] == 1
    assert info["n_filtered_control_reads"] == 10

    remaining = df_out.reset_index()
    ctrl_hot = remaining[(remaining["pos"] == 100) & (remaining["signature_type"] == "control")]
    assert len(ctrl_hot) == 0, "cohort control reads at hot locus should be removed"

    ctrl_clean = remaining[(remaining["pos"] == 200) & (remaining["signature_type"] == "control")]
    assert len(ctrl_clean) == 1, "cohort control reads at clean locus must remain"

    # Matched reads must be unaffected
    assert info["n_filtered_loci"] == 0


def test_apply_multi_read_locus_filter_removes_outlier_db_control_loci():
    """Synthetic (db_control) loci with unexpectedly many reads must be removed."""
    from ugbio_mrd.mrd_utils import apply_multi_read_locus_filter

    df_features_filt, df_tf, df_signatures_filt = _make_ctrl_filter_test_data()

    # Override df_tf: make syn0 have a very hot locus at pos 100 (many total reads)
    # Use a single HOT db_control signature with 10 reads at one locus
    records = list(df_features_filt.reset_index().to_dict("records"))
    # Add 9 more db_control reads at pos 100 for syn0 (already has 1)
    for _ in range(9):
        records.append({"chrom": "chr1", "pos": 100, "signature": "syn0", "signature_type": "db_control"})
    df_features_hot = pd.DataFrame(records).set_index(["chrom", "pos"])

    df_out, info = apply_multi_read_locus_filter(df_features_hot, df_tf, df_signatures_filt, 0.01)

    # Outlier db_control locus must be removed
    assert info["n_filtered_db_control_loci"] >= 1
    remaining = df_out.reset_index()
    db_ctrl_hot = remaining[(remaining["pos"] == 100) & (remaining["signature_type"] == "db_control")]
    assert len(db_ctrl_hot) == 0, "db_control reads at hot locus should be removed"

    # Matched reads at same locus must be unaffected (different type)
    matched_100 = remaining[(remaining["pos"] == 100) & (remaining["signature_type"] == "matched")]
    assert len(matched_100) == 1


def test_apply_multi_read_locus_filter_never_removes_single_read_loci():
    """Single-read loci must never be filtered regardless of how low the TF estimate is."""
    from ugbio_mrd.mrd_utils import apply_multi_read_locus_filter

    # Use an extremely low TF so that the Bonferroni test would flag single reads
    # without the guard (TF=1e-9 << the 5e-8 transition point).
    tiny_vaf = 1e-9
    records = [
        {"chrom": "chr1", "pos": 100, "signature": "sig1", "signature_type": "matched"},
        {"chrom": "chr1", "pos": 200, "signature": "sig1", "signature_type": "matched"},
        {"chrom": "chr1", "pos": 300, "signature": "sig1", "signature_type": "matched"},
    ]
    df_features_filt = pd.DataFrame(records).set_index(["chrom", "pos"])

    df_tf = pd.DataFrame(
        [{"ctdna_vaf": tiny_vaf, "supporting_reads": 3, "corrected_coverage": 1000.0}],
        index=pd.MultiIndex.from_tuples([("matched", "sig1")], names=["signature_type", "signature"]),
    )
    df_signatures_filt = pd.DataFrame(
        [{"signature": "sig1", "signature_type": "matched", "coverage": 1000.0} for _ in range(1000)]
    )

    df_out, info = apply_multi_read_locus_filter(df_features_filt, df_tf, df_signatures_filt, 0.05)

    assert info["n_filtered_loci"] == 0, "Single-read loci must never be removed"
    assert len(df_out) == len(df_features_filt)


# ─────────────────────────────────────────────────────────────────────────────
# Boundary / sensitivity tests — how many reads trigger the filter at each TF
# ─────────────────────────────────────────────────────────────────────────────


def _min_reads_to_filter(matched_vaf: float, mean_coverage: float, sig_size: int, thresh: float) -> int:
    """
    Return the smallest k such that the Bonferroni-corrected Poisson p-value
    P(X ≥ k | Poisson(matched_vaf × mean_coverage)) × sig_size < thresh.
    This is the exact criterion used by apply_multi_read_locus_filter.
    """
    from scipy.stats import poisson

    lam = matched_vaf * mean_coverage
    for k in range(1, 500):
        if poisson.sf(k - 1, lam) * sig_size < thresh:
            return k
    raise ValueError(f"No filter threshold found for λ={lam}")


def _make_threshold_test_fixtures(matched_vaf: float, mean_coverage: float, sig_size: int):
    """
    Build (df_features_filt, df_tf, df_signatures_filt) for a threshold test.
    The locus at pos=100 is set up as a HOT locus (read count injected by the caller);
    pos=200 and pos=300 each have a single read.
    """
    # df_features_filt is built by the caller – return only df_tf and df_signatures_filt here
    df_tf = pd.DataFrame(
        [
            {
                "ctdna_vaf": matched_vaf,
                "supporting_reads": 1,
                "corrected_coverage": float(mean_coverage),
            }
        ],
        index=pd.MultiIndex.from_tuples([("matched", "sig1")], names=["signature_type", "signature"]),
    )
    # sig_size matched loci, each with mean_coverage coverage
    df_signatures_filt = pd.DataFrame(
        [
            {"signature": "sig1", "signature_type": "matched", "coverage": float(mean_coverage)}
            for _ in range(sig_size)
        ]
    )
    return df_tf, df_signatures_filt


@pytest.mark.parametrize(
    "matched_vaf,mean_coverage,sig_size,thresh,expected_k_min",
    [
        # λ = matched_vaf × mean_coverage
        # expected_k_min: minimum reads per locus that triggers the filter
        (1e-5, 1000, 1000, 0.05, 2),  # λ = 0.01  — very low TF
        (1e-4, 1000, 1000, 0.05, 4),  # λ = 0.10  — low TF
        (5e-4, 1000, 1000, 0.05, 6),  # λ = 0.50  — intermediate
        (1e-3, 1000, 1000, 0.05, 8),  # λ = 1.00  — medium TF
        (5e-3, 1000, 1000, 0.05, 17),  # λ = 5.00  — high TF
        (1e-2, 1000, 1000, 0.05, 25),  # λ = 10.0  — very high TF
    ],
)
def test_apply_multi_read_filter_boundary_at_each_tf(matched_vaf, mean_coverage, sig_size, thresh, expected_k_min):
    """
    At each TF level verify that:
    * A locus with exactly k_min - 1 reads is NOT filtered.
    * A locus with exactly k_min reads IS filtered.

    k_min is the minimum per-locus read count that makes the Bonferroni
    Poisson p-value fall below *thresh*.  The test also prints k_min so the
    caller can see the sensitivity at each TF.
    """
    from ugbio_mrd.mrd_utils import apply_multi_read_locus_filter

    lam = matched_vaf * mean_coverage
    k_min = _min_reads_to_filter(matched_vaf, mean_coverage, sig_size, thresh)
    assert k_min == expected_k_min, f"TF={matched_vaf}, λ={lam:.3f}: expected k_min={expected_k_min}, got {k_min}"

    df_tf, df_signatures_filt = _make_threshold_test_fixtures(matched_vaf, mean_coverage, sig_size)

    def _build_features(reads_at_hot_locus: int) -> pd.DataFrame:
        rows = [
            {"chrom": "chr1", "pos": 100, "signature": "sig1", "signature_type": "matched"}
            for _ in range(reads_at_hot_locus)
        ] + [
            {"chrom": "chr1", "pos": 200, "signature": "sig1", "signature_type": "matched"},
            {"chrom": "chr1", "pos": 300, "signature": "sig1", "signature_type": "matched"},
        ]
        return pd.DataFrame(rows).set_index(["chrom", "pos"])

    # ── one read below threshold: must NOT be filtered ──
    if k_min > 1:
        df_below = _build_features(k_min - 1)
        df_out_below, info_below = apply_multi_read_locus_filter(df_below, df_tf, df_signatures_filt, thresh)
        assert info_below["n_filtered_loci"] == 0, (
            f"TF={matched_vaf}, λ={lam:.3f}: locus with {k_min - 1} reads should NOT be filtered "
            f"(Bonferroni p={info_below['min_bonferroni_pval']:.4f} ≥ {thresh}), k_min={k_min}"
        )
        assert len(df_out_below) == len(df_below)

    # ── exactly at threshold: must BE filtered ──
    df_at = _build_features(k_min)
    df_out_at, info_at = apply_multi_read_locus_filter(df_at, df_tf, df_signatures_filt, thresh)
    assert info_at["n_filtered_loci"] == 1, (
        f"TF={matched_vaf}, λ={lam:.3f}: locus with {k_min} reads SHOULD be filtered "
        f"(Bonferroni p={info_at['min_bonferroni_pval']:.4f} < {thresh}), k_min={k_min}"
    )
    # Clean loci (single reads at pos 200, 300) must survive
    remaining = df_out_at.reset_index()
    assert set(remaining["pos"].unique()) == {
        200,
        300,
    }, f"Only the hot locus (pos=100) should be removed; k_min={k_min}"
