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
    Verify the noise locus filter flags/removes the correct loci.

    Scenario (read_filter_query = "filt>0 and snvq>60 and mapq>=60"):
    - Locus A (pos=100): 1 HQ + 2 LQ reads  -> NOISY  (n_lq=2>=1, n_hq=1<3)
    - Locus B (pos=200): 5 HQ + 1 LQ reads  -> EXEMPT (n_hq=5>=3)
    - Locus C (pos=300): 3 HQ + 0 LQ reads  -> CLEAN  (n_lq=0<1)
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
        thresh_noise_hq_exemption=3,
    )

    assert "locus_filter_noise" in df_features.columns
    assert "n_lq_reads_per_locus" in df_features.columns
    assert "n_hq_reads_per_locus" in df_features.columns

    pos_idx = df_features.index.get_level_values("pos")

    assert df_features.loc[pos_idx == 100, "locus_filter_noise"].all(), "Locus A should be noisy"
    assert (df_features.loc[pos_idx == 100, "n_lq_reads_per_locus"] == 2).all()
    assert (df_features.loc[pos_idx == 100, "n_hq_reads_per_locus"] == 1).all()
    assert not df_features.loc[pos_idx == 200, "locus_filter_noise"].any(), "Locus B should be exempt"
    assert not df_features.loc[pos_idx == 300, "locus_filter_noise"].any(), "Locus C should be clean"

    filt_pos = df_features_filt.index.get_level_values("pos")
    assert 100 not in filt_pos, "Locus A should be removed from df_features_filt"
    assert 200 in filt_pos, "Locus B HQ reads should remain"
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
