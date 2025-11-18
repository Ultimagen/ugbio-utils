from pathlib import Path

import pandas as pd
import pytest
from ugbio_mrd.generate_mrd_report import MrdReportInputs, generate_mrd_report


@pytest.fixture
def resources_dir():
    return Path(__file__).parent.parent / "resources" / "report"


@pytest.fixture
def output_path(tmpdir):
    return Path(tmpdir)


def test_generate_mrd_report(output_path, resources_dir):
    mrd_report_inputs = MrdReportInputs(
        intersected_featuremaps_parquet=[
            str(resources_dir / "Pa_46_333_LuNgs_08.Pa_46_FreshFrozen.matched.intersection.parquet"),
            str(resources_dir / "Pa_46_333_LuNgs_08.Pa_67_FFPE.control.intersection.parquet"),
            str(resources_dir / "Pa_46_333_LuNgs_08.syn0_Pa_46_FreshFrozen.db_control.intersection.parquet"),
            str(resources_dir / "Pa_46_333_LuNgs_08.syn1_Pa_46_FreshFrozen.db_control.intersection.parquet"),
            str(resources_dir / "Pa_46_333_LuNgs_08.syn2_Pa_46_FreshFrozen.db_control.intersection.parquet"),
            str(resources_dir / "Pa_46_333_LuNgs_08.syn3_Pa_46_FreshFrozen.db_control.intersection.parquet"),
            str(resources_dir / "Pa_46_333_LuNgs_08.syn4_Pa_46_FreshFrozen.db_control.intersection.parquet"),
        ],
        matched_signatures_vcf_files=[str(resources_dir / "Pa_46_FreshFrozen.ann.chr20.filtered.vcf.gz")],
        control_signatures_vcf_files=[str(resources_dir / "Pa_67_FFPE.ann.chr20.filtered.vcf.gz")],
        db_control_signatures_vcf_files=[
            str(resources_dir / "syn0_Pa_46_FreshFrozen.ann.chr20.filtered_pancan_pcawg_2020.chr20.filtered.vcf.gz"),
            str(resources_dir / "syn1_Pa_46_FreshFrozen.ann.chr20.filtered_pancan_pcawg_2020.chr20.filtered.vcf.gz"),
            str(resources_dir / "syn2_Pa_46_FreshFrozen.ann.chr20.filtered_pancan_pcawg_2020.chr20.filtered.vcf.gz"),
            str(resources_dir / "syn3_Pa_46_FreshFrozen.ann.chr20.filtered_pancan_pcawg_2020.chr20.filtered.vcf.gz"),
            str(resources_dir / "syn4_Pa_46_FreshFrozen.ann.chr20.filtered_pancan_pcawg_2020.chr20.filtered.vcf.gz"),
        ],
        coverage_bed=str(resources_dir / "Pa_46_333_LuNgs_08.regions.bed.gz"),
        # tumor_sample=args_in.tumor_sample,
        output_dir=output_path,
        output_basename="test_report",
        featuremap_file=str(resources_dir / "Pa_46_333_LuNgs_08.featuremap_df.parquet"),
        signature_filter_query="(norm_coverage <= 2.5) and (norm_coverage >= 0.6)",
        read_filter_query="filt>0 and snvq>60 and mapq>=60",
        srsnv_metadata_json=str(resources_dir / "Pa_46_333_LuNgs_08.srsnv_metadata.json"),
    )

    output_report_html = generate_mrd_report(mrd_report_inputs)

    # assert report_html exists
    assert output_report_html.exists()

    # assert report_html is not empty
    assert output_report_html.stat().st_size > 0

    # test h5 output
    h5_output = str(output_path / "test_report.ctdna_vaf.h5")
    h5_expected = str(resources_dir / "test_report.ctdna_vaf.expected_output.h5")
    with pd.HDFStore(h5_expected) as store:
        h5_keys = store.keys()
    with pd.HDFStore(h5_output) as store:
        h5_keys_output = store.keys()

    # assert that the keys in the output h5 file are the same as the keys in the expected
    assert h5_keys == h5_keys_output

    # assert h5 output values are as expected
    for key in h5_keys:
        pd.testing.assert_frame_equal(pd.read_hdf(h5_output, key), pd.read_hdf(h5_expected, key))
