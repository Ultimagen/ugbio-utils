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
            str(resources_dir / "Pa_46.Pa_46_FreshFrozen.matched.intersection.parquet"),
            str(resources_dir / "Pa_46.Pa_67_FFPE.control.intersection.parquet"),
        ],
        matched_signatures_vcf_files=[str(resources_dir / "Pa_46_FreshFrozen.ann.chr20.filtered.vcf")],
        control_signatures_vcf_files=[str(resources_dir / "Pa_67_FFPE.ann.chr20.filtered.vcf")],
        # db_control_signatures_vcf_files=args_in.db_control_signatures_vcf,
        coverage_csv=str(resources_dir / "Pa_46_matched_no_db.coverage.csv"),
        # tumor_sample=args_in.tumor_sample,
        output_dir=output_path,
        output_basename="test_report",
        featuremap_file=str(resources_dir / "Pa_46_333_LuNgs_08.featuremap_df.parquet"),
        signature_filter_query="(norm_coverage <= 2.5) and (norm_coverage >= 0.6)",
        read_filter_query="qual>60",
    )

    output_report_html = generate_mrd_report(mrd_report_inputs)

    # assert report_html exists
    assert output_report_html.exists()

    # assert report_html is not empty
    assert output_report_html.stat().st_size > 0

    # assert h5 output values are as expected
    h5_output = str(output_path / "test_report.tumor_fraction.h5")
    h5_expected = str(resources_dir / "test_report.tumor_fraction.expected_output.h5")
    with pd.HDFStore(h5_expected) as store:
        h5_keys = store.keys()
    for key in h5_keys:
        pd.testing.assert_frame_equal(pd.read_hdf(h5_output, key), pd.read_hdf(h5_expected, key))
