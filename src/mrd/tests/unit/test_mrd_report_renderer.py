import pandas as pd
from ugbio_mrd.mrd_report_renderer import render_intersection_snvq_combined


def test_render_intersection_snvq_combined_all_below_display_floor():
    """Regression test for BIOIN crash: all reads present have snvq < 40 (the hard-coded
    display floor), so matched/cohort/db_ctrl are non-empty before filtering but all become
    empty afterwards. Previously raised ValueError: No objects to concatenate (pd.concat on
    an empty list) inside pd.concat's own emptiness check.
    """
    features_df = pd.DataFrame(
        {
            "signature_type": ["matched"] * 5 + ["db_control"] * 5,
            "snvq": [0.0] * 10,
        }
    )

    result = render_intersection_snvq_combined(features_df)

    assert result == ""


def test_render_intersection_snvq_combined_some_above_display_floor():
    features_df = pd.DataFrame(
        {
            "signature_type": ["matched"] * 5 + ["db_control"] * 5,
            "snvq": [0.0] * 4 + [80.0] + [0.0] * 5,
        }
    )

    result = render_intersection_snvq_combined(features_df)

    assert result != ""
