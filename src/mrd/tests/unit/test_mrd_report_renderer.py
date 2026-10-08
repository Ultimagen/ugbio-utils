from types import SimpleNamespace

import pandas as pd
from ugbio_mrd.mrd_report_renderer import (
    _format_vaf_ci,
    render_binomial_distribution,
    render_intersection_snvq_combined,
)


def _detection(**overrides):
    values = {
        "vaf_ci_low": 7.2e-4,
        "vaf_ci_high": 1.5e-3,
        "total_coverage": 83_788.0,
        "snvq_recall": 0.36,
        "noise_rate": 3.2e-6,
        "matched_supporting_reads": 32,
        "p_value": 1e-10,
        "alpha": 0.01,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_format_vaf_ci():
    assert _format_vaf_ci(_detection()) == "7.2 \u00d7 10\u207b\u2074 \u2013 1.5 \u00d7 10\u207b\u00b3"


def test_format_vaf_ci_zero_lower_bound():
    assert _format_vaf_ci(_detection(vaf_ci_low=0.0)).startswith("0 \u2013 ")


def test_format_vaf_ci_not_available():
    assert _format_vaf_ci(_detection(vaf_ci_low=None, vaf_ci_high=None)) == "N/A"


def test_render_binomial_distribution_uses_raw_coverage():
    assert render_binomial_distribution(_detection()) != ""
    assert render_binomial_distribution(_detection(total_coverage=0.0)) == ""


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
