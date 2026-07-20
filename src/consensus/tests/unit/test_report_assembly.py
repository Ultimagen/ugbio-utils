"""Report table/figure assembly from a synthetic per-sample metrics table."""

import pandas as pd
import pytest
from ugbio_consensus import consensus_report, duplex_metrics


@pytest.fixture
def metrics_df():
    return pd.DataFrame(
        [
            {
                "sample": "S1",
                "PCT_duplicates": 34.85,
                "Mean_cvg": 19.49,
                "genome_mean_cvg": 10.5,
                "on_target_rate": 0.286,
                "target_mean_cvg": 236.2,
                f"{duplex_metrics.DUPLEX}_avg_family_size": 13.0,
                f"{duplex_metrics.SINGLE_STRAND}_avg_family_size": 6.0,
                f"{duplex_metrics.DUPLEX}_coverage": 12.4,
                f"{duplex_metrics.SINGLE_STRAND}_coverage": 16.9,
                "consensus_PCT_unhandled": 9.8,
                "consensus_PCT_dup_set_members": 90.2,
            },
            {
                "sample": "S2",
                "PCT_duplicates": 36.0,
                "Mean_cvg": 21.0,
                "genome_mean_cvg": 11.0,
                "on_target_rate": 0.30,
                "target_mean_cvg": 250.0,
                f"{duplex_metrics.DUPLEX}_avg_family_size": 14.0,
                f"{duplex_metrics.SINGLE_STRAND}_avg_family_size": 6.5,
                f"{duplex_metrics.DUPLEX}_coverage": 13.0,
                f"{duplex_metrics.SINGLE_STRAND}_coverage": 17.5,
                "consensus_PCT_unhandled": 10.2,
                "consensus_PCT_dup_set_members": 89.8,
            },
        ]
    )


def test_summarize_medians(metrics_df):
    summary = consensus_report.summarize(metrics_df)
    assert summary["PCT_duplicates"] == pytest.approx(35.425)
    assert summary[f"{duplex_metrics.DUPLEX}_avg_family_size"] == pytest.approx(13.5)


def test_generate_report_html(metrics_df, tmp_path):
    out = tmp_path / "report.html"
    result = consensus_report.generate_report(metrics_df, str(out), title="My Consensus Report")
    assert result == str(out)
    content = out.read_text()
    assert "My Consensus Report" in content
    assert "plotly" in content.lower()
    # duplex, on-target and consensus-performance sections all present
    assert "Both-strands duplex" in content
    assert "On-target rate" in content
    assert "Consensus tool performance" in content
    assert "rs:B:i" in content


def test_consensus_table_absent_without_log(metrics_df):
    without_log = metrics_df.drop(columns=["consensus_PCT_unhandled", "consensus_PCT_dup_set_members"])
    assert consensus_report._consensus_metrics_table(without_log) is None


def test_on_target_figure_absent_without_targets(metrics_df):
    without_targets = metrics_df.assign(on_target_rate=None)
    assert consensus_report.build_on_target_figure(without_targets) is None
