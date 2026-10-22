"""End-to-end ``consensus_report`` run on a real ReadFuserAlignSort output subset.

The fixtures are the ReadFuserAlignSort nightly regression output for the region
``chr1:13400000-13403000`` (sample ``502285-L11546-Z0236``), subset to a
self-contained BAM so the CRAM's hg38 reference is not needed to decode it. The
BAM keeps the consensus ``rn:Z`` read-name list and the ``nf:i``/``nr:i`` strand
tags, so the duplex/family path is exercised for real; ``--duplex-chrom all``
scans the whole input with no region limitation.
"""

from pathlib import Path

import pandas as pd
import pytest
from ugbio_consensus import consensus_report


@pytest.fixture
def resources_dir():
    return Path(__file__).parent.parent / "resources"


@pytest.fixture
def region_inputs(resources_dir):
    """Local input files for the chr1:13400000-13403000 test region."""
    return {
        "cram": resources_dir / "consensus.chr1_13400000_13403000.bam",
        "sorter_stats_csv": resources_dir / "sorter_stats.chr1_13400000_13403000.csv",
        "sorter_stats_json": resources_dir / "sorter_stats.chr1_13400000_13403000.json",
        "bedgraph": resources_dir / "coverage.chr1_13400000_13403000.bedGraph.gz",
        "consensus_log": resources_dir / "consensus.chr1_13400000_13403000.stdout.log",
    }


def test_consensus_report_on_chr1_region(region_inputs, tmp_path):
    """Full CLI run: HTML + sidecar CSVs, with real duplex metrics from nf/nr tags."""
    output = tmp_path / "report.html"
    metrics = consensus_report.run(
        [
            "consensus_report",
            "--name",
            "Z0236_chr1_test",
            "--cram",
            str(region_inputs["cram"]),
            "--sorter-stats-csv",
            str(region_inputs["sorter_stats_csv"]),
            "--sorter-stats-json",
            str(region_inputs["sorter_stats_json"]),
            "--bedgraph",
            str(region_inputs["bedgraph"]),
            "--consensus-log",
            str(region_inputs["consensus_log"]),
            # BAM embeds its sequence, so no external reference is needed to decode it.
            "--reference",
            "/dev/null",
            # Scan the whole input, no region limitation.
            "--duplex-chrom",
            "all",
            "--output",
            str(output),
        ]
    )

    # --- Outputs written ---
    assert output.exists()
    per_sample_csv = tmp_path / "report_per_sample.csv"
    manifest_csv = tmp_path / "report_manifest.csv"
    assert per_sample_csv.exists()
    assert manifest_csv.exists()

    html = output.read_text()
    assert "Both-strands duplex" in html
    assert "Consensus tool performance" in html

    # --- One sample, expected duplex/consensus values for this fixed region ---
    assert isinstance(metrics, pd.DataFrame)
    assert len(metrics) == 1
    row = metrics.iloc[0]
    assert row["sample"] == "Z0236_chr1_test"

    # --duplex-chrom all scans the whole CRAM, so all 533 merged consensus reads
    # are classified: 215 both-strands duplex + 318 single-strand (the rest of the
    # BAM passes through as singletons). This matches the whole-run dup_sets_merged
    # total from the consensus log (533) below.
    assert row["duplex_n_reads"] == 215
    assert row["single_strand_n_reads"] == 318
    assert row["duplex_n_reads"] + row["single_strand_n_reads"] == row["consensus_dup_sets_merged"]
    assert row["duplex_avg_family_size"] == pytest.approx(2.1441860, rel=1e-4)
    assert row["single_strand_avg_family_size"] == pytest.approx(2.1069182, rel=1e-4)

    # Consensus stdout log parsed into the table: whole-run dup-set totals.
    assert row["consensus_dup_sets_merged"] == 533
    assert row["consensus_unchanged_segments"] == 4791

    # Coverage came from the bedGraph (genome-wide; no targets BED -> on-target is null).
    assert row["genome_mean_cvg"] > 0
    assert pd.isna(row["on_target_rate"])


def test_three_way_input_singletons_consensus_report(region_inputs, tmp_path):
    """The RFAS report shape: input / singletons / consensus rows side by side.

    Only the consensus row gets a ``cram=``: it is the only population carrying
    ``rn``/``nf``/``nr``, so it is the only one worth scanning for duplex families.
    The other two rows contribute alignment metrics alone. ``--no-summary`` drops
    the median-across-samples table, which is meaningless across these three.
    """
    output = tmp_path / "three_way.html"
    csv = str(region_inputs["sorter_stats_csv"])
    js = str(region_inputs["sorter_stats_json"])

    metrics = consensus_report.run(
        [
            "consensus_report",
            # Input CRAM: alignment metrics only, no duplex scan.
            "--sample",
            "name=Z0236.input",
            f"sorter_stats_csv={csv}",
            f"sorter_stats_json={js}",
            # Singleton (pass-through) CRAM: alignment metrics only.
            "--sample",
            "name=Z0236.singletons",
            f"sorter_stats_csv={csv}",
            f"sorter_stats_json={js}",
            # Consensus CRAM: the only one scanned for families.
            "--sample",
            "name=Z0236.consensus",
            f"cram={region_inputs['cram']}",
            f"sorter_stats_csv={csv}",
            f"sorter_stats_json={js}",
            f"bedgraph={region_inputs['bedgraph']}",
            f"consensus_log={region_inputs['consensus_log']}",
            "--reference",
            "/dev/null",
            "--duplex-chrom",
            "all",
            "--no-summary",
            "--output",
            str(output),
        ]
    )

    assert output.exists()
    html = output.read_text()
    assert "Summary (median across samples)" not in html
    assert "Per-sample metrics" in html

    assert len(metrics) == 3
    assert list(metrics["sample"]) == ["Z0236.input", "Z0236.singletons", "Z0236.consensus"]
    by_name = metrics.set_index("sample")

    # Alignment metrics are populated for all three rows.
    assert by_name["PCT_duplicates"].notna().all()

    # Duplex metrics only for the consensus row; the two CRAM-less rows are null.
    assert by_name.loc["Z0236.consensus", "duplex_n_reads"] == 215
    assert by_name.loc["Z0236.consensus", "single_strand_n_reads"] == 318
    assert pd.isna(by_name.loc["Z0236.input", "duplex_n_reads"])
    assert pd.isna(by_name.loc["Z0236.singletons", "duplex_n_reads"])
