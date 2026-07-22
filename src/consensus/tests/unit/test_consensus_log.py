from pathlib import Path

from ugbio_consensus.consensus_log import parse_consensus_log

_LOG = """There are dark flows : [c]
total=200020, 23660 unhandled, 176360 dup_set_members
total=400040, 53512 unhandled, 346528 dup_set_members
total=1000000, 100000 unhandled, 900000 dup_set_members
done process_all_records
stats=Stats { alt_prob_too_high: 0, hmer_too_long: 5712685, no_clear_hmer_choice: 14009063, \
not_same_contig: 4, other_error: 1006093, dup_sets_merged: 216196151, unchanged_segments: 297230665 }
"""


def _write_log(tmp_path: Path) -> str:
    p = tmp_path / "consensus.stdout.log"
    p.write_text(_LOG, encoding="utf-8")
    return str(p)


def test_parse_uses_last_total_line(tmp_path):
    metrics = parse_consensus_log(_write_log(tmp_path))
    assert metrics["total_reads"] == 1_000_000
    assert metrics["unhandled"] == 100_000
    assert metrics["dup_set_members"] == 900_000


def test_parse_derived_rates(tmp_path):
    metrics = parse_consensus_log(_write_log(tmp_path))
    assert metrics["PCT_unhandled"] == 10.0
    assert metrics["PCT_dup_set_members"] == 90.0


def test_parse_stats_counters(tmp_path):
    metrics = parse_consensus_log(_write_log(tmp_path))
    assert metrics["hmer_too_long"] == 5_712_685
    assert metrics["no_clear_hmer_choice"] == 14_009_063
    assert metrics["dup_sets_merged"] == 216_196_151
    assert metrics["unchanged_segments"] == 297_230_665
    assert metrics["alt_prob_too_high"] == 0


def test_parse_missing_pieces_are_absent(tmp_path):
    p = tmp_path / "only_stats.log"
    p.write_text("stats=Stats { hmer_too_long: 5 }\n", encoding="utf-8")
    metrics = parse_consensus_log(str(p))
    assert metrics == {"hmer_too_long": 5}
    assert "total_reads" not in metrics
