"""Unit tests for merge_snvfind_stats module."""

import json

import pytest
from ugbio_featuremap.merge_snvfind_stats import merge_snvfind_stats, merge_trinuc_freq


@pytest.fixture
def simple_stats_shard_1(tmp_path):
    data = {
        "filters": {
            "raw": {"funnel": 100000},
            "coverage_ge_min": {
                "name": "coverage_ge_min",
                "type": "region",
                "field": "DP",
                "op": "ge",
                "value": 20,
                "funnel": 80000,
                "pass": 80000,
            },
            "mapq_ge_60": {
                "name": "mapq_ge_60",
                "type": "mapping",
                "field": "MAPQ",
                "op": "ge",
                "value": 60,
                "funnel": 70000,
                "pass": 90000,
            },
        },
        "combinations": {"000": 5000, "001": 3000, "010": 2000, "011": 10000, "100": 1000, "111": 70000},
        "combinations_total": 100000,
    }
    path = tmp_path / "shard1.json"
    path.write_text(json.dumps(data))
    return path


@pytest.fixture
def simple_stats_shard_2(tmp_path):
    data = {
        "filters": {
            "raw": {"funnel": 50000},
            "coverage_ge_min": {
                "name": "coverage_ge_min",
                "type": "region",
                "field": "DP",
                "op": "ge",
                "value": 20,
                "funnel": 40000,
                "pass": 40000,
            },
            "mapq_ge_60": {
                "name": "mapq_ge_60",
                "type": "mapping",
                "field": "MAPQ",
                "op": "ge",
                "value": 60,
                "funnel": 35000,
                "pass": 45000,
            },
        },
        "combinations": {"000": 2500, "001": 1500, "010": 1000, "011": 5000, "100": 500, "111": 35000},
        "combinations_total": 50000,
    }
    path = tmp_path / "shard2.json"
    path.write_text(json.dumps(data))
    return path


def test_merge_two_simple_shards(tmp_path, simple_stats_shard_1, simple_stats_shard_2):
    output = tmp_path / "merged.json"
    merge_snvfind_stats([simple_stats_shard_1, simple_stats_shard_2], output)

    result = json.loads(output.read_text())

    assert result["filters"]["raw"]["funnel"] == 150000
    assert result["filters"]["coverage_ge_min"]["funnel"] == 120000
    assert result["filters"]["coverage_ge_min"]["pass"] == 120000
    assert result["filters"]["mapq_ge_60"]["funnel"] == 105000
    assert result["filters"]["mapq_ge_60"]["pass"] == 135000

    assert result["combinations"]["000"] == 7500
    assert result["combinations"]["111"] == 105000
    assert result["combinations_total"] == 150000

    assert result["filters"]["coverage_ge_min"]["name"] == "coverage_ge_min"
    assert result["filters"]["coverage_ge_min"]["value"] == 20


def test_merge_single_input(tmp_path, simple_stats_shard_1):
    output = tmp_path / "merged.json"
    merge_snvfind_stats([simple_stats_shard_1], output)

    result = json.loads(output.read_text())
    original = json.loads(simple_stats_shard_1.read_text())

    assert result["filters"]["raw"]["funnel"] == original["filters"]["raw"]["funnel"]
    assert result["combinations_total"] == original["combinations_total"]


def test_merge_unified_format(tmp_path):
    shard_data = {
        "filters_full_output": {
            "filters": {
                "raw": {"funnel": 62000000},
                "coverage_ge_min": {
                    "name": "coverage_ge_min",
                    "type": "region",
                    "field": "DP",
                    "op": "ge",
                    "value": 20,
                    "funnel": 44000000,
                    "pass": 44000000,
                },
            },
            "combinations": {"00": 18000000, "01": 0, "10": 0, "11": 44000000},
            "combinations_total": 62000000,
        },
        "filters_random_sample": {
            "filters": {
                "raw": {"funnel": 2700},
                "coverage_ge_min": {
                    "name": "coverage_ge_min",
                    "type": "region",
                    "field": "DP",
                    "op": "ge",
                    "value": 20,
                    "funnel": 1900,
                    "pass": 1900,
                },
            },
            "combinations": {"00": 800, "01": 0, "10": 0, "11": 1900},
            "combinations_total": 2700,
        },
    }

    shard1 = tmp_path / "shard1.json"
    shard1.write_text(json.dumps(shard_data))

    shard2_data = {
        "filters_full_output": {
            "filters": {
                "raw": {"funnel": 30000000},
                "coverage_ge_min": {
                    "name": "coverage_ge_min",
                    "type": "region",
                    "field": "DP",
                    "op": "ge",
                    "value": 20,
                    "funnel": 22000000,
                    "pass": 22000000,
                },
            },
            "combinations": {"00": 8000000, "01": 0, "10": 0, "11": 22000000},
            "combinations_total": 30000000,
        },
        "filters_random_sample": {
            "filters": {
                "raw": {"funnel": 1300},
                "coverage_ge_min": {
                    "name": "coverage_ge_min",
                    "type": "region",
                    "field": "DP",
                    "op": "ge",
                    "value": 20,
                    "funnel": 900,
                    "pass": 900,
                },
            },
            "combinations": {"00": 400, "01": 0, "10": 0, "11": 900},
            "combinations_total": 1300,
        },
    }
    shard2 = tmp_path / "shard2.json"
    shard2.write_text(json.dumps(shard2_data))

    output = tmp_path / "merged.json"
    merge_snvfind_stats([shard1, shard2], output)

    result = json.loads(output.read_text())

    assert result["filters_full_output"]["filters"]["raw"]["funnel"] == 92000000
    assert result["filters_full_output"]["filters"]["coverage_ge_min"]["funnel"] == 66000000
    assert result["filters_full_output"]["combinations_total"] == 92000000
    assert result["filters_full_output"]["combinations"]["11"] == 66000000

    assert result["filters_random_sample"]["filters"]["raw"]["funnel"] == 4000
    assert result["filters_random_sample"]["filters"]["coverage_ge_min"]["funnel"] == 2800
    assert result["filters_random_sample"]["combinations_total"] == 4000


def test_merge_mismatched_filters_raises(tmp_path):
    shard1 = tmp_path / "shard1.json"
    shard1.write_text(
        json.dumps(
            {
                "filters": {
                    "raw": {"funnel": 100},
                    "filter_a": {
                        "name": "filter_a",
                        "type": "region",
                        "field": "X",
                        "op": "ge",
                        "value": 1,
                        "funnel": 50,
                        "pass": 50,
                    },
                },
                "combinations": {},
                "combinations_total": 100,
            }
        )
    )

    shard2 = tmp_path / "shard2.json"
    shard2.write_text(
        json.dumps(
            {
                "filters": {
                    "raw": {"funnel": 100},
                    "filter_b": {
                        "name": "filter_b",
                        "type": "region",
                        "field": "Y",
                        "op": "ge",
                        "value": 1,
                        "funnel": 50,
                        "pass": 50,
                    },
                },
                "combinations": {},
                "combinations_total": 100,
            }
        )
    )

    output = tmp_path / "merged.json"
    with pytest.raises(ValueError, match="Filter mismatch"):
        merge_snvfind_stats([shard1, shard2], output)


def test_merge_empty_input_raises(tmp_path):
    output = tmp_path / "merged.json"
    with pytest.raises(ValueError, match="No input files"):
        merge_snvfind_stats([], output)


def test_merge_three_shards(tmp_path):
    shards = []
    for i in range(3):
        data = {
            "filters": {
                "raw": {"funnel": 10000 * (i + 1)},
                "cov": {
                    "name": "cov",
                    "type": "region",
                    "field": "DP",
                    "op": "ge",
                    "value": 20,
                    "funnel": 8000 * (i + 1),
                    "pass": 8000 * (i + 1),
                },
            },
            "combinations": {"00": 2000 * (i + 1), "11": 8000 * (i + 1)},
            "combinations_total": 10000 * (i + 1),
        }
        path = tmp_path / f"shard{i}.json"
        path.write_text(json.dumps(data))
        shards.append(path)

    output = tmp_path / "merged.json"
    merge_snvfind_stats(shards, output)

    result = json.loads(output.read_text())
    # 10000 + 20000 + 30000 = 60000
    assert result["filters"]["raw"]["funnel"] == 60000
    # 8000 + 16000 + 24000 = 48000
    assert result["filters"]["cov"]["funnel"] == 48000
    assert result["combinations_total"] == 60000


def test_merge_combinations_with_new_keys(tmp_path):
    shard1 = tmp_path / "shard1.json"
    shard1.write_text(
        json.dumps(
            {
                "filters": {"raw": {"funnel": 100}},
                "combinations": {"00": 50, "01": 30, "10": 20},
                "combinations_total": 100,
            }
        )
    )

    shard2 = tmp_path / "shard2.json"
    shard2.write_text(
        json.dumps(
            {
                "filters": {"raw": {"funnel": 80}},
                "combinations": {"00": 40, "11": 40},
                "combinations_total": 80,
            }
        )
    )

    output = tmp_path / "merged.json"
    merge_snvfind_stats([shard1, shard2], output)

    result = json.loads(output.read_text())
    assert result["combinations"]["00"] == 90
    assert result["combinations"]["01"] == 30
    assert result["combinations"]["10"] == 20
    assert result["combinations"]["11"] == 40
    assert result["combinations_total"] == 180


def test_merge_trinuc_freq(tmp_path):
    shard1 = tmp_path / "shard1.csv"
    shard1.write_text(
        "A[A>C]A\t0.0121235971\t4383\t0.0091489761\t0.7546420497\n"
        "A[A>G]A\t0.0121235971\t5176\t0.0108042666\t0.8911766482\n"
        "A[A>T]A\t0.0121235971\t3736\t0.0077984428\t0.6432449686\n"
    )

    shard2 = tmp_path / "shard2.csv"
    shard2.write_text(
        "A[A>C]A\t0.0121235971\t4000\t0.0090000000\t0.7400000000\n"
        "A[A>G]A\t0.0121235971\t5000\t0.0110000000\t0.9000000000\n"
        "A[A>T]A\t0.0121235971\t3500\t0.0080000000\t0.6500000000\n"
    )

    output = tmp_path / "merged.csv"
    merge_trinuc_freq([shard1, shard2], output)

    lines = output.read_text().strip().split("\n")
    assert len(lines) == 3

    parts = lines[0].split("\t")
    assert parts[0] == "A[A>C]A"
    assert int(parts[2]) == 8383  # 4383 + 4000
    total = 8383 + 10176 + 7236  # sum of all counts
    expected_freq = 8383 / total
    assert abs(float(parts[3]) - expected_freq) < 1e-6


def test_merge_trinuc_freq_single_shard(tmp_path):
    shard1 = tmp_path / "shard1.csv"
    shard1.write_text("A[A>C]A\t0.0121235971\t4383\t0.0091489761\t0.7546420497\n")

    output = tmp_path / "merged.csv"
    merge_trinuc_freq([shard1], output)

    lines = output.read_text().strip().split("\n")
    assert len(lines) == 1
    parts = lines[0].split("\t")
    assert parts[0] == "A[A>C]A"
    assert int(parts[2]) == 4383
    assert abs(float(parts[3]) - 1.0) < 1e-6  # single trinuc = freq 1.0
