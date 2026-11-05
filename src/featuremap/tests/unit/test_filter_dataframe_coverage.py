"""filter_dataframe: any_of validation / evaluation, filter naming, preserve_field stats, and the CLI."""

import json
import sys

import polars as pl
import pytest
import ugbio_featuremap.filter_dataframe as fd


def _q(field, op, value=None, **extra):
    rule = {fd.KEY_FIELD: field, fd.KEY_OP: op, fd.KEY_TYPE: fd.TYPE_QUALITY, **extra}
    if value is not None:
        rule[fd.KEY_VALUE] = value
    return rule


def _any_of(*subs, **extra):
    return {fd.KEY_OP: fd.OP_ANY_OF, fd.KEY_TYPE: fd.TYPE_QUALITY, fd.KEY_FILTERS: list(subs), **extra}


# ---------------------------------------------------------------- any_of validation


@pytest.mark.parametrize(
    ("rule", "msg"),
    [
        (_any_of(), "non-empty 'filters' list"),
        ({fd.KEY_OP: fd.OP_ANY_OF, fd.KEY_TYPE: fd.TYPE_QUALITY, fd.KEY_FILTERS: "nf"}, "non-empty 'filters' list"),
        (_any_of("nf_gt_0"), "sub-rule 0 must be a dictionary"),
        (_any_of(_q("nf", "gt", 0), {fd.KEY_FIELD: "nr"}), "sub-rule 1 missing required 'op'"),
        (_any_of({fd.KEY_OP: fd.OP_ANY_NOT_NULL}), "requires a non-empty 'fields' list"),
        (_any_of({fd.KEY_OP: fd.OP_ANY_NOT_NULL, fd.KEY_FIELDS: []}), "requires a non-empty 'fields' list"),
        (_any_of({fd.KEY_OP: "gt", fd.KEY_VALUE: 0}), "sub-rule 0 missing required 'field'"),
        (_any_of({fd.KEY_FIELD: "nf", fd.KEY_OP: "gt"}), "must have either 'value'"),
        (_any_of({fd.KEY_FIELD: "nf", fd.KEY_OP: "bogus", fd.KEY_VALUE: 1}), "unsupported operator: bogus"),
        # nested composite: the inner any_of is validated recursively
        (_any_of(_q("nf", "gt", 0), _any_of()), "non-empty 'filters' list"),
        (_any_of(_any_of({fd.KEY_OP: "gt", fd.KEY_VALUE: 0})), "missing required 'field'"),
    ],
)
def test_any_of_validation_errors(rule, msg):
    with pytest.raises(ValueError, match=msg):
        fd.validate_filter_config({fd.KEY_FILTERS: [rule]})


def test_any_of_valid_nested_and_sub_rules_need_no_type():
    rule = _any_of(
        {fd.KEY_FIELD: "nf", fd.KEY_OP: "gt", fd.KEY_VALUE: 0},  # no "type" on sub-rules
        _any_of(
            {fd.KEY_OP: fd.OP_ANY_NOT_NULL, fd.KEY_FIELDS: ["dx", "po"]}, {fd.KEY_FIELD: "nr", fd.KEY_OP: "is_null"}
        ),
    )
    fd.validate_filter_config({fd.KEY_FILTERS: [rule]})  # does not raise


# ---------------------------------------------------------------- any_of evaluation


def test_any_of_or_semantics_with_nested_and_nulls():
    frame = pl.DataFrame(
        {
            "nf": [0, 2, 0, None, 0],
            "nr": [0, 0, 3, None, None],
            "dx": [None, None, None, None, 1],
        }
    )
    rule = _any_of(
        _q("nf", "gt", 0),
        _any_of(_q("nr", "gt", 0), {fd.KEY_OP: fd.OP_ANY_NOT_NULL, fd.KEY_FIELDS: ["dx"]}),
    )
    got = frame.select(fd._mask_for_rule(rule).alias("m"))["m"].to_list()
    # row 3: null under every sub-rule -> False (not null); row 4 passes via the nested any_not_null(dx)
    assert got == [False, True, True, False, True]


def test_mask_for_rule_without_comparison_target_raises():
    with pytest.raises(ValueError, match="missing a comparison target"):
        fd._mask_for_rule({fd.KEY_FIELD: "nf", fd.KEY_OP: "gt"})


# ---------------------------------------------------------------- naming


def test_unnamed_any_of_names_come_from_sub_rules():
    a = _any_of(_q("nf", "gt", 0), _q("nr", "gt", 0))
    b = _any_of(_q("nf", "ge", 2), _any_of(_q("dx", "eq", 1)))
    assert fd._get_filter_name(a) == "any_of(nf_gt|nr_gt)"
    assert fd._get_filter_name(b) == "any_of(nf_ge|any_of(dx_eq))"
    assert fd._get_filter_name({**a, fd.KEY_NAME: "consensus"}) == "consensus"
    assert fd._get_filter_name({fd.KEY_OP: fd.OP_ANY_NOT_NULL, fd.KEY_FIELDS: ["a", "b"]}) == "a_b_any_not_null"


def test_duplicate_filter_names_get_distinct_mask_columns():
    lf = pl.DataFrame({"score": [1, 3, 6, 9]}).lazy()
    # "cut" (explicit), an unnamed rule whose fallback name is score_gt twice, and an explicit "cut__2" that
    # collides with the generated suffix of the 2nd "cut".
    filters = [
        _q("score", "gt", 2, name="cut"),
        _q("score", "gt", 5),
        _q("score", "gt", 0),
        _q("score", "lt", 9, name="cut"),
        _q("score", "ge", 0, name="cut__3"),
    ]
    lf, cols = fd._create_filter_columns(lf, filters)
    assert len(set(cols)) == len(cols) == 5
    out = lf.collect()
    # every rule keeps its own mask (no overwrite), so the AND over all of them is correct
    assert out[cols[0]].to_list() == [False, True, True, True]
    assert out[cols[1]].to_list() == [False, False, True, True]
    assert out[cols[3]].to_list() == [True, True, True, False]
    final = fd._create_final_filter_column(lf, cols, None).collect()[fd.COL_FILTER_FINAL].to_list()
    assert final == [False, False, True, False]


# ---------------------------------------------------------------- preserve_field


def test_preserve_field_must_be_string():
    cfg = {fd.KEY_FILTERS: [_q("score", "ge", 0)], fd.KEY_DOWNSAMPLE: {fd.KEY_SIZE: 3, fd.KEY_PRESERVE_FIELD: 5}}
    with pytest.raises(ValueError, match="preserve_field' must be a string"):
        fd.validate_filter_config(cfg)


@pytest.fixture
def snv_indel_parquet(tmp_path):
    # 30 SNVs (X_IC null) + 4 indels; score >= 10 removes 10 SNVs and 1 indel
    frame = pl.DataFrame(
        {
            "score": list(range(30)) + [5, 20, 25, 28],
            "X_IC": [None] * 30 + ["ins", "del", "ins", "del"],
        }
    )
    path = tmp_path / "in.parquet"
    frame.write_parquet(path)
    return path


def _run(tmp_path, in_path, downsample):
    cfg = {fd.KEY_FILTERS: [_q("score", "ge", 10, name="min_score")], fd.KEY_DOWNSAMPLE: downsample}
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg))
    out, stats = tmp_path / "out.parquet", tmp_path / "stats.json"
    fd.filter_parquet(str(in_path), str(out), None, str(cfg_path), str(stats))
    return pl.read_parquet(out), fd.read_filtering_stats_json(stats)


def test_preserve_field_funnel_counts_preserved_rows(tmp_path, snv_indel_parquet):
    out, stats = _run(
        tmp_path,
        snv_indel_parquet,
        {fd.KEY_SIZE: 5, fd.KEY_METHOD: fd.METHOD_RANDOM, fd.KEY_SEED: 1, fd.KEY_PRESERVE_FIELD: "X_IC"},
    )
    # 20 SNVs + 3 indels pass; all 3 indels bypass the downsample, the SNVs are cut to the size 5
    assert out.filter(pl.col("X_IC").is_not_null()).height == 3
    assert out.filter(pl.col("X_IC").is_null()).height == 5
    rows = {f[fd.KEY_NAME]: f[fd.STAT_ROWS] for f in stats[fd.KEY_FILTERS]}
    assert rows == {"raw": 34, "min_score": 23, "downsample": 8}
    assert rows["downsample"] == out.height
    ds = stats[fd.KEY_FILTERS][-1]
    assert ds[fd.KEY_TYPE] == fd.TYPE_DOWNSAMPLE
    assert ds[fd.KEY_METHOD] == fd.METHOD_RANDOM
    assert ds[fd.KEY_SEED] == 1


def test_preserve_field_budget_larger_than_pool(tmp_path, snv_indel_parquet):
    out, stats = _run(tmp_path, snv_indel_parquet, {fd.KEY_SIZE: 100, fd.KEY_PRESERVE_FIELD: "X_IC"})
    assert out.height == 23
    assert stats[fd.KEY_FILTERS][-1][fd.STAT_ROWS] == 23


def test_missing_preserve_field_funnel_is_plain_min(tmp_path, snv_indel_parquet):
    out, stats = _run(tmp_path, snv_indel_parquet, {fd.KEY_SIZE: 5, fd.KEY_PRESERVE_FIELD: "NOT_A_COLUMN"})
    assert out.height == 5
    assert stats[fd.KEY_FILTERS][-1][fd.STAT_ROWS] == 5


# ---------------------------------------------------------------- CLI


def test_cli_end_to_end(tmp_path, snv_indel_parquet, monkeypatch):
    out, out_full, stats = tmp_path / "o.parquet", tmp_path / "full.parquet", tmp_path / "s.json"
    argv = ["filter_featuremap", "--in", str(snv_indel_parquet), "--out", str(out), "--out-full", str(out_full)]
    argv += ["--stats", str(stats), "--filter", "name=hi:field=score:op=ge:value=25:type=quality"]
    argv += ["--downsample", "head:2", "-v"]
    monkeypatch.setattr(sys, "argv", argv)
    level = fd.logger.level
    try:
        fd.main()
    finally:
        fd.logger.setLevel(level)
    kept = pl.read_parquet(out)
    assert kept["score"].to_list() == [25, 26]  # head:2 of the rows with score >= 25, in file order
    full = pl.read_parquet(out_full)
    assert full.height == 34
    assert f"{fd.COL_PREFIX_FILTER}hi" in full.columns
    rows = [f[fd.STAT_ROWS] for f in fd.read_filtering_stats_json(stats)[fd.KEY_FILTERS]]
    assert rows == [34, 7, 2]


@pytest.mark.parametrize(
    ("extra", "msg"),
    [
        (["--filter", "name=a:field=score:op=ge:value=1"], "No output file specified"),
        (["--out", "x.parquet"], "No filters specified"),
    ],
)
def test_cli_requires_output_and_filters(monkeypatch, extra, msg):
    monkeypatch.setattr(sys, "argv", ["filter_featuremap", "--in", "in.parquet", "--stats", "s.json", *extra])
    with pytest.raises(ValueError, match=msg):
        fd.main()
