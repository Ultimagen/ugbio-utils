from __future__ import annotations

import argparse
import json
import logging
from typing import Any

import polars as pl

log = logging.getLogger(__name__)

# ───────────────────────────── utilities ──────────────────────────────────
_OPS = {
    "eq": lambda c, v: c == v,
    "ne": lambda c, v: c != v,
    "lt": lambda c, v: c < v,
    "le": lambda c, v: c <= v,
    "gt": lambda c, v: c > v,
    "ge": lambda c, v: c >= v,
    "in": lambda c, v: c.is_in(v),
    "not_in": lambda c, v: ~c.is_in(v),
    "between": lambda c, v: (c >= v[0]) & (c <= v[1]),
    "regex": lambda c, v: c.str.contains(v),
}


def _mask_for_rule(featuremap_dataframe: pl.DataFrame, rule: dict[str, Any]) -> pl.Series:
    """Return a boolean mask for a single rule."""
    field = rule["field"]
    op = rule["op"]
    if op not in _OPS:
        raise ValueError(f"Unsupported op: {op}")
    if "value_field" in rule:
        rhs = featuremap_dataframe[rule["value_field"]]
    else:
        rhs = rule.get("value") or rule.get("values")
    return _OPS[op](featuremap_dataframe[field], rhs)


def _apply_filters(
    featuremap_dataframe: pl.DataFrame,
    filters: list[dict[str, Any]],
) -> tuple[pl.DataFrame, list[tuple[str, int]]]:
    """Apply filters sequentially, store a boolean column per filter."""
    stats: list[tuple[str, int]] = [("raw", featuremap_dataframe.height)]

    for rule in filters:
        name = rule.get("name") or f"{rule['field']}_{rule['op']}"
        mask = _mask_for_rule(featuremap_dataframe, rule)
        featuremap_dataframe = featuremap_dataframe.with_columns(mask.alias(f"__f_{name}"))
        featuremap_dataframe = featuremap_dataframe.filter(mask)
        stats.append((name, featuremap_dataframe.height))
    return featuremap_dataframe, stats


def _downsample(featuremap_dataframe: pl.DataFrame, cfg: dict[str, Any]) -> tuple[pl.DataFrame, tuple[str, int]]:
    """Down-sample (random or stratified-by-CHROM)."""
    size = cfg["size"]
    if featuremap_dataframe.height <= size:
        return featuremap_dataframe, ("downsample", featuremap_dataframe.height)

    method = cfg.get("method", "random")
    if method == "head":
        featuremap_dataframe = featuremap_dataframe.head(size)
    elif method == "random":
        featuremap_dataframe = featuremap_dataframe.sample(n=size, shuffle=True, seed=cfg.get("seed", 0))
    elif method == "stratified":
        raise NotImplementedError()
    else:
        raise ValueError(f"Unknown down-sample method {method}")
    return featuremap_dataframe, ("downsample", featuremap_dataframe.height)


# ───────────────────────────── main driver ────────────────────────────────
def filter_parquet(in_path: str, out_path: str, cfg_path: str, stats_path: str) -> None:
    featuremap_dataframe = pl.read_parquet(in_path)
    cfg = json.load(open(cfg_path))

    featuremap_dataframe, funnel = _apply_filters(featuremap_dataframe, cfg["filters"])

    if "downsample" in cfg:
        featuremap_dataframe, ds_stat = _downsample(featuremap_dataframe, cfg["downsample"])
        funnel.append(ds_stat)

    # --- single-effect ----------------------------------------------------
    single_effect: dict[str, int] = {}
    for rule in cfg["filters"]:
        name = rule.get("name") or f"{rule['field']}_{rule['op']}"
        single_effect[name] = _mask_for_rule(pl.read_parquet(in_path), rule).sum()

    # --- combinations -----------------------------------------------------
    combo_cols = [(f"__f_{r.get('name') or r['field']}_{r['op']}") for r in cfg["filters"]]
    combos = (
        featuremap_dataframe.groupby(combo_cols)
        .agg(count=pl.count())
        .with_columns(pl.concat_str(combo_cols).alias("pattern"))
        .select(["pattern", "count"])
        .to_dicts()
    )

    featuremap_dataframe.write_parquet(out_path)

    json.dump(
        {
            "filters": [{"name": n, "rows": r} for n, r in funnel],
            "single_effect": single_effect,
            "combinations": combos,
        },
        open(stats_path, "w"),
        indent=2,
    )
    log.info("Wrote %s (%d rows) and %s", out_path, featuremap_dataframe.height, stats_path)


def _build_cli() -> argparse.ArgumentParser:  # ...existing code trimmed...
    ap = argparse.ArgumentParser(description="Filter / down-sample featuremap Parquet")
    ap.add_argument("--in", dest="inp", required=True, help="input parquet")
    ap.add_argument("--out", required=True, help="output parquet")
    ap.add_argument("--config", required=True, help="JSON with filters + downsample")
    ap.add_argument("--stats", required=True, help="output JSON with statistics")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap


def main() -> None:
    args = _build_cli().parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    filter_parquet(args.inp, args.out, args.config, args.stats)


if __name__ == "__main__":
    main()
