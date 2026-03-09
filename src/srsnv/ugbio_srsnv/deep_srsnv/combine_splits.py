"""Combine positive/negative tensor caches and write per-fold train/val/test files.

Uses XGBoost-style fold_id assignment (via ``split_manifest.py``) so that
the DNN and XGBoost pipelines use identical splits. Each fold directory is
self-contained and can be shipped to a separate instance for parallel training.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from ugbio_core.logger import logger

from ugbio_srsnv.deep_srsnv.cram_to_tensors import PrepProfile, _cpu_times, _resource_rss_gb
from ugbio_srsnv.split_manifest import (
    SPLIT_MODE_CHROM_KFOLD,
    SPLIT_MODE_SINGLE_MODEL_CHROM_VAL,
    SPLIT_MODE_SINGLE_MODEL_READ_HASH,
    build_single_model_chrom_val_manifest,
    build_single_model_read_hash_manifest,
    build_split_manifest,
    load_split_manifest,
    save_split_manifest,
)

# ---------------------------------------------------------------------------
# Shard loading
# ---------------------------------------------------------------------------


def _load_shard_dir(shard_dir: str | Path) -> dict:
    """Load all shard pickle files from a cache directory and concatenate."""
    shard_path = Path(shard_dir)
    index_path = shard_path / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"No index.json found in {shard_path}")

    shard_files = sorted(shard_path.glob("shard_*.pkl"))
    if not shard_files:
        raise FileNotFoundError(f"No shard files found in {shard_path}")

    all_chunks: dict[str, list] = {
        "read_base_idx": [],
        "ref_base_idx": [],
        "t0_idx": [],
        "tm_idx": [],
        "st_idx": [],
        "et_idx": [],
        "x_num_pos": [],
        "x_num_const": [],
        "mask": [],
        "label": [],
        "chrom": [],
        "pos": [],
        "rn": [],
    }

    for sf in shard_files:
        with sf.open("rb") as f:
            chunk = pickle.load(f)  # noqa: S301
        for key in (
            "read_base_idx",
            "ref_base_idx",
            "t0_idx",
            "tm_idx",
            "st_idx",
            "et_idx",
            "x_num_pos",
            "x_num_const",
            "mask",
            "label",
        ):
            if key in chunk:
                all_chunks[key].append(chunk[key])
        if isinstance(chunk["chrom"], np.ndarray):
            all_chunks["chrom"].extend(chunk["chrom"].tolist())
        else:
            all_chunks["chrom"].extend(chunk["chrom"])
        all_chunks["pos"].append(np.asarray(chunk["pos"], dtype=np.int32))
        if isinstance(chunk["rn"], np.ndarray):
            all_chunks["rn"].extend(chunk["rn"].tolist())
        else:
            all_chunks["rn"].extend(chunk["rn"])

    result = {}
    for key in (
        "read_base_idx",
        "ref_base_idx",
        "t0_idx",
        "tm_idx",
        "st_idx",
        "et_idx",
        "x_num_pos",
        "x_num_const",
        "mask",
        "label",
    ):
        if all_chunks[key]:
            result[key] = torch.cat(all_chunks[key], dim=0)
    result["chrom"] = all_chunks["chrom"]
    result["pos"] = np.concatenate(all_chunks["pos"])
    result["rn"] = all_chunks["rn"]

    return result


def _save_fold_cache(cache: dict, indices: np.ndarray, path: Path) -> None:
    """Extract rows at ``indices`` from ``cache`` and save as a single file."""
    idx = indices.astype(np.int64)
    idx_t = torch.from_numpy(idx)
    subset = {}
    for key in (
        "read_base_idx",
        "ref_base_idx",
        "t0_idx",
        "x_num_pos",
        "x_num_const",
        "mask",
        "label",
        "tm_idx",
        "st_idx",
        "et_idx",
    ):
        if key in cache:
            subset[key] = cache[key][idx_t]
    subset["chrom"] = [cache["chrom"][i] for i in idx]
    subset["pos"] = cache["pos"][idx]
    subset["rn"] = [cache["rn"][i] for i in idx]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(subset, f, protocol=pickle.HIGHEST_PROTOCOL)


# ---------------------------------------------------------------------------
# Fold ID assignment (mirrors XGBoost logic)
# ---------------------------------------------------------------------------


def _assign_fold_ids(  # noqa: C901
    chroms: list[str],
    rns: list[str],
    split_manifest: dict,
) -> np.ndarray:
    """Assign fold_id to every row, mirroring XGBoost's split logic.

    Returns an array of float64 where NaN indicates holdout/test rows.
    """
    from ugbio_srsnv.split_manifest import (  # noqa: PLC0415
        assign_single_model_chrom_val_role,
        assign_single_model_read_hash_role,
    )

    n = len(chroms)
    fold_ids = np.full(n, np.nan, dtype=np.float64)
    split_mode = split_manifest.get("split_mode", SPLIT_MODE_CHROM_KFOLD)

    if split_mode == SPLIT_MODE_CHROM_KFOLD:
        chrom_to_fold = {str(k): int(v) for k, v in split_manifest["chrom_to_fold"].items()}
        for i in range(n):
            fid = chrom_to_fold.get(str(chroms[i]))
            if fid is not None:
                fold_ids[i] = float(fid)

    elif split_mode == SPLIT_MODE_SINGLE_MODEL_CHROM_VAL:
        for i in range(n):
            role = assign_single_model_chrom_val_role(str(chroms[i]), split_manifest)
            if role == "train":
                fold_ids[i] = 0.0
            elif role == "val":
                fold_ids[i] = 1.0
            # test -> stays NaN

    elif split_mode == SPLIT_MODE_SINGLE_MODEL_READ_HASH:
        for i in range(n):
            role = assign_single_model_read_hash_role(str(chroms[i]), str(rns[i]), split_manifest)
            if role == "train":
                fold_ids[i] = 0.0
            elif role == "val":
                fold_ids[i] = 1.0
            # test -> stays NaN

    return fold_ids


# ---------------------------------------------------------------------------
# Main combine + split
# ---------------------------------------------------------------------------


def combine_and_split(  # noqa: PLR0913, PLR0912, PLR0915, C901
    positive_cache_dir: str,
    negative_cache_dir: str,
    split_manifest_path: str | None = None,
    training_regions: str | None = None,
    k_folds: int = 3,
    holdout_chromosomes: list[str] | None = None,
    val_chromosomes: list[str] | None = None,
    single_model_split: bool = False,  # noqa: FBT001, FBT002
    random_seed: int = 42,
    val_fraction: float = 0.1,
    output_dir: str = "folds",
) -> dict:
    """Combine positive/negative caches and write per-fold train/val/test files.

    Parameters
    ----------
    positive_cache_dir, negative_cache_dir
        Directories produced by ``cram_to_tensor_cache``.
    split_manifest_path
        Path to a pre-built split manifest JSON. If ``None``, one is built
        from ``training_regions`` and the other split parameters.
    training_regions
        Path to training regions interval list (required when building a manifest).
    k_folds
        Number of cross-validation folds (chrom k-fold mode).
    holdout_chromosomes
        Chromosomes to hold out as the test set.
    val_chromosomes
        Chromosomes for validation (single-model chrom-val mode).
    single_model_split
        If True, use single-model split (hash or chrom-val).
    random_seed
        Seed for deterministic shuffling.
    val_fraction
        Fraction of reads for validation in read-hash mode.
    output_dir
        Root directory for fold output files.

    Returns
    -------
    dict
        Summary metadata including profiling.
    """
    cpu_u0, cpu_s0 = _cpu_times()
    t_wall_start = time.perf_counter()
    peak_rss = _resource_rss_gb()
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Phase 1: load caches
    t_load = time.perf_counter()
    logger.info("Loading positive cache from %s", positive_cache_dir)
    pos_cache = _load_shard_dir(positive_cache_dir)
    n_pos = int(pos_cache["label"].shape[0])
    logger.info("Loading negative cache from %s", negative_cache_dir)
    neg_cache = _load_shard_dir(negative_cache_dir)
    n_neg = int(neg_cache["label"].shape[0])
    load_seconds = round(time.perf_counter() - t_load, 4)
    logger.info("Loaded %d positive + %d negative = %d total rows in %.1fs", n_pos, n_neg, n_pos + n_neg, load_seconds)
    peak_rss = max(peak_rss, _resource_rss_gb())

    # Concatenate
    combined: dict = {}
    for key in (
        "read_base_idx",
        "ref_base_idx",
        "t0_idx",
        "tm_idx",
        "st_idx",
        "et_idx",
        "x_num_pos",
        "x_num_const",
        "mask",
        "label",
    ):
        if key in pos_cache and key in neg_cache:
            combined[key] = torch.cat([pos_cache[key], neg_cache[key]], dim=0)
    combined["chrom"] = pos_cache["chrom"] + neg_cache["chrom"]
    combined["pos"] = np.concatenate([pos_cache["pos"], neg_cache["pos"]])
    combined["rn"] = pos_cache["rn"] + neg_cache["rn"]
    del pos_cache, neg_cache

    n_total = int(combined["label"].shape[0])

    # Phase 2: build or load split manifest
    t_fold = time.perf_counter()
    if split_manifest_path:
        split_manifest = load_split_manifest(split_manifest_path)
        logger.info("Loaded split manifest from %s", split_manifest_path)
    elif training_regions is None:
        raise ValueError("Either --split-manifest or --training-regions must be provided")
    elif single_model_split and val_chromosomes:
        split_manifest = build_single_model_chrom_val_manifest(
            training_regions=training_regions,
            holdout_chromosomes=holdout_chromosomes or [],
            val_chromosomes=val_chromosomes,
        )
    elif single_model_split:
        split_manifest = build_single_model_read_hash_manifest(
            training_regions=training_regions,
            random_seed=random_seed,
            holdout_chromosomes=holdout_chromosomes or [],
            val_fraction=val_fraction,
        )
    else:
        split_manifest = build_split_manifest(
            training_regions=training_regions,
            k_folds=k_folds,
            random_seed=random_seed,
            holdout_chromosomes=holdout_chromosomes,
        )

    save_split_manifest(split_manifest, out_root / "split_manifest.json")

    fold_id_arr = _assign_fold_ids(combined["chrom"], combined["rn"], split_manifest)
    fold_assign_seconds = round(time.perf_counter() - t_fold, 4)

    split_mode = split_manifest.get("split_mode", SPLIT_MODE_CHROM_KFOLD)
    if split_mode == SPLIT_MODE_CHROM_KFOLD:
        effective_k = int(split_manifest.get("k_folds", k_folds))
    else:
        effective_k = 1

    logger.info(
        "Fold assignment: mode=%s k=%d holdout=%s (%.1fs)",
        split_mode,
        effective_k,
        ",".join(split_manifest.get("test_chromosomes", [])),
        fold_assign_seconds,
    )

    # Phase 3: shuffle and write per-fold files
    t_shuffle = time.perf_counter()
    rng = np.random.default_rng(random_seed)

    fold_summary: list[dict] = []
    test_path_fold0: Path | None = None

    for fold_idx in range(effective_k):
        if split_mode == SPLIT_MODE_CHROM_KFOLD:
            val_mask = fold_id_arr == fold_idx
            train_mask = (~val_mask) & (~np.isnan(fold_id_arr))
        else:
            val_mask = fold_id_arr == 1.0
            train_mask = fold_id_arr == 0.0
        test_mask = np.isnan(fold_id_arr)

        train_indices = rng.permutation(np.where(train_mask)[0])
        val_indices = rng.permutation(np.where(val_mask)[0])
        test_indices = rng.permutation(np.where(test_mask)[0])

        fold_dir = out_root / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        _save_fold_cache(combined, train_indices, fold_dir / "train.pt")
        _save_fold_cache(combined, val_indices, fold_dir / "val.pt")

        if fold_idx == 0:
            _save_fold_cache(combined, test_indices, fold_dir / "test.pt")
            test_path_fold0 = fold_dir / "test.pt"
        else:
            test_target = fold_dir / "test.pt"
            if test_path_fold0 and not test_target.exists():
                try:
                    test_target.symlink_to(os.path.relpath(test_path_fold0, fold_dir))
                except OSError:
                    _save_fold_cache(combined, test_indices, test_target)

        train_labels = combined["label"][torch.from_numpy(train_indices.astype(np.int64))].numpy()
        val_labels = combined["label"][torch.from_numpy(val_indices.astype(np.int64))].numpy()
        test_labels = combined["label"][torch.from_numpy(test_indices.astype(np.int64))].numpy()

        fold_info = {
            "fold": fold_idx,
            "train_rows": int(len(train_indices)),
            "val_rows": int(len(val_indices)),
            "test_rows": int(len(test_indices)),
            "train_positives": int(np.sum(train_labels)),
            "val_positives": int(np.sum(val_labels)),
            "test_positives": int(np.sum(test_labels)),
        }
        fold_summary.append(fold_info)
        logger.info(
            "Fold %d: train=%d (pos=%d) val=%d (pos=%d) test=%d (pos=%d)",
            fold_idx,
            fold_info["train_rows"],
            fold_info["train_positives"],
            fold_info["val_rows"],
            fold_info["val_positives"],
            fold_info["test_rows"],
            fold_info["test_positives"],
        )

    shuffle_write_seconds = round(time.perf_counter() - t_shuffle, 4)
    peak_rss = max(peak_rss, _resource_rss_gb())

    # Profiling
    cpu_u1, cpu_s1 = _cpu_times()
    wall = time.perf_counter() - t_wall_start

    total_bytes = sum(f.stat().st_size for f in out_root.rglob("*") if f.is_file() and f.name != "index.json")

    profile = PrepProfile(
        wall_seconds=round(wall, 3),
        cpu_user_seconds=round(cpu_u1 - cpu_u0, 3),
        cpu_system_seconds=round(cpu_s1 - cpu_s0, 3),
        cpu_utilization=round(((cpu_u1 - cpu_u0) + (cpu_s1 - cpu_s0)) / max(wall, 0.001), 4),
        peak_rss_gb=round(peak_rss, 3),
        total_input_rows=n_total,
        total_output_rows=n_total,
        rows_per_second=round(n_total / max(wall, 0.001), 1),
        missing_rows=0,
        bytes_written=total_bytes,
        phase_seconds={
            "load": load_seconds,
            "fold_assign": fold_assign_seconds,
            "shuffle_and_write": shuffle_write_seconds,
        },
    )

    index = {
        "split_mode": split_mode,
        "effective_k": effective_k,
        "random_seed": random_seed,
        "total_rows": n_total,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "fold_summary": fold_summary,
        "profile": asdict(profile),
    }
    index_path = out_root / "index.json"
    index_path.write_text(json.dumps(index, indent=2))
    profile.log_summary("combine_and_split")

    return index


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Combine pos/neg tensor caches and write per-fold splits")
    ap.add_argument("--positive", required=True, help="Positive tensor cache directory")
    ap.add_argument("--negative", required=True, help="Negative tensor cache directory")
    ap.add_argument("--output", required=True, help="Output root directory for fold files")
    ap.add_argument("--split-manifest", default=None, help="Pre-built split manifest JSON")
    ap.add_argument("--training-regions", default=None, help="Training regions interval list")
    ap.add_argument("--k-folds", type=int, default=3)
    ap.add_argument("--holdout-chromosomes", default="chr21,chr22")
    ap.add_argument("--val-chromosomes", default=None)
    ap.add_argument("--single-model-split", action="store_true")
    ap.add_argument("--random-seed", type=int, default=42)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    return ap.parse_args(argv)


def run(argv: list[str] | None = None) -> dict:
    args = _parse_args(argv)
    holdout = (
        [c.strip() for c in args.holdout_chromosomes.split(",") if c.strip()] if args.holdout_chromosomes else None
    )
    val_chroms = [c.strip() for c in args.val_chromosomes.split(",") if c.strip()] if args.val_chromosomes else None
    return combine_and_split(
        positive_cache_dir=args.positive,
        negative_cache_dir=args.negative,
        split_manifest_path=args.split_manifest,
        training_regions=args.training_regions,
        k_folds=args.k_folds,
        holdout_chromosomes=holdout,
        val_chromosomes=val_chroms,
        single_model_split=args.single_model_split,
        random_seed=args.random_seed,
        val_fraction=args.val_fraction,
        output_dir=args.output,
    )


def main() -> None:
    run()


if __name__ == "__main__":
    main()
