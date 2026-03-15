"""Build an ensemble manifest JSON for DNN k-fold inference.

Combines per-fold metadata JSONs with a split manifest (from combine_splits)
into the format expected by ``dnn_vcf_inference.py:load_ensemble_manifest()``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ugbio_core.logger import logger


def create_ensemble_manifest(
    fold_metadata_paths: list[str],
    split_manifest_path: str,
    output_path: str,
) -> Path:
    """Build and write an ensemble manifest JSON.

    Parameters
    ----------
    fold_metadata_paths
        Ordered list of per-fold metadata JSON paths (index = fold_idx).
    split_manifest_path
        Path to split_manifest.json from combine_splits.
    output_path
        Output path for ensemble_manifest.json.

    Returns
    -------
    Path
        Path to the written manifest.
    """
    split_manifest = json.loads(Path(split_manifest_path).read_text())
    chrom_to_fold = split_manifest.get("chrom_to_fold", {})
    k_folds = len(fold_metadata_paths)

    folds = []
    for i, meta_path in enumerate(fold_metadata_paths):
        folds.append({"fold_idx": i, "metadata_path": meta_path.strip()})

    # Extract quality_recalibration_table from fold 0 metadata if present
    meta_0 = json.loads(Path(folds[0]["metadata_path"]).read_text())
    recal_table = meta_0.get("quality_recalibration_table")

    manifest = {
        "k_folds": k_folds,
        "chrom_to_fold": chrom_to_fold,
        "folds": folds,
    }
    if recal_table:
        manifest["quality_recalibration_table"] = recal_table

    out = Path(output_path)
    out.write_text(json.dumps(manifest, indent=2))
    logger.info("Ensemble manifest written: %s (%d folds)", out, k_folds)
    return out


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build ensemble manifest JSON for DNN k-fold inference",
    )
    ap.add_argument(
        "--fold-metadata",
        nargs="+",
        required=True,
        help="Ordered fold metadata JSONs (post-recalibration)",
    )
    ap.add_argument(
        "--split-manifest",
        required=True,
        help="split_manifest.json from combine_splits",
    )
    ap.add_argument(
        "--output",
        required=True,
        help="Output ensemble_manifest.json path",
    )
    return ap.parse_args(argv)


def run(argv: list[str]) -> None:
    args = parse_args(argv[1:])
    create_ensemble_manifest(
        fold_metadata_paths=args.fold_metadata,
        split_manifest_path=args.split_manifest,
        output_path=args.output,
    )


def main() -> None:
    run(sys.argv)


if __name__ == "__main__":
    main()
