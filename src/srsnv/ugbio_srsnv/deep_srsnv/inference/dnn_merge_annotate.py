"""Merge per-fold DNN predictions and annotate the featuremap VCF.

Standalone module that does NOT import torch/pycuda/tensorrt, so it can
run on CPU-only instances (no GPU required for merge+annotate).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pyarrow.compute as pc
import pysam
from pyarrow import parquet as pq
from ugbio_core.logger import logger
from ugbio_core.vcfbed.variant_annotation import VcfAnnotator

from ugbio_srsnv.srsnv_utils import MAX_PHRED


def _parse_merge_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Merge per-fold DNN predictions and annotate VCF",
    )
    ap.add_argument("--featuremap-vcf", required=True, help="Input featuremap VCF to annotate")
    ap.add_argument("--fold-predictions", nargs="+", required=True, help="Per-fold prediction parquet files")
    ap.add_argument("--fold-metadata", required=True, help="Fold 0 metadata JSON (for quality recalibration LUT)")
    ap.add_argument("--output", required=True, help="Output annotated VCF path")
    ap.add_argument("--low-qual-threshold", type=float, default=40.0, help="SNVQ threshold for PASS filter")
    ap.add_argument("--process-number", type=int, default=-2, help="Parallel processes for VCF annotation")
    return ap.parse_args(argv)


def _load_contig_predictions(
    contig: str,
    pred_paths: list[str],
) -> tuple[np.ndarray, list[str], list[float]] | None:
    """Load and sort predictions for one contig from all fold parquet files."""
    all_pos: list[np.ndarray] = []
    all_rn: list[str] = []
    all_prob: list[np.ndarray] = []
    for path in pred_paths:
        table = pq.read_table(path, filters=[("chrom", "==", contig)])
        if table.num_rows == 0:
            continue
        all_pos.append(table.column("pos").to_numpy())
        all_rn.extend(table.column("rn").to_pylist())
        all_prob.append(table.column("prob").to_numpy())

    if not all_pos:
        return None

    pos_arr = np.concatenate(all_pos)
    prob_arr = np.concatenate(all_prob)
    del all_pos, all_prob

    order = np.argsort(pos_arr, kind="stable")
    pos_arr = pos_arr[order]
    prob_list = prob_arr[order].tolist()
    rn_list = [all_rn[i] for i in order]
    return pos_arr, rn_list, prob_list


def _build_snvq_lut(
    quality_lut_x: list[float] | None,
    quality_lut_y: list[float] | None,
    max_phred: float,
) -> dict[int, float]:
    """Pre-compute MQUAL->SNVQ lookup dict keyed by int(mqual*100)."""
    if quality_lut_x is None or quality_lut_y is None:
        return {}
    lut_x = np.asarray(quality_lut_x, dtype=np.float64)
    lut_y = np.asarray(quality_lut_y, dtype=np.float64)
    right = float(lut_y[-1])
    snvq_lut: dict[int, float] = {}
    for mq_cent in range(int(max_phred * 100) + 1):
        sq = float(np.interp(mq_cent / 100.0, lut_x, lut_y, left=0.0, right=right))
        snvq_lut[mq_cent] = round(sq, 2)
    return snvq_lut


def _annotate_record(
    rec: pysam.VariantRecord,
    preds_at_pos: dict[str, float],
    snvq_lut: dict[int, float],
    max_phred: float,
    min_prob_error: float,
    low_qual_threshold: float,
) -> None:
    """Annotate a single VCF record with MQUAL/SNVQ from matched predictions."""
    try:
        rn_val = rec.samples[0].get("RN")
    except (KeyError, IndexError):
        rn_val = None

    if not rn_val or not preds_at_pos:
        return

    rns = rn_val if isinstance(rn_val, tuple) else (rn_val,)

    mquals = []
    for rn in rns:
        prob = preds_at_pos.get(rn if isinstance(rn, str) else str(rn))
        prob = prob if prob is not None and prob > 0 else 0.0
        if prob > 0:
            mq = -10.0 * math.log10(max(1.0 - prob, min_prob_error))
            mq = min(mq, max_phred)
        else:
            mq = 0.0
        mquals.append(round(mq, 2))

    if snvq_lut:
        snvqs = [snvq_lut.get(int(mq * 100), mq) for mq in mquals]
    else:
        snvqs = list(mquals)

    rec.samples[0]["MQUAL"] = tuple(mquals)
    rec.samples[0]["SNVQ"] = tuple(snvqs)

    max_snvq = max(snvqs)
    rec.qual = max_snvq
    if max_snvq >= low_qual_threshold:
        rec.filter.add("PASS")
    else:
        rec.filter.add("LowQual")


def _copy_contig_records(vcf_in: str, vcf_out: str, contig: str) -> str:
    """Copy contig records through without annotation."""
    with pysam.VariantFile(vcf_in) as inp:
        try:
            inp.header.add_line('##FILTER=<ID=LowQual,Description="SNVQ below quality threshold">')
        except ValueError:
            pass
        with pysam.VariantFile(vcf_out, "w", header=inp.header) as out:
            for rec in inp.fetch(contig):
                out.write(rec)
    pysam.tabix_index(vcf_out, preset="vcf", force=True)
    return vcf_out


def _merge_contig_worker(
    contig: str,
    vcf_in: str,
    vcf_out: str,
    pred_paths: list[str],
    quality_lut_x: list[float] | None,
    quality_lut_y: list[float] | None,
    low_qual_threshold: float,
) -> str:
    """Annotate one contig using streaming merge-join."""
    loaded = _load_contig_predictions(contig, pred_paths)
    if loaded is None:
        return _copy_contig_records(vcf_in, vcf_out, contig)

    pos_arr, rn_list, prob_list = loaded
    n_preds = len(pos_arr)
    max_phred = float(MAX_PHRED)
    min_prob_error = 10.0 ** (-max_phred / 10.0)
    snvq_lut = _build_snvq_lut(quality_lut_x, quality_lut_y, max_phred)

    cursor = 0
    with pysam.VariantFile(vcf_in) as inp:
        try:
            inp.header.add_line('##FILTER=<ID=LowQual,Description="SNVQ below quality threshold">')
        except ValueError:
            pass

        with pysam.VariantFile(vcf_out, "w", header=inp.header) as out:
            for rec in inp.fetch(contig):
                pos = rec.pos

                while cursor < n_preds and pos_arr[cursor] < pos:
                    cursor += 1

                preds_at_pos: dict[str, float] = {}
                i = cursor
                while i < n_preds and pos_arr[i] == pos:
                    preds_at_pos[rn_list[i]] = prob_list[i]
                    i += 1

                _annotate_record(rec, preds_at_pos, snvq_lut, max_phred, min_prob_error, low_qual_threshold)
                out.write(rec)

    pysam.tabix_index(vcf_out, preset="vcf", force=True)
    return vcf_out


def run_merge(argv: list[str] | None = None) -> None:
    """Merge per-fold predictions and annotate the featuremap VCF."""
    if argv is None:
        argv = sys.argv[1:]
    args = _parse_merge_args(argv)

    t0 = time.perf_counter()

    with open(args.fold_metadata) as f:
        metadata = json.load(f)
    lut = metadata.get("quality_recalibration_table")
    min_lut_entries = 2
    quality_lut_x = lut[0] if lut and len(lut) >= min_lut_entries else None
    quality_lut_y = lut[1] if lut and len(lut) >= min_lut_entries else None

    contig_counts: dict[str, int] = {}
    total_preds = 0
    for pred_path in args.fold_predictions:
        chrom_col = pq.read_table(pred_path, columns=["chrom"]).column("chrom")
        for entry in pc.value_counts(chrom_col).to_pylist():
            c = str(entry["values"])
            contig_counts[c] = contig_counts.get(c, 0) + entry["counts"]
            total_preds += entry["counts"]

    logger.info(
        "Found %d predictions across %d contigs from %d fold files",
        total_preds,
        len(contig_counts),
        len(args.fold_predictions),
    )

    out_dir = os.path.dirname(args.output) or "."

    with pysam.VariantFile(args.featuremap_vcf) as input_vcf:
        contig_tasks = []
        skipped = 0
        for contig in input_vcf.header.contigs:
            if contig not in contig_counts:
                skipped += 1
                continue
            try:
                next(input_vcf.fetch(contig))
            except StopIteration:
                skipped += 1
                continue
            contig_tasks.append(contig)

    if skipped:
        logger.info("Skipped %d empty contigs", skipped)

    contig_tasks.sort(key=lambda c: contig_counts.get(c, 0), reverse=True)

    max_cpus = args.process_number if args.process_number > 0 else (os.cpu_count() or 4)
    num_workers = min(max_cpus, max(1, len(contig_tasks)))
    logger.info(
        "Annotating %d contigs with %d processes (largest: %s with %d predictions)",
        len(contig_tasks),
        num_workers,
        contig_tasks[0] if contig_tasks else "?",
        contig_counts.get(contig_tasks[0], 0) if contig_tasks else 0,
    )

    tmp_output_paths: list[str] = []
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = {}
        for contig in contig_tasks:
            out_path = os.path.join(out_dir, contig + ".vcf.gz")
            fut = pool.submit(
                _merge_contig_worker,
                contig=contig,
                vcf_in=args.featuremap_vcf,
                vcf_out=out_path,
                pred_paths=list(args.fold_predictions),
                quality_lut_x=quality_lut_x,
                quality_lut_y=quality_lut_y,
                low_qual_threshold=args.low_qual_threshold,
            )
            futures[fut] = (contig, out_path)

        for fut, (contig, out_path) in futures.items():
            fut.result()
            tmp_output_paths.append(out_path)
            logger.info("Annotated contig %s", contig)

    VcfAnnotator.merge_temp_files(tmp_output_paths, args.output, process_number=1)

    elapsed = time.perf_counter() - t0
    logger.info("Merge + annotate complete: %d predictions -> %s in %.1fs", total_preds, args.output, elapsed)


def main() -> None:
    run_merge()


if __name__ == "__main__":
    main()
