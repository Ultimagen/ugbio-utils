"""
Training pieces of known large CNVs for the depth CNV filter.

The depth caller gives one call per event, so a sample with a known multi-megabase CNV gives one training row. The
filter then sees almost no true events without breakpoint evidence, such as a mid-sized duplication in clean sequence
whose junction lies in repeats. cnv_known_event_pieces cuts windows inside each known event (lengths log-uniform in
[min_len, max_len], at 100 bp resolution), scores each with the depth caller's own segment statistics (GC-corrected
copy number of the 1 kb units, noise-weighted mean and z, segdup / unique fraction, LOW_CONF rule) and keeps the
windows the caller would call in the event's direction, without N bases and with at most max_segdup / max_lowmap
segmental-duplication / low-mappability bins. The windows are written as depth calls (call_depth_cnvs format).

Run the pipeline's evidence and snapping steps on them (analyze_cnv_breakpoint_reads, snap_cnv_breakpoints), and pass
the snapped VCF with the events BED to training_prep_depth_cnv_pipeline --known_events: every piece is a positive.
"""

import argparse
import sys

import numpy as np
import pandas as pd
from ugbio_cnv import depth_cnv_caller as dc
from ugbio_core.logger import logger


def unit_table(bins: pd.DataFrame, p: dc.Params) -> pd.DataFrame:
    """The depth caller's 1 kb units of normalized bins, with the difficult flag and the noise weight w."""
    units = dc.make_units(bins, p)
    units["difficult"] = (units["segdup"] >= p.difficult_frac) | (units["lowmap"] >= p.difficult_frac)
    units["w"] = 1.0 / units["difficult"].map(dc.unit_noise(units)) ** 2
    return units


def piece_features(  # noqa: PLR0913
    bins: pd.DataFrame, units: pd.DataFrame, chrom: str, start: int, end: int, svtype: str, p: dc.Params
) -> dict | None:
    """Depth-call features of the window [start, end), as call_depth_cnvs computes them for a segment."""
    u = units[(units["chrom"] == chrom) & (units["start"] >= start) & (units["end"] <= end)]
    b = bins[(bins["chrom"] == chrom) & (bins["start"] >= start) & (bins["end"] <= end)]
    if len(u) < p.min_units or not len(b):
        return None
    w, x = u["w"].to_numpy(), u["cn"].to_numpy()
    mean = float((w * x).sum() / w.sum())
    base = float(u["baseline"].iloc[0])
    dcn = mean - base
    z = dcn * np.sqrt(w.sum())
    called = abs(dcn) >= p.min_abs_dcn and abs(z) >= p.min_z and (dcn < 0) == (svtype == "DEL")
    confident = called and (u["difficult"].mean() < p.difficult_frac or abs(dcn) >= p.min_abs_dcn_difficult)
    return {
        "chrom": chrom,
        "start": int(start),
        "end": int(end),
        "direction": svtype,
        "low_conf": not confident,
        "cn_mean": mean,
        "baseline": base,
        "z": z,
        "n_merged": 1,
        "segdup": float(u["segdup"].mean()),
        "uniq_frac": float(u["uniq_frac"].mean()),
        "n_frac": float(b["n_frac"].mean()),
        "bin_segdup": float(b["segdup"].mean()),
        "bin_lowmap": float(b["lowmap"].mean()),
        "called": called,
    }


def cut_pieces(  # noqa: PLR0913
    bins: pd.DataFrame,
    units: pd.DataFrame,
    event: tuple[str, int, int, str],
    n_pieces: int,
    min_len: int,
    max_len: int,
    rng: np.random.Generator,
    p: dc.Params,
    max_segdup: float = 0.1,
    max_lowmap: float = 0.1,
    max_overlap: float = 0.0,
    max_tries: int = 20000,
) -> pd.DataFrame:
    """
    Up to n_pieces windows inside event (chrom, start, end, svtype) that the depth caller would call.

    Windows overlapping a kept window by more than max_overlap (reciprocal) are skipped; fewer than n_pieces are
    returned when max_tries random windows do not give enough.
    """
    chrom, ev_start, ev_end, svtype = event
    # the bins and units of the event only (window starts are rounded down to the bin, so one bin of margin)
    lo, hi = ev_start - dc.BIN_SIZE, ev_end + dc.BIN_SIZE
    bins = bins[(bins["chrom"] == chrom) & (bins["start"] >= lo) & (bins["end"] <= hi)]
    units = units[(units["chrom"] == chrom) & (units["start"] >= lo) & (units["end"] <= hi)]
    out: list[dict] = []
    for _ in range(max_tries):
        if len(out) == n_pieces:
            break
        length = int(np.exp(rng.uniform(np.log(min_len), np.log(min(max_len, ev_end - ev_start)))))
        start = int(rng.integers(ev_start, ev_end - length)) // dc.BIN_SIZE * dc.BIN_SIZE
        end = start + length // dc.BIN_SIZE * dc.BIN_SIZE
        if any(
            min(end, q["end"]) - max(start, q["start"]) > max_overlap * max(end - start, q["end"] - q["start"])
            for q in out
        ):
            continue
        f = piece_features(bins, units, chrom, start, end, svtype, p)
        if (
            f is None
            or not f["called"]
            or f["n_frac"] > 0
            or f["bin_segdup"] > max_segdup
            or f["bin_lowmap"] > max_lowmap
        ):
            continue
        out.append(f)
    if len(out) < n_pieces:
        logger.warning(f"{chrom}:{ev_start}-{ev_end} {svtype}: {len(out)} of {n_pieces} pieces found")
    return pd.DataFrame(out)


def known_event_pieces(  # noqa: PLR0913
    depth_mq0: str,
    depth_mq20: str,
    annotation: str,
    events_bed: str,
    sample_name: str,
    fasta_index: str,
    output_vcf: str,
    contigs: list[str] | None = None,
    haploid: set[str] | None = None,
    n_pieces: int = 15,
    min_len: int = 10_000,
    max_len: int = 100_000,
    max_segdup: float = 0.1,
    max_lowmap: float = 0.1,
    max_overlap: float = 0.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Cut training pieces of the known events (BED: chrom, start, end, DEL / DUP) and write them as depth calls."""
    p = dc.Params()
    haploid = set(haploid or ())
    bins = dc.normalize(dc.load_bins(depth_mq0, depth_mq20, annotation, contigs), haploid, p)
    units = unit_table(bins, p)
    events = pd.read_csv(
        events_bed, sep="\t", header=None, usecols=[0, 1, 2, 3], names=["chrom", "start", "end", "svtype"]
    )
    rng = np.random.default_rng(seed)
    pieces = pd.concat(
        [
            cut_pieces(
                bins,
                units,
                (str(e.chrom), int(e.start), int(e.end), str(e.svtype)),
                n_pieces,
                min_len,
                max_len,
                rng,
                p,
                max_segdup,
                max_lowmap,
                max_overlap,
            )
            for e in events.itertuples(index=False)
        ],
        ignore_index=True,
    )
    skipped = {"chrY"} if haploid else set()
    sex = dc.SexChromosomes(haploid=haploid, skipped=skipped, karyotype="given (known-event pieces)")
    dc.write_depth_calls(pieces, output_vcf, sample_name, fasta_index, sex)
    logger.info(f"{output_vcf}: {len(pieces)} pieces of {len(events)} events")
    return pieces


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="cnv_known_event_pieces", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--depth-mq0", required=True, help="mosdepth regions.bed.gz, all reads (with .csi)")
    parser.add_argument("--depth-mq20", required=True, help="mosdepth regions.bed.gz, MAPQ>=20 (with .csi)")
    parser.add_argument("--annotation", required=True, help="Reference annotation parquet")
    parser.add_argument("--events-bed", required=True, help="Known events: chrom, start (0-based), end, DEL / DUP")
    parser.add_argument("--sample-name", required=True, help="Sample name in the VCF")
    parser.add_argument("--fasta-index", required=True, help="Reference .fai (VCF contigs)")
    parser.add_argument("--output-vcf", required=True, help="Output .vcf.gz")
    parser.add_argument("--contigs", nargs="+", default=None, help="Contigs of the depth files (default: all)")
    parser.add_argument("--haploid-contigs", nargs="*", default=[], help="Haploid contigs of the sample (male chrX)")
    parser.add_argument("--n-pieces", type=int, default=15, help="Pieces per event")
    parser.add_argument("--min-len", type=int, default=10_000, help="Shortest piece")
    parser.add_argument("--max-len", type=int, default=100_000, help="Longest piece")
    parser.add_argument("--max-segdup", type=float, default=0.1, help="Max fraction of segdup bins in a piece")
    parser.add_argument("--max-lowmap", type=float, default=0.1, help="Max fraction of low-mappability bins")
    parser.add_argument("--max-overlap", type=float, default=0.0, help="Max reciprocal overlap between pieces")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    return parser.parse_args(argv)


def run(argv: list[str]):
    a = parse_args(argv[1:])
    known_event_pieces(
        a.depth_mq0,
        a.depth_mq20,
        a.annotation,
        a.events_bed,
        a.sample_name,
        a.fasta_index,
        a.output_vcf,
        a.contigs,
        set(a.haploid_contigs),
        a.n_pieces,
        a.min_len,
        a.max_len,
        a.max_segdup,
        a.max_lowmap,
        a.max_overlap,
        a.seed,
    )


def main():
    run(sys.argv)


if __name__ == "__main__":
    main()
