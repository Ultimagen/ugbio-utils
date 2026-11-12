"""
Static per-reference annotation for the depth CNV caller.

For every 100 bp bin of the requested contigs: GC fraction (of ACGT bases), N fraction, and the fraction of the bin
covered by low-mappability and segmental-duplication regions. Built once per reference; no sample data.
"""

import argparse
import sys

import numpy as np
import pandas as pd
import pysam
from ugbio_core.logger import logger

BIN_SIZE = 100
DEFAULT_CONTIGS = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]


def read_bed_intervals(bed_file: str) -> pd.DataFrame:
    """Read the first three columns of a (gzipped) BED file."""
    return pd.read_csv(
        bed_file, sep="\t", header=None, usecols=[0, 1, 2], names=["chrom", "start", "end"], comment="#", dtype={0: str}
    )


def bed_fraction(bed: pd.DataFrame, chrom: str, n_bins: int, bin_size: int = BIN_SIZE) -> np.ndarray:
    """
    Fraction of each bin covered by the BED intervals of one contig (overlapping intervals count once).

    Parameters
    ----------
    bed : pd.DataFrame
        Intervals with columns chrom, start, end.
    chrom : str
        Contig to compute.
    n_bins : int
        Number of bins in the contig.
    bin_size : int
        Bin size in bp.

    Returns
    -------
    np.ndarray
        Covered fraction per bin, float32.
    """
    sub = bed[bed["chrom"] == chrom]
    length = n_bins * bin_size
    diff = np.zeros(length + 1, dtype=np.int32)
    np.add.at(diff, np.clip(sub["start"].to_numpy(), 0, length), 1)
    np.add.at(diff, np.clip(sub["end"].to_numpy(), 0, length), -1)
    covered = (np.cumsum(diff[:-1]) > 0).astype(np.float32)
    return covered.reshape(n_bins, bin_size).mean(axis=1)


def contig_annotation(
    sequence: str, chrom: str, lowmap: pd.DataFrame, segdup: pd.DataFrame, bin_size: int = BIN_SIZE
) -> pd.DataFrame:
    """Annotation table (chrom, start, gc, n_frac, lowmap, segdup) for one contig sequence."""
    seq = np.frombuffer(sequence.upper().encode(), dtype=np.uint8)
    n_bins = len(seq) // bin_size + (len(seq) % bin_size > 0)
    padded = np.full(n_bins * bin_size, ord("N"), dtype=np.uint8)
    padded[: len(seq)] = seq
    bins = padded.reshape(n_bins, bin_size)
    gc = ((bins == ord("G")) | (bins == ord("C"))).sum(axis=1)
    acgt = np.isin(bins, np.frombuffer(b"ACGT", dtype=np.uint8)).sum(axis=1)
    return pd.DataFrame(
        {
            "chrom": chrom,
            "start": np.arange(n_bins, dtype=np.int64) * bin_size,
            "gc": np.where(acgt > 0, gc / np.maximum(acgt, 1), np.nan).astype(np.float32),
            "n_frac": (1 - acgt / bin_size).astype(np.float32),
            "lowmap": bed_fraction(lowmap, chrom, n_bins, bin_size),
            "segdup": bed_fraction(segdup, chrom, n_bins, bin_size),
        }
    )


def parse_region(region: str) -> tuple[str, int, int | None]:
    """'chr1:1-12950000' (1-based, inclusive) or 'chr9' -> (contig, 0-based start, end or None for the whole contig)."""
    if ":" not in region:
        return region, 0, None
    contig, span = region.rsplit(":", 1)
    start, end = span.replace(",", "").split("-")
    return contig, int(start) - 1, int(end)


def build_depth_annotation(  # noqa: PLR0913
    reference_fasta: str,
    lowmap_bed: str,
    segdup_bed: str,
    output_parquet: str,
    contigs: list[str] | None = None,
    regions: list[str] | None = None,
) -> pd.DataFrame:
    """
    Build the 100 bp annotation for the given contigs and write it as parquet.

    Parameters
    ----------
    reference_fasta : str
        Indexed reference FASTA.
    lowmap_bed : str
        Low-mappability regions (e.g. GIAB GRCh38_lowmappabilityall.bed.gz).
    segdup_bed : str
        Segmental duplications (e.g. GIAB GRCh38_segdups.bed.gz).
    output_parquet : str
        Output parquet path.
    contigs : list[str], optional
        Contigs to annotate (default chr1-22, chrX, chrY); contigs missing from the FASTA are skipped.
    regions : list[str], optional
        Restrict the annotation to these regions ('chr1:1-12950000', 1-based inclusive, or a whole contig 'chr9'),
        e.g. for a CRAM that covers only part of the genome. Only annotated bins are called.

    Returns
    -------
    pd.DataFrame
        The annotation table.
    """
    lowmap, segdup = read_bed_intervals(lowmap_bed), read_bed_intervals(segdup_bed)
    spans = {}
    for region in regions or []:
        contig, start, end = parse_region(region)
        spans.setdefault(contig, []).append((start, end))
    frames = []
    with pysam.FastaFile(reference_fasta) as fasta:
        for chrom in list(spans) if spans else contigs or DEFAULT_CONTIGS:
            if chrom not in fasta.references:
                logger.warning(f"{chrom} not in {reference_fasta}; skipped")
                continue
            ann = contig_annotation(fasta.fetch(chrom), chrom, lowmap, segdup)
            if chrom in spans:
                keep = np.zeros(len(ann), bool)
                for start, end in spans[chrom]:
                    keep |= (ann["start"].to_numpy() + BIN_SIZE > start) & (ann["start"].to_numpy() < (end or 1 << 62))
                ann = ann[keep]
            frames.append(ann)
            logger.info(f"annotated {chrom}: {len(frames[-1])} bins")
    annotation = pd.concat(frames, ignore_index=True)
    annotation["chrom"] = annotation["chrom"].astype("category")
    annotation.to_parquet(output_parquet, index=False)
    return annotation


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="build_depth_annotation", description=__doc__)
    parser.add_argument("--reference-fasta", required=True, help="Indexed reference FASTA")
    parser.add_argument("--lowmap-bed", required=True, help="Low-mappability regions BED")
    parser.add_argument("--segdup-bed", required=True, help="Segmental duplication regions BED")
    parser.add_argument("--output", required=True, help="Output parquet")
    parser.add_argument("--contigs", nargs="+", default=None, help="Contigs to annotate (default chr1-22, chrX, chrY)")
    parser.add_argument(
        "--regions",
        nargs="+",
        default=None,
        help="Only these regions ('chr1:1-12950000', 1-based inclusive, or a contig); overrides --contigs",
    )
    return parser.parse_args(argv)


def run(argv: list[str]):
    args = parse_args(argv[1:])
    build_depth_annotation(
        args.reference_fasta, args.lowmap_bed, args.segdup_bed, args.output, args.contigs, args.regions
    )


def main():
    run(sys.argv)


if __name__ == "__main__":
    main()
