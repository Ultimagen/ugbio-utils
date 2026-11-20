"""
Static per-reference annotation for the depth CNV caller.

For every 100 bp bin of the requested contigs: GC fraction (of ACGT bases), N fraction, and, for each given BED track,
the fraction of the bin the track covers (one column per track, named after it). Built once per reference; no sample
data. The depth caller reads the tracks named lowmap (low mappability) and segdup (segmental duplications); a track
that was not given counts as 0 everywhere.
"""

import argparse
import sys

import numpy as np
import pandas as pd
import pysam
from ugbio_core.logger import logger

BIN_SIZE = 100
DEFAULT_CONTIGS = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
CALLER_TRACKS = ("lowmap", "segdup")  # BED tracks the depth caller reads


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
    sequence: str, chrom: str, tracks: dict[str, pd.DataFrame] | None = None, bin_size: int = BIN_SIZE
) -> pd.DataFrame:
    """
    Annotation table of one contig sequence.

    Parameters
    ----------
    sequence : str
        Contig sequence.
    chrom : str
        Contig name.
    tracks : dict[str, pd.DataFrame], optional
        BED intervals (chrom, start, end) by track name.
    bin_size : int
        Bin size in bp.

    Returns
    -------
    pd.DataFrame
        Columns chrom, start, gc, n_frac and one covered-fraction column per track.
    """
    seq = np.frombuffer(sequence.upper().encode(), dtype=np.uint8)
    n_bins = len(seq) // bin_size + (len(seq) % bin_size > 0)
    padded = np.full(n_bins * bin_size, ord("N"), dtype=np.uint8)
    padded[: len(seq)] = seq
    bins = padded.reshape(n_bins, bin_size)
    gc = ((bins == ord("G")) | (bins == ord("C"))).sum(axis=1)
    acgt = np.isin(bins, np.frombuffer(b"ACGT", dtype=np.uint8)).sum(axis=1)
    ann = pd.DataFrame(
        {
            "chrom": chrom,
            "start": np.arange(n_bins, dtype=np.int64) * bin_size,
            "gc": np.where(acgt > 0, gc / np.maximum(acgt, 1), np.nan).astype(np.float32),
            "n_frac": (1 - acgt / bin_size).astype(np.float32),
        }
    )
    for name, bed in (tracks or {}).items():
        ann[name] = bed_fraction(bed, chrom, n_bins, bin_size)
    return ann


def read_annotation(annotation: str) -> pd.DataFrame:
    """
    Read an annotation parquet for the caller; a track the caller reads but the annotation lacks is 0 everywhere.

    Parameters
    ----------
    annotation : str
        Parquet from build_depth_annotation.

    Returns
    -------
    pd.DataFrame
        The annotation, chrom as str, with all CALLER_TRACKS columns.
    """
    ann = pd.read_parquet(annotation)
    ann["chrom"] = ann["chrom"].astype(str)
    for name in CALLER_TRACKS:
        if name not in ann:
            logger.warning(f"{annotation} has no {name} track; taken as 0")
            ann[name] = np.float32(0)
    return ann


def parse_bed_track(arg: str) -> tuple[str, str]:
    """'name=path' -> (name, path)."""
    name, sep, path = arg.partition("=")
    if not sep or not name or not path:
        raise argparse.ArgumentTypeError(f"expected NAME=BED, got {arg!r}")
    return name, path


def parse_region(region: str) -> tuple[str, int, int | None]:
    """'chr1:1-12950000' (1-based, inclusive) or 'chr9' -> (contig, 0-based start, end or None for the whole contig)."""
    if ":" not in region:
        return region, 0, None
    contig, span = region.rsplit(":", 1)
    start, end = span.replace(",", "").split("-")
    return contig, int(start) - 1, int(end)


def build_depth_annotation(
    reference_fasta: str,
    output_parquet: str,
    bed_tracks: dict[str, str] | None = None,
    contigs: list[str] | None = None,
    regions: list[str] | None = None,
) -> pd.DataFrame:
    """
    Build the 100 bp annotation for the given contigs and write it as parquet.

    Parameters
    ----------
    reference_fasta : str
        Indexed reference FASTA.
    output_parquet : str
        Output parquet path.
    bed_tracks : dict[str, str], optional
        BED file by track name; each becomes a column with the fraction of the bin it covers. The caller reads
        lowmap (e.g. GIAB GRCh38_lowmappabilityall.bed.gz) and segdup (e.g. GIAB GRCh38_segdups.bed.gz).
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
    tracks = {name: read_bed_intervals(path) for name, path in (bed_tracks or {}).items()}
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
            ann = contig_annotation(fasta.fetch(chrom), chrom, tracks)
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
    parser.add_argument(
        "--bed",
        nargs="+",
        type=parse_bed_track,
        default=[],
        metavar="NAME=BED",
        help="BED tracks, each written as the fraction of the bin it covers; the caller reads lowmap and segdup "
        "(a missing one counts as 0), e.g. lowmap=GRCh38_lowmappabilityall.bed.gz segdup=GRCh38_segdups.bed.gz",
    )
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
    build_depth_annotation(args.reference_fasta, args.output, dict(args.bed), args.contigs, args.regions)


def main():
    run(sys.argv)


if __name__ == "__main__":
    main()
