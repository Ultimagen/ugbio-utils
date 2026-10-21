#!/usr/bin/env python3
"""
Mark the consensus reads of duplex families with ``dx:i:1``.

A consensus read carries ``CS`` (the id shared by the two strands of one original molecule) and ``po``
(the orientation of the pair it came from: F1R2 or F2R1). The two orientations of a molecule are its two
strands, so a CS family is duplex when its consensus reads include *both* F1R2 and F2R1.

``DS`` from the sorter does not say this. It is the size of a duplicate set of consensus reads, so it is
2 for a duplex pair and also for two consensus reads of the *same* orientation, for example when a symmetric
UMI (u3 == u5) put both orientations into one MI upstream. ``dx`` looks at ``po`` and cannot be fooled that way.

Every read of a duplex CS family gets ``dx=1``, whichever of its reads it is, so a downstream filter needs
no second look at the partner. Reads of any other CS family get no ``dx``, and a ``dx`` that was already on a
read is replaced. Reads without ``CS`` are left alone.

The input must be coordinate-sorted. The two orientations of a molecule start within one fragment length of
each other, so the file is streamed with a window of ``--pad`` bp: a read is written once no later read can
belong to its family. A family whose orientations lie further apart than ``--pad`` is not recognised as duplex.

Usage:
    mark_duplex_consensus in.cram out.cram -T ref.fa
"""

import argparse
import logging
from collections import deque

import pysam

DEFAULT_PAD = 2_000
F1R2, F2R1 = "F1R2", "F2R1"
_ORIENTATION_BIT = {F1R2: 1, F2R1: 2}
_BOTH = 3


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Tag consensus reads of duplex (both-orientation) CS families with dx=1"
    )
    parser.add_argument("input", help="Coordinate-sorted BAM/CRAM with CS and po tags")
    parser.add_argument("output", help="Output BAM/CRAM")
    parser.add_argument("--reference", "-T", help="Reference FASTA (required for CRAM I/O)")
    parser.add_argument("--tag", default="dx", help="Tag to write (default dx)")
    parser.add_argument("--cs-tag", default="CS", help="Tag that identifies the molecule (default CS)")
    parser.add_argument("--po-tag", default="po", help="Tag holding the pair orientation (default po)")
    parser.add_argument(
        "--pad",
        type=int,
        default=DEFAULT_PAD,
        metavar="BP",
        help=f"How far apart the orientations of one family may start (default {DEFAULT_PAD})",
    )
    parser.add_argument("--threads", type=int, default=1, help="htslib compression threads (default 1)")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args(argv)


def _strip(read, tag: str) -> None:
    if read.has_tag(tag):
        read.tags = [(k, v) for k, v in read.tags if k != tag]


def mark_duplex(  # noqa: C901
    input_path: str,
    output_path: str,
    open_kwargs: dict,
    *,
    tag: str = "dx",
    cs_tag: str = "CS",
    po_tag: str = "po",
    pad: int = DEFAULT_PAD,
    threads: int = 1,
) -> dict[str, int]:
    """Stream `input_path` to `output_path`, stamping `tag`. Returns read and family counters."""
    write_mode = "wc" if output_path.endswith(".cram") else "wb"
    counters = {"reads": 0, "reads_with_cs": 0, "reads_dx1": 0, "cs_families": 0, "duplex_families": 0}

    # cs -> [orientation mask, reads still in the window]; dropped when its last read has been written
    state: dict[str, list[int]] = {}
    window: deque = deque()

    def finish(read, cs: str | None, out) -> None:
        _strip(read, tag)
        if cs is not None:
            counters["reads_with_cs"] += 1
            duplex = state[cs][0] == _BOTH
            if duplex:
                read.set_tag(tag, 1, value_type="i")
                counters["reads_dx1"] += 1
            state[cs][1] -= 1
            if state[cs][1] == 0:
                counters["cs_families"] += 1
                counters["duplex_families"] += duplex
                del state[cs]
        out.write(read)

    with pysam.AlignmentFile(input_path, "r", threads=threads, **open_kwargs) as bam:
        write_kwargs = dict(open_kwargs) if write_mode == "wc" else {}
        with pysam.AlignmentFile(output_path, write_mode, header=bam.header, threads=threads, **write_kwargs) as out:
            for read in bam:
                counters["reads"] += 1
                # written before this read is counted: a read further than pad behind it cannot be its partner
                while window and (
                    window[0][0].reference_id != read.reference_id
                    or window[0][0].reference_start < read.reference_start - pad
                ):
                    finish(*window.popleft(), out)
                cs = None
                if not (read.is_secondary or read.is_supplementary) and read.has_tag(cs_tag):
                    cs = read.get_tag(cs_tag)
                    entry = state.setdefault(cs, [0, 0])
                    entry[1] += 1
                    if read.has_tag(po_tag):
                        entry[0] |= _ORIENTATION_BIT.get(read.get_tag(po_tag), 0)
                window.append((read, cs))
                if counters["reads"] % 5_000_000 == 0:
                    logging.info("  %d reads, %d in the window", counters["reads"], len(window))
            while window:
                finish(*window.popleft(), out)
    return counters


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
    )
    open_kwargs: dict = {}
    if args.reference:
        open_kwargs["reference_filename"] = args.reference

    counters = mark_duplex(
        args.input,
        args.output,
        open_kwargs,
        tag=args.tag,
        cs_tag=args.cs_tag,
        po_tag=args.po_tag,
        pad=args.pad,
        threads=args.threads,
    )
    logging.info("Indexing %s", args.output)
    pysam.index("-@", str(args.threads), args.output)
    logging.info(
        "Done: %(reads)d reads, %(reads_with_cs)d with %(cs)s, %(cs_families)d CS families, "
        "%(duplex_families)d duplex, %(reads_dx1)d reads tagged %(tag)s=1",
        {**counters, "cs": args.cs_tag, "tag": args.tag},
    )


if __name__ == "__main__":
    main()
