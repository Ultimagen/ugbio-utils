#!/usr/bin/env python3
"""
Turn a paired-end, duplicate-marked BAM/CRAM into a single-end one whose reads carry their strand in the name.

Every primary read gets the digit of its strand (0 forward, 1 reverse) appended to its QNAME, and to its ``MI``
tag, so the consensus tool builds each duplicate family per strand. The pair FLAG bits (0x1, 0x2, 0x8, 0x20,
0x40, 0x80) and the mate fields are cleared. ``CS`` is left as it is: it is the only link between the two
strands of a molecule. Supplementary and secondary records are written unchanged.

With ``--pair-orientation-tag`` the orientation of each pair (F1R2, F2R1, FF or RR) is written to that tag before
the mate information is dropped. The mates are separate records without mate fields, so they are paired by QNAME
while streaming the sorted input: a read waits in a queue until its mate arrives or ``--pair-window`` bases have
passed. A read whose mate is further away, on another contig or absent gets no tag. The queue is bounded, so memory
does not grow with the file.

The file is cut into windows of ``--shard-size`` bases that are processed in parallel. A window also reads
``--pair-window`` bases on each side, so the pairing sees the same neighbours as in a serial pass, but only
the reads whose own position is in the window are written. The output is the same for every shard size.
The input needs an index (it is built when missing).

Usage:
    unpair_rename_by_strand in.bam out.bam --reference ref.fa --threads 4 --pair-orientation-tag po
"""

import argparse
import logging
import os
import re
import sys
import tempfile
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import pysam

logger = logging.getLogger(__name__)

CO_LINE = "@CO\tread pairs unpaired and QNAMEs renamed with strand suffix: 0 (forward) or 1 (reverse)"
DEFAULT_SHARD_SIZE = 1_000_000
DEFAULT_PAIR_WINDOW = 2_000
DEFAULT_MAX_QUEUE = 4_194_304
ORIENTATIONS = ("F1R2", "F2R1", "FF", "RR")
UNPLACED = "*"

PAIRED, REVERSE, READ1, READ2 = 0x1, 0x10, 0x40, 0x80
SECONDARY_OR_SUPPLEMENTARY = 0x100 | 0x800
PAIR_BITS = 0x1 | 0x2 | 0x8 | 0x20 | 0x40 | 0x80


@dataclass
class Window:
    index: int
    contig: str
    first: int  # 1-based, inclusive; 0 for the unplaced reads
    last: int


def _orientation(rev1, rev2) -> str:
    if rev1 == rev2:
        return "RR" if rev1 else "FF"
    return "F2R1" if rev1 else "F1R2"


def _remove_tag(read: pysam.AlignedSegment, tag: str) -> None:
    while read.has_tag(tag):
        read.set_tag(tag, None)


def _replace_tag(read: pysam.AlignedSegment, tag: str, value: str) -> None:
    """Replace a string tag keeping its place among the other tags (``set_tag`` moves it to the end)."""
    tags = read.get_tags(with_value_type=True)
    first = next(i for i, (name, _, _) in enumerate(tags) if name == tag)
    tail = tags[first:]
    for name, _, _ in tail:
        read.set_tag(name, None)
    read.set_tag(tag, value, "Z")
    for name, v, vtype in tail[1:]:
        # arrays carry their element type, which pysam takes from the array itself
        read.set_tag(name, v, None if vtype == "B" else vtype)


class _Entry:
    __slots__ = ("read", "contig", "pos", "orientation", "ready", "primary", "reverse", "is_read1", "name")

    def __init__(self, read, contig, pos):
        self.read = read
        self.contig = contig
        self.pos = pos
        self.orientation = ""
        self.ready = True
        self.primary = False
        self.reverse = False
        self.is_read1 = False
        self.name = None


class PairOrientationTagger:
    """
    Streams reads in coordinate order and sets the pair orientation tag on the primary reads of a pair.

    Reads leave in the order they arrived. ``owned`` selects the reads that are counted: the ones whose own
    position is in the window, because the others are only read to find mates.
    """

    def __init__(self, tag: str, pair_window: int, max_queue: int, owned):
        self.tag = tag
        self.pair_window = pair_window
        self.max_queue = max_queue
        self.owned = owned
        self.counts: Counter = Counter()
        self._queue: dict[int, _Entry] = {}
        self._waiting: dict[str, int] = {}  # QNAME -> queue index of the mate that waits for its partner
        self._head = 1
        self._tail = 0

    def process(self, reads: Iterator[pysam.AlignedSegment]) -> Iterator[pysam.AlignedSegment]:
        for read in reads:
            contig = read.reference_name or UNPLACED
            pos = read.reference_start + 1
            yield from self._release(contig, pos)
            if self._tail - self._head + 1 >= self.max_queue - 1:
                yield self._force()
            self._push(read, contig, pos)
            yield from self._release(contig, pos)
        for entry in self._queue.values():
            entry.ready = True
        yield from self._release("", 0)

    def _push(self, read, contig, pos):
        self._tail += 1
        entry = _Entry(read, contig, pos)
        self._queue[self._tail] = entry
        flag = read.flag
        if flag & SECONDARY_OR_SUPPLEMENTARY:
            return
        entry.primary = True
        is_read1, is_read2 = bool(flag & READ1), bool(flag & READ2)
        if not (flag & PAIRED) or is_read1 == is_read2:
            return
        entry.reverse = bool(flag & REVERSE)
        entry.is_read1 = is_read1
        name = read.query_name
        mate_index = self._waiting.get(name)
        if mate_index is not None:
            mate = self._queue[mate_index]
            if mate.is_read1 != is_read1:
                del self._waiting[name]
                orientation = _orientation(
                    entry.reverse if is_read1 else mate.reverse, mate.reverse if is_read1 else entry.reverse
                )
                mate.orientation = entry.orientation = orientation
                mate.ready = True
                return
            mate.ready = True  # same mate twice: the older one stops waiting
        self._waiting[name] = self._tail
        entry.name = name
        entry.ready = False

    def _release(self, contig, pos):
        while self._head <= self._tail:
            entry = self._queue[self._head]
            if not entry.ready:
                if entry.contig == contig and pos - entry.pos <= self.pair_window:
                    break
                self._waiting.pop(entry.name, None)
                entry.ready = True
            yield self._emit(self._head)
            self._head += 1

    def _force(self):
        entry = self._queue[self._head]
        if not entry.ready:
            self._waiting.pop(entry.name, None)
            entry.ready = True
        read = self._emit(self._head)
        self._head += 1
        return read

    def _emit(self, index):
        entry = self._queue.pop(index)
        read = entry.read
        if entry.primary:
            counted = self.owned(entry.contig, entry.pos)
            _remove_tag(read, self.tag)
            if entry.orientation:
                if counted:
                    self.counts[entry.orientation] += 1
                read.set_tag(self.tag, entry.orientation, "Z")
            elif counted:
                self.counts["none"] += 1
        return read


def unpair_and_rename(read: pysam.AlignedSegment, counters: Counter) -> None:
    """Rename a primary read with its strand digit and clear the pair information; count it."""
    counters["total_records"] += 1
    if read.flag & SECONDARY_OR_SUPPLEMENTARY:
        counters["supp_or_secondary"] += 1
        return
    reverse = bool(read.flag & REVERSE)
    suffix = "1" if reverse else "0"
    read.query_name = read.query_name + suffix
    if read.has_tag("MI"):
        value, value_type = read.get_tag("MI", with_value_type=True)
        if value_type == "Z":
            _replace_tag(read, "MI", value + suffix)
    if read.has_tag("CS") and read.get_tag("CS", with_value_type=True)[1] == "Z":
        counters["cs_tagged"] += 1
    read.flag &= ~PAIR_BITS
    read.next_reference_id = -1
    read.next_reference_start = -1
    read.template_length = 0
    counters["renamed_reverse" if reverse else "renamed_forward"] += 1


def _counter_names(tag: str | None) -> list[str]:
    names = ["total_records", "supp_or_secondary", "renamed_forward", "renamed_reverse", "cs_tagged"]
    if tag:
        names += [f"pair_orientation_{o}" for o in (*ORIENTATIONS, "none")]
    return names


def _open(path: str, reference: str | None):
    kwargs = {"reference_filename": reference} if reference else {}
    return pysam.AlignmentFile(path, **kwargs)


def _window_reads(fin: pysam.AlignmentFile, window: Window, pair_window: int):
    if window.contig == UNPLACED:
        return fin.fetch(UNPLACED)
    start = window.first - pair_window - 1 if window.first > pair_window else 0
    return fin.fetch(window.contig, start, window.last + pair_window)


def process_window(task: tuple) -> tuple[str, Counter]:
    """Write one window to ``<workdir>/<index>.bam`` and return its path and counters."""
    window, path, reference, workdir, tag, pair_window, max_queue = task
    counters: Counter = Counter()
    out_path = os.path.join(workdir, f"{window.index:06d}.bam")

    def owned(contig, pos):
        return window.contig == UNPLACED or (contig == window.contig and window.first <= pos <= window.last)

    with _open(path, reference) as fin:
        header = pysam.AlignmentHeader.from_text(str(fin.header).rstrip("\n") + "\n" + CO_LINE + "\n")
        reads = _window_reads(fin, window, pair_window)
        tagger = None
        if tag:
            tagger = PairOrientationTagger(tag, pair_window, max_queue, owned)
            reads = tagger.process(reads)
        with pysam.AlignmentFile(out_path, "wb", header=header) as fout:
            for read in reads:
                contig = read.reference_name or UNPLACED
                if not owned(contig, read.reference_start + 1):
                    continue
                unpair_and_rename(read, counters)
                fout.write(read)
    if tagger:
        for orientation in ORIENTATIONS:
            counters[f"pair_orientation_{orientation}"] = tagger.counts[orientation]
        counters["pair_orientation_none"] = tagger.counts["none"]
    return out_path, counters


def _cram_record_count(path: str) -> int:
    """Records in a CRAM, summed from its container headers (CRAI has no per-contig counts)."""
    for line in pysam.cram_size(path).splitlines():
        if line.startswith("Number of sequences"):
            return int(line.split()[-1])
    raise RuntimeError(f"samtools cram-size did not report a record count for {path}")


def make_windows(fin: pysam.AlignmentFile, shard_size: int) -> tuple[list[Window], int]:
    """The windows to process and the number of records in the file.

    For BAM the index statistics give per-contig counts, so contigs without reads are skipped. CRAI has
    no such statistics, so a CRAM gets windows over every contig plus the unplaced reads, and its record
    count comes from the container headers.
    """
    windows, total = [], 0
    if fin.is_cram:
        contigs = list(fin.references)
        total = _cram_record_count(os.fsdecode(fin.filename))
        has_unplaced = True
    else:
        contigs = []
        for stats in fin.get_index_statistics():
            if stats.mapped + stats.unmapped == 0:
                continue
            total += stats.mapped + stats.unmapped
            contigs.append(stats.contig)
        total += fin.nocoordinate
        has_unplaced = bool(fin.nocoordinate)
    for contig in contigs:
        length = fin.get_reference_length(contig)
        for first in range(1, length + 1, shard_size):
            windows.append(Window(len(windows), contig, first, min(first + shard_size - 1, length)))
    if has_unplaced:
        windows.append(Window(len(windows), UNPLACED, 0, 0))
    return windows, total


def _has_index(path: str) -> bool:
    return any(os.path.exists(path + ext) for ext in (".bai", ".csi", ".crai")) or os.path.exists(
        re.sub(r"\.(bam|cram)$", ".bai" if path.endswith(".bam") else ".crai", path)
    )


def unpair_by_strand(
    input_path: str,
    output_path: str,
    *,
    reference: str | None = None,
    shard_size: int = DEFAULT_SHARD_SIZE,
    pair_orientation_tag: str | None = None,
    pair_window: int = DEFAULT_PAIR_WINDOW,
    max_queue: int = DEFAULT_MAX_QUEUE,
    threads: int = 1,
    workdir: str | None = None,
) -> dict[str, int]:
    """Process the whole file window by window and return the summed counters."""
    with tempfile.TemporaryDirectory(dir=workdir or ".", prefix="windows_") as tmp:
        if not _has_index(input_path):
            linked = os.path.join(tmp, os.path.basename(input_path))
            os.symlink(os.path.abspath(input_path), linked)
            pysam.index("-@", str(threads), linked)
            input_path = linked
        with _open(input_path, reference) as fin:
            windows, expected = make_windows(fin, shard_size)
        tasks = [(w, input_path, reference, tmp, pair_orientation_tag, pair_window, max_queue) for w in windows]
        with ProcessPoolExecutor(max_workers=threads) as pool:
            results = list(pool.map(process_window, tasks))

        total: Counter = Counter()
        for _, counters in results:
            total.update(counters)
        written = total["total_records"]
        if written != expected:
            raise RuntimeError(f"wrote {written} records but the input holds {expected}")

        listing = os.path.join(tmp, "windows.fofn")
        with open(listing, "w") as fh:
            fh.writelines(path + "\n" for path, _ in results)
        if results:
            pysam.samtools.cat("-b", listing, "-o", output_path, catch_stdout=False)
        else:
            header = pysam.AlignmentHeader.from_text(_header_with_co(input_path, reference))
            pysam.AlignmentFile(output_path, "wb", header=header).close()
    return {name: total[name] for name in _counter_names(pair_orientation_tag)}


def _header_with_co(path: str, reference: str | None) -> str:
    with _open(path, reference) as fin:
        return str(fin.header).rstrip("\n") + "\n" + CO_LINE + "\n"


def write_counters(counters: dict[str, int], path: str) -> None:
    with open(path, "w") as out:
        for name, value in counters.items():
            out.write(f"{name}\t{value}\n")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Unpair a BAM/CRAM and rename the reads with their strand")
    parser.add_argument("input", help="Coordinate-sorted, indexed BAM/CRAM, duplicates marked")
    parser.add_argument("output", help="Output BAM")
    parser.add_argument("--reference", "-T", help="Reference FASTA (required for CRAM input)")
    parser.add_argument("--counters", default="unpair_rename_counters.txt", help="Counters file to write")
    parser.add_argument(
        "--pair-orientation-tag", help="Two-character tag that receives the pair orientation (F1R2, F2R1, FF, RR)"
    )
    parser.add_argument(
        "--pair-window",
        type=int,
        default=DEFAULT_PAIR_WINDOW,
        metavar="BP",
        help=f"Mates further apart than this are not paired (default {DEFAULT_PAIR_WINDOW})",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=DEFAULT_SHARD_SIZE,
        metavar="BP",
        help=f"Size of the windows processed in parallel (default {DEFAULT_SHARD_SIZE})",
    )
    parser.add_argument(
        "--max-queue",
        type=int,
        default=DEFAULT_MAX_QUEUE,
        help="Most reads held while looking for mates; the oldest is written untagged when it is full",
    )
    parser.add_argument("--threads", type=int, default=1, help="Windows processed in parallel (default 1)")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    tag = args.pair_orientation_tag
    if tag and not re.fullmatch(r"[A-Za-z][A-Za-z0-9]", tag):
        logger.error("--pair-orientation-tag must be a two-character SAM tag, got %r", tag)
        return 1
    counters = unpair_by_strand(
        args.input,
        args.output,
        reference=args.reference,
        shard_size=args.shard_size,
        pair_orientation_tag=tag,
        pair_window=args.pair_window,
        max_queue=args.max_queue,
        threads=args.threads,
    )
    write_counters(counters, args.counters)
    with open(args.counters) as fh:
        sys.stdout.write(fh.read())
    return 0


if __name__ == "__main__":
    sys.exit(main())
