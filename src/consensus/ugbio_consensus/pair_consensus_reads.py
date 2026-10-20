#!/usr/bin/env python3
"""
Pair the two strand-consensus reads of one pair, so that the aligner places them as a pair.

After ``unpair_reads_by_strand`` every read name carries a strand digit (0 for a forward read, 1 for a
reverse one), so ``read_fuser`` builds one single-end consensus read per strand of a pair. They come out
unaligned, in reference orientation, with the name ``<stem><digit>`` and with the same ``CS`` and ``po``.
Aligned alone, one of the two can land with a low MAPQ in a repeat that the other would have resolved.

For every stem that has both digits and a ``po`` of F1R2 or F2R1, this writes the two reads as an
unaligned pair, adjacent and named ``<stem>``:

* R1 is the forward read for F1R2 and the reverse read for F2R1; flags 77 (R1) and 141 (R2).
* The reverse read is reverse-complemented (quality reversed), because a mate is given to the aligner as
  it was sequenced. ``tp``/``t0`` stay as they are: the read group declares them reference-oriented.

Everything else is written unchanged: reads with no partner, a ``po`` other than F1R2/F2R1, or partners
that disagree on ``po`` or ``CS``.

The aligner then sets the strand and pair flags itself, so ``SetConsensusStrandByMajority`` must not run
on this output.

The input need not be sorted. Reads are spread over ``--buckets`` temporary files by stem, and each
bucket is paired in memory on its own.

Usage:
    pair_consensus_reads consensus.bam consensus.paired.bam
"""

import argparse
import logging
import os
import tempfile
import zlib
from collections import defaultdict

import pysam

R1_FLAG = 1 | 4 | 8 | 64
R2_FLAG = 1 | 4 | 8 | 128
_ORIENTATIONS = ("F1R2", "F2R1")
_COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Pair the two strand-consensus reads of a pair, for the aligner")
    parser.add_argument("input", help="Unaligned consensus BAM/SAM (read_fuser output, consensus reads only)")
    parser.add_argument("output", help="Output unaligned BAM")
    parser.add_argument("--po-tag", default="po", help="Tag holding the pair orientation (default po)")
    parser.add_argument("--cs-tag", default="CS", help="Tag shared by the two strands of a molecule (default CS)")
    parser.add_argument("--buckets", type=int, default=64, help="Temporary files to spread the reads over (default 64)")
    parser.add_argument("--tmp-dir", help="Directory for the temporary files (default: system temp)")
    parser.add_argument("--threads", type=int, default=1, help="htslib compression threads (default 1)")
    parser.add_argument("--counters", help="Write one 'name<TAB>count' line per outcome to this file")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args(argv)


def _tag(read, tag):
    return read.get_tag(tag) if read.has_tag(tag) else None


def _make_pair(forward, reverse, po: str):
    """Turn the two strand reads into R1/R2 records, in sequencing orientation, sharing one name."""
    stem = forward.query_name[:-1]
    seq, qual = reverse.query_sequence, reverse.query_qualities
    reverse.query_sequence = seq.translate(_COMPLEMENT)[::-1]
    reverse.query_qualities = qual[::-1]
    r1, r2 = (forward, reverse) if po == "F1R2" else (reverse, forward)
    r1.query_name = r2.query_name = stem
    r1.flag, r2.flag = R1_FLAG, R2_FLAG
    return r1, r2


def pair_group(reads, po_tag: str, cs_tag: str):
    """Return the records to write for one stem group and 'paired' or the reason it was not."""
    by_digit = defaultdict(list)
    for read in reads:
        by_digit[read.query_name[-1]].append(read)
    if set(by_digit) != {"0", "1"} or any(len(v) != 1 for v in by_digit.values()):
        return reads, "no_partner" if len(reads) == 1 else "ambiguous"
    forward, reverse = by_digit["0"][0], by_digit["1"][0]
    po = _tag(forward, po_tag)
    if po not in _ORIENTATIONS:
        return reads, "other_orientation"
    if po != _tag(reverse, po_tag) or _tag(forward, cs_tag) != _tag(reverse, cs_tag):
        return reads, "partners_disagree"
    return _make_pair(forward, reverse, po), "paired"


def pair_reads(input_path: str, output_path: str, *, po_tag="po", cs_tag="CS", buckets=64, tmp_dir=None, threads=1):
    counters = defaultdict(int)
    with tempfile.TemporaryDirectory(dir=tmp_dir) as tmp:
        paths = [os.path.join(tmp, f"bucket{i}.bam") for i in range(buckets)]
        with pysam.AlignmentFile(input_path, "r", check_sq=False, threads=threads) as bam:
            header = bam.header
            files = [pysam.AlignmentFile(p, "wbu", header=header) for p in paths]
            try:
                for read in bam:
                    counters["reads"] += 1
                    files[zlib.crc32(read.query_name[:-1].encode()) % buckets].write(read)
            finally:
                for f in files:
                    f.close()
        with pysam.AlignmentFile(output_path, "wb", header=header, threads=threads) as out:
            for path in paths:
                groups = defaultdict(list)
                with pysam.AlignmentFile(path, "r", check_sq=False) as bam:
                    for read in bam:
                        groups[read.query_name[:-1]].append(read)
                for reads in groups.values():
                    records, outcome = pair_group(reads, po_tag, cs_tag)
                    counters[outcome] += len(records) // 2 if outcome == "paired" else len(records)
                    for record in records:
                        out.write(record)
                os.remove(path)
    return dict(counters)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
    )
    counters = pair_reads(
        args.input,
        args.output,
        po_tag=args.po_tag,
        cs_tag=args.cs_tag,
        buckets=args.buckets,
        tmp_dir=args.tmp_dir,
        threads=args.threads,
    )
    logging.info(
        "Done: %d reads in; %d pairs made; left single: %d without a partner, %d other orientation, "
        "%d partners disagreeing, %d ambiguous",
        counters.get("reads", 0),
        counters.get("paired", 0),
        counters.get("no_partner", 0),
        counters.get("other_orientation", 0),
        counters.get("partners_disagree", 0),
        counters.get("ambiguous", 0),
    )
    if args.counters:
        with open(args.counters, "w") as fh:
            for key in ("reads", "paired", "no_partner", "other_orientation", "partners_disagree", "ambiguous"):
                fh.write(f"{key}\t{counters.get(key, 0)}\n")


if __name__ == "__main__":
    main()
