#!/usr/bin/env python3
"""
Set the reverse-strand flag (0x10) of consensus reads from the majority of their source reads.

read_fuser builds each consensus read from the flow matrix of its duplicate set and emits it
reference-forward, so UA aligns it forward and the strand of the source molecule is lost. The strand
composition is still in the tags: ``nf`` and ``nr`` count the forward- and reverse-strand reads that were fused.
The flag is therefore set from the majority: ``nr > nf`` is a reverse read, a tie is forward. Only FLAG changes:
for a mapped read SEQ is stored in reference orientation whatever 0x10 says, so the alignment keeps its meaning.

Records that are left untouched: supplementary and secondary alignments, paired records (they were aligned as
a pair, so their strand is already set) and records without ``nf``/``nr`` (not consensus reads).

The input does not have to be sorted. The counters are written to ``--counters`` and the step fails when a
record has no ``nf``/``nr`` tag, or, with ``--require-cs-tag``, when a consensus read has no ``CS`` tag.

Usage:
    set_consensus_strand in.bam out.bam --counters strand_flag_counters.txt
"""

import argparse
import itertools
import logging
import sys
from collections import Counter

import pysam

logger = logging.getLogger(__name__)

CO_LINE = "@CO\tconsensus reverse-strand flag (0x10) set from the nf/nr majority: nr>nf reverse, otherwise forward"
COUNTER_NAMES = (
    "total_records",
    "supplementary_or_secondary",
    "paired_left_alone",
    "no_nf_nr_tags",
    "no_cs_tag",
    "majority_forward",
    "majority_reverse",
    "majority_tie",
    "flag_reverse_set",
    "flag_reverse_cleared",
    "flag_unchanged",
)
REVERSE = 0x10
PAIRED = 0x1
SECONDARY_OR_SUPPLEMENTARY = 0x100 | 0x800


def _int_tag(read: pysam.AlignedSegment, tag: str) -> int:
    """Value of an integer tag, or -1 when the tag is absent or not an integer."""
    try:
        value = read.get_tag(tag)
    except KeyError:
        return -1
    return value if isinstance(value, int) else -1


def _has_string_tag(read: pysam.AlignedSegment, tag: str) -> bool:
    return read.has_tag(tag) and read.get_tag(tag, with_value_type=True)[1] == "Z"


def set_strand(read: pysam.AlignedSegment, counters: Counter, *, nf: int, nr: int) -> None:
    """Assign (do not toggle) the reverse flag from the majority and update the counters."""
    want_reverse = nr > nf
    is_reverse = bool(read.flag & REVERSE)
    if want_reverse and not is_reverse:
        read.flag |= REVERSE
        counters["flag_reverse_set"] += 1
    elif not want_reverse and is_reverse:
        read.flag &= ~REVERSE
        counters["flag_reverse_cleared"] += 1
    else:
        counters["flag_unchanged"] += 1
    if nr > nf:
        counters["majority_reverse"] += 1
    elif nf > nr:
        counters["majority_forward"] += 1
    else:
        counters["majority_tie"] += 1


def process_read(read: pysam.AlignedSegment, counters: Counter) -> None:
    counters["total_records"] += 1
    if read.flag & SECONDARY_OR_SUPPLEMENTARY:
        counters["supplementary_or_secondary"] += 1
        return
    if read.flag & PAIRED:
        counters["paired_left_alone"] += 1
        return
    nf, nr = _int_tag(read, "nf"), _int_tag(read, "nr")
    if nf < 0 or nr < 0:
        counters["no_nf_nr_tags"] += 1
        return
    if not _has_string_tag(read, "CS"):
        counters["no_cs_tag"] += 1
    set_strand(read, counters, nf=nf, nr=nr)


def set_consensus_strand(
    input_path: str,
    output_path: str,
    *,
    reference: str | None = None,
    threads: int = 1,
) -> Counter:
    """Write ``input_path`` to ``output_path`` with the consensus strand flags set. Returns the counters."""
    counters: Counter = Counter()
    open_kwargs = {"reference_filename": reference} if reference else {}
    with pysam.AlignmentFile(input_path, threads=threads, **open_kwargs) as fin:
        records = iter(fin)
        first = next(records, None)
        header_text = str(fin.header)
        if first is not None:
            header_text = header_text.rstrip("\n") + "\n" + CO_LINE + "\n"
        header = pysam.AlignmentHeader.from_text(header_text)
        with pysam.AlignmentFile(output_path, "wb", header=header, threads=threads) as fout:
            if first is not None:
                for read in itertools.chain([first], records):
                    process_read(read, counters)
                    fout.write(read)
    return counters


def write_counters(counters: Counter, path: str) -> None:
    with open(path, "w") as out:
        for name in COUNTER_NAMES:
            out.write(f"{name}\t{counters[name]}\n")


def check_counters(counters: Counter, *, require_cs_tag: bool) -> list[str]:
    """Reasons the step has to fail, empty when the counters are acceptable."""
    errors = []
    if counters["no_nf_nr_tags"]:
        errors.append(
            f"{counters['no_nf_nr_tags']} unpaired records have no nf/nr tags: read_fuser and this step disagree "
            "about what a consensus read is"
        )
    if require_cs_tag and counters["no_cs_tag"]:
        errors.append(
            f"{counters['no_cs_tag']} consensus records carry no CS tag, but CS-keyed duplicate marking was "
            "requested. CS is stamped by mark_duplicates_mi --use-umi and must be carried through read_fuser "
            "with --umi-tags u3,u5,CS"
        )
    return errors


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Set the consensus reads' reverse-strand flag from nf/nr")
    parser.add_argument("input", help="Aligned consensus BAM/CRAM")
    parser.add_argument("output", help="Output BAM")
    parser.add_argument("--reference", "-T", help="Reference FASTA (required for CRAM input)")
    parser.add_argument("--counters", default="strand_flag_counters.txt", help="Counters file to write")
    parser.add_argument(
        "--require-cs-tag", action="store_true", help="Fail if a consensus read has no CS tag (CS-keyed marking)"
    )
    parser.add_argument("--threads", type=int, default=1, help="htslib threads (default 1)")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    counters = set_consensus_strand(args.input, args.output, reference=args.reference, threads=args.threads)
    write_counters(counters, args.counters)
    with open(args.counters) as fh:
        sys.stdout.write(fh.read())
    errors = check_counters(counters, require_cs_tag=args.require_cs_tag)
    for error in errors:
        logger.error(error)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
