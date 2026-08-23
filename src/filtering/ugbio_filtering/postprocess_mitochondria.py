"""Post-process mitochondrial variants: soft-flag NUMT bleed, and fix haploid genotypes.

Two independent corrections, each switched on by its own arguments, applied in a single pass over
the VCF.  Pass the ``--numt_*`` group, ``--regenotype_contig``, or both.

NUMT annotation
---------------
NUMTs (nuclear mitochondrial DNA segments) are paralogous copies of mtDNA in the nuclear
genome.  Reads originating in a NUMT can mismap to chrM, where they manufacture apparent
low-frequency heteroplasmy.  A record is flagged only when all three terms hold:

  1. POS overlaps a chrM NUMT-homology interval, and
  2. alt-supporting reads carry a supplementary alignment (SA) to a nuclear NUMT locus at a
     rate exceeding the ref-supporting reads *at the same site* by more than ``sa_excess``.
     The local reference rate is the correct background, not a global constant: inside the
     control region every read carries a NUMT SA tag at 12-23%, alt and ref alike, so a
     fixed rate would discard true variants.
  3. the read-derived VAF is at most ``vaf_ceiling``.  This is a property of the molecule
     rather than a tuning parameter -- the two nuclear copies of a NUMT cannot outvote the
     thousands of mtDNA copies per cell, so a high-VAF call cannot have come from a NUMT
     however its reads are tagged.

All three terms are required.  On hg38 the chrM regions with a nuclear paralog cover 41.6% of
the contig, so term 1 on its own flags only real homoplasmic variants and no suspect ones.

The interval files are a pair: the chrM-side BED carries the VCF ``##INFO=<ID=...>`` header as
its first line (the same idiom ``ug_postproc --bed_annotation_files`` uses) and that ID becomes
both the INFO tag and the FILTER name; the nuclear-side BED is what a read's SA target is
tested against.  No record is dropped -- use ``bcftools view -f PASS`` for the filtered callset.

Regenotyping
------------
The caller runs a diploid model on a haploid, multi-copy contig, so its ``1/1`` state is
effectively unreachable there: over a 7-sample cohort every one of 208 chrM calls came out
``0/1``, including 183 whose own ``FORMAT/VAF`` was at or above 0.85.  ``INFO/AF`` is derived
from the genotype and therefore reads a constant 0.5 on every record.  ``FORMAT/VAF`` and
``FORMAT/AD``, by contrast, agree with an orthogonal caller to ~0.005, so the information is
already in the record and only the genotype summary needs rewriting.

On the requested contig, an alternate allele at ``FORMAT/VAF >= homoplasmy_vaf`` therefore
becomes homozygous and ``INFO/AF`` is recomputed from the observed VAF.  The default 0.85 sits
in the middle of a plateau: sweeping it over that cohort leaves 3 genotypes disagreeing with the
orthogonal caller anywhere in 0.75-0.85, 4 at 0.90, 8 at 0.95 and 142 at 0.99.  The 3 that
remain are heteroplasmies at AF 0.81-0.90, where the hom/het label is a matter of convention and
no threshold helps.

``FORMAT/PL``, ``FORMAT/GQ`` and ``QUAL`` are deliberately left as the model emitted them:
they carry no usable information on this contig (a site with 1183 alt reads out of 1192 scored
``PL=0,0,33``, ref and het exactly tied), and rewriting them would mean substituting a
contig-specific likelihood model rather than relabelling what the caller already measured.  So
``PL`` can disagree with ``GT`` on a regenotyped record; the ``MT_REGT`` header says so.  The
genotype as emitted is preserved in ``FORMAT/OGT``.
"""

import argparse

import pysam
from ugbio_core.logger import logger

# A supplementary record carries an SA tag pointing back at its own primary, so counting it
# would double-count the pair.  Primary alignments only.
DROP_FLAGS = 0x4 | 0x100 | 0x200 | 0x800

# Cigar operations, by pysam's integer encoding.
CIGAR_ALIGNED_OPS = frozenset({0, 7, 8})  # M, =, X
CIGAR_INSERTION = 1
CIGAR_DELETION = 2
CIGAR_REF_SKIP = 3

# The gVCF stand-in for "any other allele".  It has no VAF worth ranking and must never be
# genotyped, but it does occupy a slot in the Number=A and Number=R vectors.
SYMBOLIC_ALT = "<*>"

# INFO/AF is written to the same 4 decimals the orthogonal caller reports, rather than the full
# float32 expansion of FORMAT/VAF.
AF_PRECISION = 4

OGT_HEADER = (
    '##FORMAT=<ID=OGT,Number=1,Type=String,Description="Genotype as emitted by the diploid model, '
    'before mitochondrial regenotyping">'
)
MT_REGT_HEADER = (
    '##INFO=<ID=MT_REGT,Number=0,Type=Flag,Description="GT reassigned from FORMAT/VAF because the '
    "diploid model cannot express a haploid homoplasmic genotype on this contig. FORMAT/PL, FORMAT/GQ "
    'and QUAL are left as the model emitted them and may disagree with GT">'
)


def read_bed(bed_file: str) -> dict[str, list[tuple[int, int]]]:
    """Read a BED into ``contig -> [(start, end)]``, skipping headers and comments."""
    intervals: dict[str, list[tuple[int, int]]] = {}
    with open(bed_file) as fh:
        for line in fh:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            fields = line.split()
            intervals.setdefault(fields[0], []).append((int(fields[1]), int(fields[2])))
    return intervals


def info_id_from_bed(bed_file: str) -> tuple[str, str]:
    """Return ``(tag, header_line)`` from the BED's first line, which must be a VCF INFO header."""
    with open(bed_file) as fh:
        first_line = fh.readline().strip()
    if not first_line.startswith("##INFO=<"):
        raise ValueError(f"{bed_file}: first line must be a VCF ##INFO=<ID=...> header, got: {first_line}")
    for field in first_line[len("##INFO=<") :].rstrip(">").split(","):
        if field.startswith("ID="):
            return field[len("ID=") :], first_line
    raise ValueError(f"{bed_file}: ##INFO header carries no ID= field")


def overlaps(intervals: dict[str, list[tuple[int, int]]], contig: str, pos: int) -> bool:
    """True if a 1-based position falls in one of the contig's half-open intervals."""
    return any(start <= pos < end for start, end in intervals.get(contig, ()))


def match_allele(ref: str, alt: str) -> tuple[str | None, str]:
    """Convert a VCF (REF, ALT) pair to ``(token, variant_type)``; token is a base, +NSEQ or -NSEQ."""
    ref, alt = str(ref).upper(), str(alt).upper()
    if len(ref) == 1 and len(alt) == 1:
        return alt, "snp"
    if len(alt) > len(ref) and alt.startswith(ref):
        inserted = alt[len(ref) :]
        return f"+{len(inserted)}{inserted}", "indel"
    if len(ref) > len(alt) and ref.startswith(alt):
        deleted = ref[len(alt) :]
        return f"-{len(deleted)}{deleted}", "indel"
    return None, "complex"  # MNP / substitution: no single token


def has_numt_supplementary(read: pysam.AlignedSegment, numt_intervals: dict[str, list[tuple[int, int]]]) -> bool:
    """True if the read has a supplementary alignment landing in a nuclear NUMT locus."""
    if not read.has_tag("SA"):
        return False
    for entry in read.get_tag("SA").split(";"):
        fields = entry.split(",")
        if len(fields) < 2:  # noqa: PLR2004 - an SA entry is contig,pos,strand,cigar,mapq,nm
            continue
        contig, pos = fields[0], int(fields[1])
        # A split back onto the same contig is the circular chrM junction, not a paralog.
        if contig != read.reference_name and overlaps(numt_intervals, contig, pos):
            return True
    return False


def _supports_snp(read: pysam.AlignedSegment, pos: int, token: str) -> str | None:
    for query_pos, ref_pos in read.get_aligned_pairs(matches_only=True):
        if ref_pos == pos - 1:
            return "alt" if read.query_sequence[query_pos].upper() == token else "ref"
    return None


def _supports_indel(read: pysam.AlignedSegment, pos: int, token: str) -> str | None:
    want_insertion = token.startswith("+")
    length = int("".join(char for char in token[1:] if char.isdigit()))
    ref_pos = read.reference_start
    covers_site = False
    for operation, op_length in read.cigartuples or []:
        if operation in CIGAR_ALIGNED_OPS:
            if ref_pos <= pos - 1 < ref_pos + op_length:
                covers_site = True
            ref_pos += op_length
        elif operation == CIGAR_INSERTION:
            if ref_pos == pos:  # insertion immediately after pos
                return "alt" if (want_insertion and op_length == length) else "ref"
        elif operation == CIGAR_DELETION:
            if ref_pos == pos:
                return "alt" if (not want_insertion and op_length == length) else "ref"
            ref_pos += op_length
        elif operation == CIGAR_REF_SKIP:
            ref_pos += op_length
    return "ref" if covers_site else None


def supports(read: pysam.AlignedSegment, pos: int, token: str, variant_type: str) -> str | None:
    """Classify one read at a 1-based position as ``"alt"``, ``"ref"`` or None (uninformative)."""
    if variant_type == "snp":
        return _supports_snp(read, pos, token)
    return _supports_indel(read, pos, token)


def collect_evidence(
    alignment_files: list[pysam.AlignmentFile],
    contig: str,
    pos: int,
    token: str,
    variant_type: str,
    numt_intervals: dict[str, list[tuple[int, int]]],
) -> tuple[float, float, float] | None:
    """Return ``(alt_numt_fraction, ref_numt_fraction, vaf)`` from the reads, or None if unusable."""
    n_reads = {"alt": 0, "ref": 0}
    n_numt = {"alt": 0, "ref": 0}
    for alignment_file in alignment_files:
        if contig not in alignment_file.references:
            continue
        for read in alignment_file.fetch(contig, max(0, pos - 1), pos):
            if read.flag & DROP_FLAGS:
                continue
            side = supports(read, pos, token, variant_type)
            if side is None:
                continue
            n_reads[side] += 1
            if has_numt_supplementary(read, numt_intervals):
                n_numt[side] += 1
    total = n_reads["alt"] + n_reads["ref"]
    if not total or not n_reads["alt"]:
        return None
    return (
        n_numt["alt"] / n_reads["alt"],
        n_numt["ref"] / n_reads["ref"] if n_reads["ref"] else 0.0,
        n_reads["alt"] / total,
    )


def is_numt_supported(
    record: pysam.VariantRecord,
    alignment_files: list[pysam.AlignmentFile],
    numt_nuclear_intervals: dict[str, list[tuple[int, int]]],
    vaf_ceiling: float,
    sa_excess: float,
) -> bool:
    """True if any alternate allele of the record satisfies the read-evidence terms (2 and 3)."""
    for alt in record.alts or ():
        token, variant_type = match_allele(record.ref, alt)
        if token is None:
            continue
        evidence = collect_evidence(
            alignment_files, record.contig, record.pos, token, variant_type, numt_nuclear_intervals
        )
        if evidence is None:
            continue
        alt_numt_fraction, ref_numt_fraction, vaf = evidence
        if alt_numt_fraction - ref_numt_fraction > sa_excess and vaf <= vaf_ceiling:
            return True
    return False


def _reported_vafs(record: pysam.VariantRecord) -> tuple[float, ...] | None:
    """Return the single sample's FORMAT/VAF as one value per ALT, or None if it is unusable.

    A gVCF reference block reports ``VAF=.``, which arrives as None.  There is nothing to
    regenotype from and nothing to put in INFO/AF, so such a record is passed through untouched.
    """
    vafs = record.samples[0].get("VAF")
    if vafs is None:
        return None
    if not isinstance(vafs, tuple):
        vafs = (vafs,)
    if len(vafs) != len(record.alts or ()) or any(vaf is None for vaf in vafs):
        return None
    return vafs


def _dominant_alt(record: pysam.VariantRecord, vafs: tuple[float, ...]) -> tuple[int, float] | None:
    """Return ``(allele index, VAF)`` of the alternate allele with the highest reported VAF."""
    best: tuple[int, float] | None = None
    for allele_index, (alt, vaf) in enumerate(zip(record.alts or (), vafs, strict=True), start=1):
        if alt == SYMBOLIC_ALT:
            continue
        if best is None or vaf > best[1]:
            best = (allele_index, vaf)
    return best


def regenotype_record(record: pysam.VariantRecord, homoplasmy_vaf: float) -> bool:
    """Recompute INFO/AF from FORMAT/VAF and set GT homozygous above the threshold.

    Returns True if the genotype was changed.  INFO/AF is rewritten on every record with a usable
    VAF, changed genotype or not, because the emitted value is derived from GT and is therefore a
    constant on a contig where the model can only reach 0/1.
    """
    vafs = _reported_vafs(record)
    if vafs is None:
        return False

    record.info["AF"] = tuple(0.0 if vaf is None else round(vaf, AF_PRECISION) for vaf in vafs)

    dominant = _dominant_alt(record, vafs)
    if dominant is None:
        return False
    allele_index, vaf = dominant
    if vaf < homoplasmy_vaf:
        return False

    sample = record.samples[0]
    genotype = sample.get("GT")
    # A missing genotype means the model declined to call; do not invent one from the reads.
    if not genotype or all(allele is None for allele in genotype):
        return False
    if set(genotype) == {allele_index}:
        return False

    sample["OGT"] = "/".join("." if allele is None else str(allele) for allele in genotype)
    sample["GT"] = (allele_index, allele_index)
    sample.phased = False
    record.info["MT_REGT"] = True
    return True


def _setup_numt(
    header: pysam.VariantHeader,
    numt_intervals_file: str,
    numt_nuclear_intervals_file: str,
    input_alignments: list[str],
    reference: str | None,
) -> tuple[str, dict, dict, list[pysam.AlignmentFile]]:
    """Read the interval sets, open the alignments and declare the NUMT INFO and FILTER lines."""
    tag, info_line = info_id_from_bed(numt_intervals_file)
    chrm_intervals = read_bed(numt_intervals_file)
    nuclear_intervals = read_bed(numt_nuclear_intervals_file)
    logger.info(
        f"tag={tag}, chrM intervals={sum(len(v) for v in chrm_intervals.values())}, "
        f"nuclear loci={sum(len(v) for v in nuclear_intervals.values())}"
    )
    alignment_files = [pysam.AlignmentFile(path, reference_filename=reference) for path in input_alignments]
    if tag not in header.info:
        header.add_line(info_line)
    if tag not in header.filters:
        header.add_line(
            f'##FILTER=<ID={tag},Description="Alt reads carry supplementary alignments to a '
            f'nuclear mitochondrial (NUMT) paralog">'
        )
    return tag, chrm_intervals, nuclear_intervals, alignment_files


def _setup_regenotyping(header: pysam.VariantHeader, input_vcf: str) -> None:
    """Declare the FORMAT/OGT and INFO/MT_REGT lines, and reject input regenotyping cannot describe."""
    # INFO/AF and the genotype are per-sample decisions read off one sample's VAF, so a
    # multi-sample VCF would need a per-sample INFO/AF that the format cannot express.
    if len(header.samples) != 1:
        raise ValueError(f"regenotyping needs a single-sample VCF, {input_vcf} has {len(header.samples)} samples")
    if "OGT" not in header.formats:
        header.add_line(OGT_HEADER)
    if "MT_REGT" not in header.info:
        header.add_line(MT_REGT_HEADER)


def postprocess_mitochondria(  # noqa: C901, PLR0913 - two independent corrections, each with its own inputs
    input_vcf: str,
    output_vcf: str,
    numt_intervals_file: str | None = None,
    numt_nuclear_intervals_file: str | None = None,
    input_alignments: list[str] | None = None,
    reference: str | None = None,
    vaf_ceiling: float = 0.90,
    sa_excess: float = 0.02,
    regenotype_contig: str | None = None,
    homoplasmy_vaf: float = 0.85,
):
    """Annotate NUMT bleed and/or regenotype a haploid contig in one pass.

    Passing ``numt_intervals_file`` enables the NUMT annotation, which then also needs the nuclear
    intervals, the alignments and the reference.  Passing ``regenotype_contig`` enables
    regenotyping.  Every input record is written out; nothing is dropped.
    """
    annotate = numt_intervals_file is not None
    regenotype = regenotype_contig is not None
    if not annotate and not regenotype:
        raise ValueError("nothing to do: pass numt_intervals_file, regenotype_contig, or both")

    tag = ""
    chrm_intervals: dict[str, list[tuple[int, int]]] = {}
    nuclear_intervals: dict[str, list[tuple[int, int]]] = {}
    alignment_files: list[pysam.AlignmentFile] = []

    vcf_in = pysam.VariantFile(input_vcf)
    header = vcf_in.header

    if annotate:
        tag, chrm_intervals, nuclear_intervals, alignment_files = _setup_numt(
            header, numt_intervals_file, numt_nuclear_intervals_file, input_alignments, reference
        )
    if regenotype:
        _setup_regenotyping(header, input_vcf)
        logger.info(f"Regenotyping {regenotype_contig} records with FORMAT/VAF >= {homoplasmy_vaf}")

    logger.info(f"Processing {input_vcf} and writing to {output_vcf}")
    vcf_out = pysam.VariantFile(output_vcf, "wz", header=header)
    n_in_interval = 0
    n_flagged = 0
    n_on_contig = 0
    n_regenotyped = 0
    for record in vcf_in:
        if annotate:
            # Term 1: the position must overlap the chrM NUMT-homology set.
            if overlaps(chrm_intervals, record.contig, record.pos):
                n_in_interval += 1
                record.info[tag] = True
                # Terms 2 and 3.
                if is_numt_supported(record, alignment_files, nuclear_intervals, vaf_ceiling, sa_excess):
                    record.filter.add(tag)
                    n_flagged += 1
        # Regenotyping reads FORMAT/VAF while the NUMT terms read the CRAM, so neither can see the
        # other's edits and the order within the record does not matter.
        if regenotype and record.contig == regenotype_contig:
            n_on_contig += 1
            if regenotype_record(record, homoplasmy_vaf):
                n_regenotyped += 1
        vcf_out.write(record)

    vcf_out.close()
    vcf_in.close()
    for alignment_file in alignment_files:
        alignment_file.close()

    pysam.tabix_index(output_vcf, preset="vcf", force=True)
    if annotate:
        logger.info(f"Records inside a NUMT interval: {n_in_interval}, records FILTERed {tag}: {n_flagged}")
    if regenotype:
        logger.info(f"Records on {regenotype_contig}: {n_on_contig}, genotypes reassigned: {n_regenotyped}")
    logger.info(f"Annotated VCF written to: {output_vcf}")


NUMT_ARGUMENTS = ("numt_intervals", "numt_nuclear_intervals", "input_alignments", "reference")


def main():
    parser = argparse.ArgumentParser(
        description="Soft-flag NUMT bleed and fix haploid genotypes in a mitochondrial callset"
    )
    parser.add_argument("input_vcf", help="Input VCF file")
    parser.add_argument("output_vcf", help="Output VCF file (bgzipped and tabix-indexed)")
    parser.add_argument(
        "--numt_intervals",
        help="chrM-side bed of regions with a nuclear NUMT paralog. Its first line must be the VCF "
        "##INFO=<ID=...> header, whose ID becomes the INFO tag and the FILTER name. Supplying this "
        f"enables NUMT annotation and requires the rest of the group: {', '.join(NUMT_ARGUMENTS)}",
    )
    parser.add_argument(
        "--numt_nuclear_intervals",
        help="Nuclear-side bed of the NUMT loci paired with --numt_intervals. A read's supplementary "
        "alignment target is tested against these intervals",
    )
    parser.add_argument(
        "--input_alignments",
        nargs="+",
        help="CRAM/BAM files the VCF was called from, used to read SA tags",
    )
    parser.add_argument("--reference", help="Reference fasta, required to decode CRAM")
    parser.add_argument(
        "--vaf_ceiling",
        type=float,
        default=0.90,
        help="Records above this read-derived VAF are never flagged, since NUMT bleed is capped by the "
        "nuclear:mtDNA copy ratio (default: 0.90)",
    )
    parser.add_argument(
        "--sa_excess",
        type=float,
        default=0.02,
        help="Minimal excess of the NUMT supplementary-alignment rate in alt-supporting reads over "
        "ref-supporting reads at the same site. The default sits ~25x above the alt-vs-ref difference "
        "seen at real variants but is not calibrated against known NUMT-driven calls; raise it if "
        "low-VAF records are wrongly flagged (default: 0.02)",
    )
    parser.add_argument(
        "--regenotype_contig",
        help="Contig to regenotype from FORMAT/VAF, e.g. chrM. Supplying this rewrites GT and INFO/AF "
        "on that contig only, because the diploid caller cannot express a haploid homoplasmic "
        "genotype there. Omit to leave genotypes alone",
    )
    parser.add_argument(
        "--homoplasmy_vaf",
        type=float,
        default=0.85,
        help="An alternate allele at or above this FORMAT/VAF is genotyped homozygous on "
        "--regenotype_contig. Anywhere in 0.75-0.85 gives the same concordance with an orthogonal "
        "caller on the development cohort (default: 0.85)",
    )

    args = parser.parse_args()

    supplied = [name for name in NUMT_ARGUMENTS if getattr(args, name)]
    if supplied and len(supplied) != len(NUMT_ARGUMENTS):
        missing = [f"--{name}" for name in NUMT_ARGUMENTS if not getattr(args, name)]
        parser.error(f"NUMT annotation needs the whole argument group; missing: {', '.join(missing)}")
    if not supplied and args.regenotype_contig is None:
        parser.error(
            f"nothing to do: pass the NUMT group ({', '.join('--' + name for name in NUMT_ARGUMENTS)}), "
            "--regenotype_contig, or both"
        )

    postprocess_mitochondria(
        input_vcf=args.input_vcf,
        output_vcf=args.output_vcf,
        numt_intervals_file=args.numt_intervals,
        numt_nuclear_intervals_file=args.numt_nuclear_intervals,
        input_alignments=args.input_alignments,
        reference=args.reference,
        vaf_ceiling=args.vaf_ceiling,
        sa_excess=args.sa_excess,
        regenotype_contig=args.regenotype_contig,
        homoplasmy_vaf=args.homoplasmy_vaf,
    )


if __name__ == "__main__":
    main()
