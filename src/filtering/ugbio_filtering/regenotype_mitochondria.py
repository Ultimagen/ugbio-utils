"""Fix haploid genotypes on a mitochondrial contig, from FORMAT/VAF.

The caller runs a diploid model on a haploid, multi-copy contig, so its ``1/1`` state is
effectively unreachable there: over a 7-sample cohort every one of 207 chrM PASS calls came out
``0/1`` and not one ``1/1``, including all 182 records an orthogonal caller and our own reads
both call homoplasmic.  ``INFO/AF`` is derived from the genotype and therefore reads a constant
0.5 on every record.  ``FORMAT/VAF`` and ``FORMAT/AD``, by contrast, agree with the orthogonal
caller to ~0.005, so the information is already in the record and only the genotype summary
needs rewriting.

On the requested contig, an alternate allele at ``FORMAT/VAF >= homoplasmy_vaf`` therefore
becomes homozygous and ``INFO/AF`` is recomputed from the observed VAF.  The default 0.85 sits
in the middle of a plateau: swept over the 192 alleles both callers report on that cohort, it
leaves 3 genotypes disagreeing with the orthogonal caller anywhere in 0.75-0.85, then 4 at 0.90,
8 at 0.95 and 142 at 0.99, against 182 disagreements before the correction.  The 3 that remain
are heteroplasmies at AF 0.81-0.90, where the hom/het label is a matter of convention and no
threshold helps.

``FORMAT/PL``, ``FORMAT/GQ`` and ``QUAL`` are deliberately left as the model emitted them:
they carry no usable information on this contig (a site with 1183 alt reads out of 1192 scored
``PL=0,0,33``, ref and het exactly tied), and rewriting them would mean substituting a
contig-specific likelihood model rather than relabelling what the caller already measured.  So
``PL`` can disagree with ``GT`` on a regenotyped record; the ``MT_REGT`` header says so.  The
genotype as emitted is preserved in ``FORMAT/OGT``.

Nothing here reads the alignments; NUMT bleed is a separate tool, ``annotate_numt``.
"""

import argparse

import pysam
from ugbio_core.logger import logger

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


def regenotype_mitochondria(input_vcf: str, output_vcf: str, contig: str, homoplasmy_vaf: float = 0.85):
    """Rewrite GT and INFO/AF from FORMAT/VAF, on one contig only.

    Every input record is written out; nothing is dropped, and no record on any other contig is
    modified.
    """
    vcf_in = pysam.VariantFile(input_vcf)
    header = vcf_in.header
    # INFO/AF and the genotype are per-sample decisions read off one sample's VAF, so a
    # multi-sample VCF would need a per-sample INFO/AF that the format cannot express.
    if len(header.samples) != 1:
        raise ValueError(f"regenotyping needs a single-sample VCF, {input_vcf} has {len(header.samples)} samples")
    if "OGT" not in header.formats:
        header.add_line(OGT_HEADER)
    if "MT_REGT" not in header.info:
        header.add_line(MT_REGT_HEADER)

    logger.info(f"Regenotyping {contig} records with FORMAT/VAF >= {homoplasmy_vaf}")
    logger.info(f"Processing {input_vcf} and writing to {output_vcf}")
    vcf_out = pysam.VariantFile(output_vcf, "wz", header=header)
    n_on_contig = 0
    n_regenotyped = 0
    for record in vcf_in:
        # The contig guard is the only thing standing between this and a genome-wide GT rewrite.
        if record.contig == contig:
            n_on_contig += 1
            if regenotype_record(record, homoplasmy_vaf):
                n_regenotyped += 1
        vcf_out.write(record)

    vcf_out.close()
    vcf_in.close()

    pysam.tabix_index(output_vcf, preset="vcf", force=True)
    logger.info(f"Records on {contig}: {n_on_contig}, genotypes reassigned: {n_regenotyped}")
    logger.info(f"Regenotyped VCF written to: {output_vcf}")


def main():
    parser = argparse.ArgumentParser(
        description="Rewrite GT and INFO/AF from FORMAT/VAF on a haploid mitochondrial contig"
    )
    parser.add_argument("input_vcf", help="Input VCF file")
    parser.add_argument("output_vcf", help="Output VCF file (bgzipped and tabix-indexed)")
    parser.add_argument(
        "--contig",
        required=True,
        help="Contig to regenotype, e.g. chrM. GT and INFO/AF are rewritten on this contig only, "
        "because the diploid caller cannot express a haploid homoplasmic genotype there",
    )
    parser.add_argument(
        "--homoplasmy_vaf",
        type=float,
        default=0.85,
        help="An alternate allele at or above this FORMAT/VAF is genotyped homozygous on --contig. "
        "Anywhere in 0.75-0.85 gives the same concordance with an orthogonal caller on the development "
        "cohort (default: 0.85)",
    )

    args = parser.parse_args()

    regenotype_mitochondria(
        input_vcf=args.input_vcf,
        output_vcf=args.output_vcf,
        contig=args.contig,
        homoplasmy_vaf=args.homoplasmy_vaf,
    )


if __name__ == "__main__":
    main()
