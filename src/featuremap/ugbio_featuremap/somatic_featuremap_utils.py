import tempfile
from dataclasses import dataclass
from pathlib import Path

import pysam
from simppl.simple_pipeline import SimplePipeline
from ugbio_core.logger import logger
from ugbio_core.vcf_utils import VcfUtils

from ugbio_featuremap.featuremap_utils import FeatureMapFields


@dataclass(frozen=True)
class VcfInfoField:
    """VCF INFO field definition."""

    field_id: str
    number: str
    field_type: str
    description: str


@dataclass(frozen=True)
class TandemRepeatConfig:
    """Configuration for Tandem Repeat INFO fields added to VCF during TR annotation."""

    fields: tuple[VcfInfoField, ...] = (
        VcfInfoField("TR_START", "1", "Integer", "Closest tandem Repeat Start"),
        VcfInfoField("TR_END", "1", "Integer", "Closest Tandem Repeat End"),
        VcfInfoField("TR_SEQ", "1", "String", "Closest Tandem Repeat Sequence"),
        VcfInfoField("TR_LENGTH", "1", "Integer", "Closest Tandem Repeat total length"),
        VcfInfoField("TR_SEQ_UNIT_LENGTH", "1", "Integer", "Closest Tandem Repeat unit length"),
        VcfInfoField("TR_DISTANCE", "1", "Integer", "Closest Tandem Repeat Distance"),
    )

    def all_field_ids(self) -> set[str]:
        """Return set of all TR field IDs."""
        return {field.field_id for field in self.fields}

    @property
    def length_field_id(self) -> str:
        """Return the TR_LENGTH field ID."""
        return "TR_LENGTH"

    @property
    def unit_length_field_id(self) -> str:
        """Return the TR_SEQ_UNIT_LENGTH field ID."""
        return "TR_SEQ_UNIT_LENGTH"

    @property
    def distance_field_id(self) -> str:
        """Return the TR_DISTANCE field ID."""
        return "TR_DISTANCE"


@dataclass(frozen=True)
class PileupConfig:
    """Configuration for PILEUP-based ref/nonref calculations.

    PILEUP position mapping: L2→ref0, L1→ref1, C→ref2, R1→ref3, R2→ref4
    Reference column mapping: L2→X_PREV2, L1→X_PREV1, C→REF, R1→X_NEXT1, R2→X_NEXT2
    """

    positions: tuple[str, ...] = ("L2", "L1", "C", "R1", "R2")
    bases: tuple[str, ...] = ("A", "C", "G", "T")
    indels: tuple[str, ...] = ("DEL", "INS")

    def get_reference_column(self, position: str) -> str:
        """Get the reference column name for a given position.

        Parameters
        ----------
        position : str
            PILEUP position (L2, L1, C, R1, or R2).

        Returns
        -------
        str
            Reference column name (X_PREV2, X_PREV1, REF, X_NEXT1, or X_NEXT2).

        Raises
        ------
        ValueError
            If position is not recognized.
        """
        position_to_ref_col = {
            "L2": FeatureMapFields.X_PREV2.value,
            "L1": FeatureMapFields.X_PREV1.value,
            "C": FeatureMapFields.REF.value,
            "R1": FeatureMapFields.X_NEXT1.value,
            "R2": FeatureMapFields.X_NEXT2.value,
        }
        if position not in position_to_ref_col:
            raise ValueError(f"Unknown position: {position}. Expected one of {self.positions}")
        return position_to_ref_col[position]

    def get_column_name(self, element: str, position: str, sample_suffix: str) -> str:
        """Get the full PILEUP column name."""
        return f"PILEUP_{element}_{position}{sample_suffix}"

    def get_all_format_fields(self) -> set[str]:
        """Get all PILEUP FORMAT field names (without sample suffix)."""
        fields = set()
        for pos in self.positions:
            for base in self.bases:
                fields.add(f"PILEUP_{base}_{pos}")
            for indel in self.indels:
                fields.add(f"PILEUP_{indel}_{pos}")
        return fields


# Singleton instances for use throughout the codebase
TR_CONFIG = TandemRepeatConfig()
PILEUP_CONFIG = PileupConfig()


def _create_variant_bed(merged_vcf: Path, bed_file: Path) -> None:
    """Create a BED file from VCF variants."""
    sp = SimplePipeline(0, 1)
    cmd_bcftools = f"bcftools query -f '%CHROM\t%POS0\t%END\n' {merged_vcf}"
    sp.print_and_run(cmd_bcftools, out=str(bed_file))


def _find_closest_tandem_repeats(bed1: Path, bed2: Path, genome_file: Path, output_file: Path) -> None:
    """Find closest tandem repeats for each variant."""
    sp = SimplePipeline(0, 1)
    cmd = f"bedtools closest -D ref -g {genome_file} -a {bed1} -b {bed2} | cut -f1,3,5-10"
    sp.print_and_run(cmd, out=str(output_file))


def _prepare_annotation_files(tmpdir: Path, tr_tsv_file: Path) -> tuple[Path, Path]:
    """Prepare files for VCF annotation."""
    sp = SimplePipeline(0, 3)
    sorted_tsv = tmpdir / "tmp.TRdata.sorted.tsv"
    cmd = f"sort -k1,1 -k2,2n {tr_tsv_file}"
    sp.print_and_run(cmd, out=str(sorted_tsv))

    gz_tsv = sorted_tsv.with_suffix(".tsv.gz")
    cmd = f"bgzip -c {sorted_tsv}"
    sp.print_and_run(cmd, out=str(gz_tsv))

    cmd = f"tabix -s 1 -b 2 -e 2 {gz_tsv}"
    sp.print_and_run(cmd)

    header = pysam.VariantHeader()
    for tr_field in TR_CONFIG.fields:
        header.add_meta(
            "INFO",
            items=[
                ("ID", tr_field.field_id),
                ("Number", tr_field.number),
                ("Type", tr_field.field_type),
                ("Description", tr_field.description),
            ],
        )

    hdr_file = tmpdir / "tr_hdr.txt"
    with open(hdr_file, "w") as f:
        lines = str(header).splitlines()
        for line in lines[1:-1]:
            f.write(line + "\n")

    return gz_tsv, hdr_file


def filter_and_annotate_tr(
    input_vcf: Path,
    ref_tr_file: Path,
    genome_file: Path,
    out_dir: Path,
    filter_string: str | None = "PASS",
) -> Path:
    """
    Filter VCF and annotate with tandem repeat features in a single pass.

    This unified preprocessing function:
    1. Filters the VCF to keep only specified variants (e.g., PASS)
    2. Annotates the filtered variants with tandem repeat information

    Parameters
    ----------
    input_vcf : Path
        Path to the input VCF file (gzipped).
    ref_tr_file : Path
        Path to the reference tandem repeat BED file.
    genome_file : Path
        Path to the reference genome FASTA index file (.fai).
    out_dir : Path
        Output directory for the processed VCF file.
    filter_string : str, optional
        FILTER value to keep (e.g., "PASS"). If None, no filtering is applied.
        Defaults to "PASS".

    Returns
    -------
    Path
        Path to the output VCF file with FILTER applied and TR annotations added.
        The output file will have '.filtered.tr_info.vcf.gz' suffix.
    """
    vcf_utils = VcfUtils()

    with tempfile.TemporaryDirectory(dir=out_dir) as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Step 1: Filter VCF (if filter_string is provided)
        if filter_string:
            logger.info(f"Filtering VCF to keep variants with FILTER={filter_string}")
            filtered_vcf = tmpdir_path / input_vcf.name.replace(".vcf.gz", ".filtered.vcf.gz")
            extra_args = f"-f {filter_string}"
            vcf_utils.view_vcf(str(input_vcf), str(filtered_vcf), n_threads=1, extra_args=extra_args)
            vcf_utils.index_vcf(str(filtered_vcf))
            vcf_to_annotate = filtered_vcf
        else:
            vcf_to_annotate = input_vcf

        # Step 2: Create variant BED file from filtered VCF
        logger.debug("Creating variant BED file for TR annotation")
        bed_file = tmpdir_path / "variants.bed"
        _create_variant_bed(vcf_to_annotate, bed_file)

        # Step 3: Find closest tandem repeats
        logger.debug("Finding closest tandem repeats")
        tr_tsv = tmpdir_path / "variants.tr_data.tsv"
        _find_closest_tandem_repeats(bed_file, ref_tr_file, genome_file, tr_tsv)

        # Step 4: Prepare annotation files (sort, bgzip, tabix)
        logger.debug("Preparing TR annotation files")
        gz_tsv, hdr_file = _prepare_annotation_files(tmpdir_path, tr_tsv)

        # Step 5: Annotate VCF with TR fields
        logger.info("Annotating VCF with tandem repeat information")
        output_suffix = ".filtered.tr_info.vcf.gz" if filter_string else ".tr_info.vcf.gz"
        output_vcf = out_dir / input_vcf.name.replace(".vcf.gz", output_suffix)
        vcf_utils.annotate_vcf(
            input_vcf=str(vcf_to_annotate),
            output_vcf=str(output_vcf),
            annotation_file=str(gz_tsv),
            header_file=str(hdr_file),
            columns="CHROM,POS,INFO/TR_START,INFO/TR_END,INFO/TR_SEQ,INFO/TR_LENGTH,INFO/TR_SEQ_UNIT_LENGTH,INFO/TR_DISTANCE",
        )

    logger.info(f"Filtered and TR-annotated VCF written to: {output_vcf}")
    return output_vcf
