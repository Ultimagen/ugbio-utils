import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pysam
from ugbio_core.logger import logger
from ugbio_core.vcf_utils import VcfUtils

from ugbio_featuremap.featuremap_utils import (
    FeatureMapFields,
    TandemRepeatFields,
    VcfInfoField,
)


@dataclass(frozen=True)
class TandemRepeatConfig:
    """Configuration for Tandem Repeat INFO fields added to VCF during TR annotation.

    Use TandemRepeatFields enum for field name access.
    This class provides VCF metadata (type, description) and helper methods.
    """

    fields: tuple[VcfInfoField, ...] = (
        VcfInfoField(TandemRepeatFields.TR_START.value, "1", "Integer", "Closest tandem Repeat Start"),
        VcfInfoField(TandemRepeatFields.TR_END.value, "1", "Integer", "Closest Tandem Repeat End"),
        VcfInfoField(TandemRepeatFields.TR_SEQ.value, "1", "String", "Closest Tandem Repeat Sequence"),
        VcfInfoField(TandemRepeatFields.TR_LENGTH.value, "1", "Integer", "Closest Tandem Repeat total length"),
        VcfInfoField(TandemRepeatFields.TR_SEQ_UNIT_LENGTH.value, "1", "Integer", "Closest Tandem Repeat unit length"),
        VcfInfoField(TandemRepeatFields.TR_DISTANCE.value, "1", "Integer", "Closest Tandem Repeat Distance"),
    )

    def get_bcftools_annotate_columns(self) -> str:
        """Return the columns string for bcftools annotate.

        Format: CHROM,POS,INFO/TR_START,INFO/TR_END,...
        """
        info_cols = ",".join(f"INFO/{field.value}" for field in TandemRepeatFields)
        return f"{FeatureMapFields.CHROM.value},{FeatureMapFields.POS.value},{info_cols}"


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


def _run_shell_command(cmd: str, output_file: Path | None = None) -> None:
    """Run a shell command, optionally redirecting stdout to a file.

    Uses shell=True because commands may contain pipes (e.g., "bedtools ... | cut ...").
    All command strings are constructed internally - no user input is passed directly.
    """
    logger.debug(f"Running: {cmd}")
    if output_file:
        with open(output_file, "w") as f:
            subprocess.run(cmd, shell=True, check=True, stdout=f)  # noqa: S602
    else:
        subprocess.run(cmd, shell=True, check=True)  # noqa: S602


def write_vcf_info_header_file(info_fields: list[VcfInfoField], header_file: Path) -> None:
    """Write a VCF header file with INFO field definitions.

    Creates a header file suitable for use with bcftools annotate -h option.

    Parameters
    ----------
    info_fields : list[VcfInfoField]
        List of VcfInfoField objects defining the INFO fields to add.
    header_file : Path
        Path to the output header file.
    """
    header = pysam.VariantHeader()
    for field in info_fields:
        header.add_meta(
            "INFO",
            items=[
                ("ID", field.field_id),
                ("Number", field.number),
                ("Type", field.field_type),
                ("Description", field.description),
            ],
        )

    with open(header_file, "w") as f:
        lines = str(header).splitlines()
        for line in lines[1:-1]:
            f.write(line + "\n")


def _create_tr_annotation_file(
    input_vcf: Path, ref_tr_file: Path, genome_file: Path, tmpdir: Path
) -> tuple[Path, Path]:
    """Create TR annotation file from VCF in one piped command.

    Pipeline: bcftools query -> bedtools closest -> cut -> sort -> bgzip
    Then: tabix for indexing
    """
    gz_tsv = tmpdir / "tr_annotation.tsv.gz"
    cmd = (
        f"bcftools query -f '%CHROM\\t%POS0\\t%END\\n' {input_vcf} | "
        f"bedtools closest -D ref -g {genome_file} -a stdin -b {ref_tr_file} | "
        f"cut -f1,3,5-10 | "
        f"sort -k1,1 -k2,2n | "
        f"bgzip -c"
    )
    _run_shell_command(cmd, gz_tsv)

    cmd = f"tabix -s 1 -b 2 -e 2 {gz_tsv}"
    _run_shell_command(cmd)

    # Create header file for bcftools annotate
    hdr_file = tmpdir / "tr_hdr.txt"
    write_vcf_info_header_file(list(TR_CONFIG.fields), hdr_file)

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

        # Step 2: Create TR annotation file (fully piped: bcftools -> bedtools -> cut -> sort -> bgzip)
        logger.info(f"Creating TR annotation file for {vcf_to_annotate}")
        gz_tsv, hdr_file = _create_tr_annotation_file(vcf_to_annotate, ref_tr_file, genome_file, tmpdir_path)

        # Step 3: Annotate VCF with TR fields
        logger.info("Annotating VCF with tandem repeat information")
        output_suffix = ".filtered.tr_info.vcf.gz" if filter_string else ".tr_info.vcf.gz"
        output_vcf = out_dir / input_vcf.name.replace(".vcf.gz", output_suffix)
        vcf_utils.annotate_vcf(
            input_vcf=str(vcf_to_annotate),
            output_vcf=str(output_vcf),
            annotation_file=str(gz_tsv),
            header_file=str(hdr_file),
            columns=TR_CONFIG.get_bcftools_annotate_columns(),
        )

    logger.info(f"Filtered and TR-annotated VCF written to: {output_vcf}")
    return output_vcf
