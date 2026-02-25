import subprocess
from dataclasses import dataclass
from pathlib import Path

import pysam
from ugbio_core.logger import logger

from ugbio_featuremap.featuremap_utils import (
    FeatureMapFields,
    TandemRepeatFields,
    VcfInfoField,
)


# =============================================================================
# CONFIGURATIONS
# =============================================================================
@dataclass(frozen=True)
class TandemRepeatConfig:
    """Configuration for Tandem Repeat INFO fields added to VCF during TR annotation.

    Use TandemRepeatFields enum for field name access.
    This class provides VCF metadata (type, description) and helper methods.
    """

    info_fields: tuple[VcfInfoField, ...] = (
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

    def get_all_pileup_format_fields(self) -> set[str]:
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


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
_MAX_DEBUG_REGION_LINES = 10


def _log_regions_bed_preview(regions_bed_file: Path) -> None:
    """Log a limited preview of regions BED file for debug."""
    preview: list[str] = []
    total = 0
    with open(regions_bed_file) as f:
        for line in f:
            total += 1
            if len(preview) < _MAX_DEBUG_REGION_LINES:
                preview.append(line.strip())
    if preview:
        msg = f"Regions BED preview (first {len(preview)} of {total} regions):"
        logger.debug(f"{msg}\n" + "\n".join(preview))


def cleanup_intermediate_files(file_paths: list[Path]) -> None:
    """Remove intermediate files if they exist, logging any errors."""
    for file_path in file_paths:
        file_path.unlink(missing_ok=True)
        logger.debug(f"Cleaned up intermediate file: {file_path}")


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


def write_vcf_info_header_file(
    info_fields: list[VcfInfoField], header_file: Path, additional_header_lines: list[str] | None = None
) -> None:
    """Write a VCF header file with INFO field definitions and optional additional header lines.

    Creates a header file suitable for use with bcftools annotate -h option.
    Can include both INFO/FORMAT field definitions and arbitrary custom header lines.

    Parameters
    ----------
    info_fields : list[VcfInfoField]
        List of VcfInfoField objects defining the INFO fields to add.
    header_file : Path
        Path to the output header file.
    additional_header_lines : list[str], optional
        List of additional header lines to append (e.g., ["##tumor_sample=<sample_name>"]).
        These lines will be written as-is after the INFO field definitions.
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

        if additional_header_lines:
            for line in additional_header_lines:
                formatted_line = line if line.startswith("##") else f"##{line}"
                f.write(formatted_line + "\n")


def get_sample_names_from_vcf(vcf_path: Path) -> tuple[str, str]:
    """Get tumor and normal sample names from VCF file.

    Convention: index 0 = tumor, index 1 = normal.

    Parameters
    ----------
    vcf_path : Path
        Path to the VCF file.

    Returns
    -------
    tuple[str, str]
        Tuple of (tumor_sample_name, normal_sample_name).
    """
    with pysam.VariantFile(str(vcf_path)) as vcf:
        samples = list(vcf.header.samples)
        if len(samples) < 2:  # noqa: PLR2004
            raise ValueError(f"Expected at least 2 samples in VCF, found {len(samples)}: {samples}")
        return samples[0], samples[1]


# =============================================================================
# REQUIRED COLUMNS FOR ML INFERENCE
# These are the columns needed from the VCF to compute the model's expected features.
# =============================================================================

# INFO fields required for inference (includes TR fields added by annotation step)
# X_PREV1, X_PREV2, X_NEXT1, X_NEXT2 are needed for ref/nonref calculations
REQUIRED_INFO_FIELDS: set[str] = {
    TandemRepeatFields.TR_DISTANCE.value,
    TandemRepeatFields.TR_LENGTH.value,
    TandemRepeatFields.TR_SEQ_UNIT_LENGTH.value,
    FeatureMapFields.X_PREV1.value,
    FeatureMapFields.X_PREV2.value,
    FeatureMapFields.X_NEXT1.value,
    FeatureMapFields.X_NEXT2.value,
}

# FORMAT fields required for inference (per-sample fields)
# These are used directly or for deriving aggregated features
REQUIRED_FORMAT_FIELDS: set[str] = {
    FeatureMapFields.DP.value,  # Read depth -> t_dp, n_dp
    FeatureMapFields.DP_FILT.value,  # Read depth of reads that pass filters -> t_dp_filt, n_dp_filt
    FeatureMapFields.VAF.value,  # Variant allele frequency -> t_vaf, n_vaf
    FeatureMapFields.RAW_VAF.value,  # Raw VAF -> t_raw_vaf, n_raw_vaf
    FeatureMapFields.AD.value,  # Allelic depths -> AD_1 for alt_reads
    FeatureMapFields.MQUAL.value,  # Mapping quality per read -> mean/min/max aggregations
    FeatureMapFields.SNVQ.value,  # SNV quality per read -> mean/min/max aggregations
    FeatureMapFields.MAPQ.value,  # Mapping quality (for count_zero -> map0_count)
    FeatureMapFields.EDIST.value,  # Edit distance -> mean/min/max aggregations
    FeatureMapFields.RL.value,  # Read length -> mean/min/max aggregations
    FeatureMapFields.DUP.value,  # Duplicate flag -> count_duplicate, count_non_duplicate
    FeatureMapFields.REV.value,  # Reverse strand flag -> reverse_count, forward_count
    FeatureMapFields.FILT.value,  # Filter flag -> pass_alt_reads (FILT count non-zero)
    FeatureMapFields.SCST.value,  # Soft clip start -> scst_num_reads (count non-zero)
    FeatureMapFields.SCED.value,  # Soft clip end -> sced_num_reads (count non-zero)
    # PILEUP columns for ref0-4 / nonref0-4 calculations
    *PILEUP_CONFIG.get_all_pileup_format_fields(),
}

# Sample prefixes for tumor (index 0) and normal (index 1)
TUMOR_PREFIX = "t_"
NORMAL_PREFIX = "n_"

# XGBoost probability INFO field definition
XGB_PROBA_INFO_FIELD = VcfInfoField(
    FeatureMapFields.XGB_PROBA.value, "1", "Float", "XGBoost model predicted probability"
)
