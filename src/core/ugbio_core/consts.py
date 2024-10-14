from enum import Enum

from pandas.api.types import CategoricalDtype


class FileExtension(Enum):
    """File Extension enum"""

    PARQUET = ".parquet"
    HDF = ".hdf"
    H5 = ".h5"
    CSV = ".csv"
    TSV = ".tsv"
    BAM = ".bam"
    CRAM = ".cram"
    PNG = ".png"
    JPEG = ".jpeg"
    FASTA = ".fasta"
    TXT = ".txt"
    VCF = ".vcf"
    VCF_GZ = ".vcf.gz"


DEFAULT_FLOW_ORDER = "TGCA"
CHROM_DTYPE = CategoricalDtype(
    categories=[f"chr{j}" for j in range(1, 23)] + ["chrX", "chrY", "chrM"],
    ordered=True,
)

# cycle skip
CYCLE_SKIP_STATUS = "cycle_skip_status"
CYCLE_SKIP = "cycle-skip"
POSSIBLE_CYCLE_SKIP = "possible-cycle-skip"
NON_CYCLE_SKIP = "non-skip"
UNDETERMINED_CYCLE_SKIP = "NA"
IS_CYCLE_SKIP = "is_cycle_skip"
CYCLE_SKIP_DTYPE = CategoricalDtype(categories=[CYCLE_SKIP, POSSIBLE_CYCLE_SKIP, NON_CYCLE_SKIP], ordered=True)

# vcf
CHROM = "chrom"
POS = "pos"
REF = "ref"
ALT = "alt"
QUAL = "qual"
FILTER = "filter"
GT = "GT"
AD = "AD"
DP = "DP"
VAF = "VAF"
