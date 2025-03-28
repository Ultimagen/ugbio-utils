import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class OutputFiles(Enum):
    H5 = "scRNA.applicationQC.h5"
    HTML_REPORT = "scRNA.applicationQC.html"
    NOTEBOOK = "single_cell_qc_report.ipynb"
    CBC_UMI_PLOT = "cbc_umi_plot.png"
    MEAN_INSERT_QUALITY_PLOT = "mean_insert_quality_plot.png"
    QUALITY_PER_POSITION_PLOT = "quality_per_position_plot.png"
    INSERT_LENGTH_HISTOGRAM = "insert_length_histogram.png"


BASE_PATH = Path(__file__).parent  # should be: src/single_cell/ugbio_single_cell
TEMPLATE_NOTEBOOK = BASE_PATH / "reports" / OutputFiles.NOTEBOOK.value


@dataclass
class Inputs:
    trimmer_stats_csv: str
    trimmer_histogram_csv: list[str] | None
    trimmer_failure_codes_csv: str
    sorter_stats_csv: str
    star_stats: str
    star_reads_per_gene: str
    insert: str
    sorter_stats_json: str | None = None

    def __post_init__(self):
        for _, value in self.__dict__.items():
            if isinstance(value, list):
                for item in value:
                    if not os.path.isfile(item):
                        raise FileNotFoundError(f"{item} not found")
            elif value and not os.path.isfile(value):
                raise FileNotFoundError(f"{value} not found")


@dataclass
class Thresholds:
    pass_trim_rate: float  # minimal %trimmed
    read_length: int  # expected read length
    fraction_below_read_length: float  # fraction of reads below read length
    percent_aligned: float  # minimal % of reads aligned


class H5Keys(Enum):
    TRIMMER_STATS = "trimmer_stats"
    TRIMMER_FAILURE_CODES = "trimmer_failure_codes"
    TRIMMER_HISTOGRAM = "trimmer_histogram"
    SORTER_STATS = "sorter_stats"
    SORTER_STATS_JSON = "sorter_stats_json"
    STAR_STATS = "star_stats"
    STAR_READS_PER_GENE = "star_reads_per_gene"
    INSERT_QUALITY = "insert_quality"
    INSERT_LENGTHS = "insert_lengths"
    STATISTICS_SHORTLIST = "statistics_shortlist"
