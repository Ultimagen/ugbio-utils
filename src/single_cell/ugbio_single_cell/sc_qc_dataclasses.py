import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import pandas as pd
from ugbio_core.logger import logger


class OutputFiles(Enum):
    H5 = "scRNA.applicationQC.h5"
    HTML_REPORT = "scRNA.applicationQC.html"
    NOTEBOOK = "single_cell_qc_report.ipynb"
    CBC_UMI_PLOT = "cbc_umi_plot.png"
    MEAN_INSERT_QUALITY_PLOT = "mean_insert_quality_plot.png"
    QUALITY_PER_POSITION_PLOT = "quality_per_position_plot.png"
    INSERT_LENGTH_HISTOGRAM = "insert_length_histogram.png"

BASE_PATH = Path(__file__).parent # should be: src/single_cell/ugbio_single_cell
TEMPLATE_NOTEBOOK = BASE_PATH / "reports" / OutputFiles.NOTEBOOK.value


@dataclass
class Inputs:
    trimmer_stats_csv: str
    trimmer_histogram_csv: list[str]
    trimmer_failure_codes_csv: str
    sorter_stats_csv: str
    star_stats: str
    star_reads_per_gene: str
    insert: str
    insert_quality_csv: str = None
    insert_lengths_csv: str = None
    output_path: Path = None
    sample_name: str = None

    def __post_init__(self):
        for _, value in self.__dict__.items():
            if isinstance(value, list):
                for item in value:
                    assert os.path.isfile(item), f"{item} not found"
            elif value:
                assert os.path.isfile(value), f"{value} not found"
    
    def update_file_merged_histogram(self, merged_histogram_csv):
        logger.info(f"Updating merged trimmer histogram file: {merged_histogram_csv}")
        self.trimmer_histogram_csv = [merged_histogram_csv]
        
    def update_file_trimmer_failure_codes(self)->str:
        new_file = self.output_path / (self.sample_name + "updated_trimmer_failure_codes.csv")
        logger.info(f"Updating trimmer failure codes file: {new_file}")
        self.trimmer_failure_codes_csv = new_file
        return new_file
    
    def update_file_sorter_stats(self)->str:
        new_file = self.output_path / (self.sample_name + "updated_sorter_stats.csv")
        logger.info(f"Updating sorter stats file: {new_file}")
        self.sorter_stats_csv = new_file
        return new_file
    
    def update_file_star_stats(self)->str:
        new_file = self.output_path / (self.sample_name + "updated_star_stats.csv")
        logger.info(f"Updating STAR stats file: {new_file}")
        self.star_stats = new_file
        return new_file
    
    def save_insert_quality_csv(self, df_insert_quality: pd.DataFrame):
        new_file = self.output_path / (self.sample_name + "insert_quality.csv")
        logger.info(f"Saving insert quality to file: {new_file}")
        df_insert_quality.to_csv(new_file, index=False)
        self.insert_quality_csv = new_file
    
    def save_insert_lengths_csv(self, insert_lengths: list[int]):
        new_file = self.output_path / (self.sample_name + "insert_lengths.csv")
        logger.info(f"Saving insert lengths to file: {new_file}")
        s = pd.Series(insert_lengths)
        s.to_csv(new_file, index=False)
        self.insert_lengths_csv = new_file


@dataclass
class Thresholds:
    pass_trim_rate: float  # minimal %trimmed
    read_length: int  # expected read length
    fraction_below_read_length: float  # fraction of reads below read length
    percent_aligned: float  # minimal % of reads aligned


class H5Keys(Enum):
    # TODO: remove keys
    TRIMMER_STATS = "trimmer_stats"
    TRIMMER_FAILURE_CODES = "trimmer_failure_codes"
    TRIMMER_HISTOGRAM = "trimmer_histogram"
    SORTER_STATS = "sorter_stats"
    STAR_STATS = "star_stats"
    STAR_READS_PER_GENE = "star_reads_per_gene"
    INSERT_QUALITY = "insert_quality"
    INSERT_LENGTHS = "insert_lengths"
    STATISTICS_SHORTLIST = "statistics_shortlist"
