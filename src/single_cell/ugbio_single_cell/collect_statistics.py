import gzip
from collections import defaultdict
from pathlib import Path

import pandas as pd
from Bio import SeqIO

from ugbio_single_cell.sc_qc_dataclasses import H5Keys, Inputs, OutputFiles
from ugbio_core.sorter_utils import read_sorter_statistics_csv
from ugbio_core.trimmer_utils import merge_trimmer_histograms, read_trimmer_failure_codes


def collect_statistics(input_files: Inputs, output_path: Path) -> Path:
    """
    Collect statistics from input files, parse and save them into h5 file

    Parameters
    ----------
    input_files : Inputs
        Input files containing the necessary data for statistics collection.
    output_path : str
        Path to the output directory.
    sample_name : str
        Sample name to be included as a prefix in the output files.

    Returns
    -------
    Path
        Path to the h5 file with statistics.
    """
    output_path.mkdir(parents=True, exist_ok=True)

    # Merge Trimmer histograms from two optical paths (if needed)
    merged_histogram_csv = merge_trimmer_histograms(
        input_files.trimmer_histogram_csv, output_path=str(output_path)
    )
    input_files.update_file_merged_histogram(merged_histogram_csv)

    # Read and save updated statistics
    df_trimmer_failure_codes = read_trimmer_failure_codes(
        input_files.trimmer_failure_codes_csv,
        add_total=True,
    )
    df_trimmer_failure_codes.to_csv(input_files.update_file_trimmer_failure_codes())

    sorter_stats = read_sorter_statistics_csv(input_files.sorter_stats_csv)
    sorter_stats.to_csv(input_files.update_file_sorter_stats())

    star_stats = read_star_stats(input_files.star_stats)
    star_stats.to_csv(input_files.update_file_star_stats())

    # Get insert subsample quality and lengths and save to files
    insert_quality, insert_lengths = get_insert_properties(input_files.insert)
    input_files.save_insert_quality_csv(insert_quality)
    input_files.save_insert_lengths_csv(insert_lengths)


def read_star_stats(star_stats_file: str) -> pd.Series:
    """
    Read STAR stats file (Log.final.out) and return parsed pandas object

    Parameters
    ----------
    star_stats_file : str
        Path to the STAR stats file.

    Returns
    -------
    pd.Series
        Series with parsed STAR stats.
    """
    df = pd.read_csv(star_stats_file, header=None, sep="\t")
    df.columns = ["metric", "value"]

    # parse metric description
    df["metric"] = df["metric"].str.replace("|", "").str.strip().str.replace(" ", "_")
    df.loc[:, "metric"] = df["metric"].str.replace(",", "").str.replace(":", "")

    # Add type (general, unique_reads, multi_mapping_reads, unmapped_reads, chimeric_reads)
    df.loc[:, "type"] = (
        df["metric"]
        .where(df["value"].isnull())
        .ffill()
        .fillna("general")
        .str.lower()
        .str.replace(":", "")
    )
    df = df.dropna(subset=["value"])

    # Add "pct_" to metric name if value ends with "%" (and remove "%" from the name)
    df.loc[df["value"].str.endswith("%"), "metric"] = df.loc[df["value"].str.endswith("%"), "metric"].apply(
        lambda x: "pct_" + x.replace("_%", "").replace("%_", "")
    )

    # Remove "%" from value
    df["value"] = df["value"].str.replace("%", "")

    # Set index
    df.set_index(['type', 'metric'], inplace=True)
    # convert df to pd.series for easier access
    s = df['value']

    # convert types
    s = s.apply(convert_value)

    return s

def convert_value(value):
    """
    Convert value to numeric or datetime if possible.
    """
    try:
        # Try to convert to numeric
        return pd.to_numeric(value)
    except ValueError:
        try:
            # Try to convert to datetime
            return pd.to_datetime(value)
        except ValueError:
            # If both conversions fail, return the original value
            return value

def get_insert_properties(insert, max_reads=None) -> tuple[pd.DataFrame, list[int]]:
    """
    Read insert subsample fastq.gz file and return quality scores per position and read lengths.

    Parameters
    ----------
    insert : str
        Path to the insert .fastq.gz file.
    max_reads : int, optional
        Maximum number of reads to process, by default None.

    Returns
    -------
    tuple[pd.DataFrame, list[int]]
        DataFrame with quality scores per position and list with read lengths.
    """
    insert_lengths = []
    counter = defaultdict(lambda: defaultdict(int))

    # read fastq file
    fastq_parser = SeqIO.parse(gzip.open(insert, "rt"), "fastq")

    for j, record in enumerate(fastq_parser):
        insert_lengths.append(len(record))
        score = record.letter_annotations["phred_quality"]
        for i in range(len(score)):
            counter[i + 1][score[i]] += 1
        if max_reads and j > max_reads:
            break

    df_insert_quality = pd.DataFrame(counter).sort_index()
    df_insert_quality.index.name = "quality"
    df_insert_quality.columns.name = "position"

    # normalize
    df_insert_quality = df_insert_quality / df_insert_quality.sum().sum()

    return df_insert_quality, insert_lengths


def extract_statistics_table(h5_file: Path, input_files: Inputs):
    """
    Create shortlist of statistics from h5 file and append it to h5 file.

    Parameters
    ----------
    h5_file : Path
        Path to the h5 file with statistics.
    """
    stats = {}

    # with pd.HDFStore(h5_file, "r") as store:
    # number of Input Reads
    df_trimmer_stats = pd.read_csv(input_files.trimmer_stats_csv)
    num_input_reads = df_trimmer_stats["num input reads"].values[0]
    stats["num_input_reads"] = num_input_reads

    # number of Trimmed reads

    num_trimmed_reads = df_trimmer_stats[
        "num trimmed reads"
    ].values[0]
    stats["num_trimmed_reads"] = num_trimmed_reads

    # pct_pass_trimmer
    pass_trimmer_rate = num_trimmed_reads / num_input_reads
    stats["pct_pass_trimmer"] = pass_trimmer_rate * 100

    # Mean UMI per cell
    mean_umi_per_cell = None  # TODO: waiting for the calculation details from Gila
    stats["mean_umi_per_cell"] = mean_umi_per_cell

    # Mean read length
    df_star_stats = pd.read_csv(input_files.star_stats, index_col=[0, 1])
    mean_read_length = int(df_star_stats.loc[('general','Average_input_read_length')]) + 1
    stats["mean_read_length"] = mean_read_length

    # %q >= 20 for insert
    df_sorter_stats = pd.read_csv(input_files.sorter_stats_csv, index_col=[0])
    q20 = df_sorter_stats.loc["PCT_PF_Q20_bases"].value
    stats["pct_q20"] = q20

    # %q >= 30 for insert
    q30 = df_sorter_stats.loc["PCT_PF_Q30_bases"].value
    stats["pct_q30"] = q30

    # %Aligned to genome
    ur_tmm = float(df_star_stats.loc[('unmapped_reads','pct_of_reads_unmapped_too_many_mismatches')])
    ur_ts = float(df_star_stats.loc[('unmapped_reads','pct_of_reads_unmapped_too_short')])
    ur_other = float(df_star_stats.loc[('unmapped_reads','pct_of_reads_unmapped_other')])
    pct_aligned_to_genome = 100 - ur_tmm - ur_ts - ur_other
    stats["pct_aligned_to_genome"] = pct_aligned_to_genome

    # %Assigned to genes (unique)
    df_star_reads_per_gene = pd.read_csv(input_files.star_reads_per_gene, header=None, sep="\t")
    unassigned_genes_df = df_star_reads_per_gene[
        df_star_reads_per_gene[0].astype(str).str.startswith("N_")
    ]  # unmapped, multimapping, noFeature, ambiguous
    unassigned_genes_unstranded = unassigned_genes_df.iloc[:, 1].sum()
    star_input_reads = int(df_star_stats.loc[('general','Number_of_input_reads')])

    pct_aligned_to_genes_unstranded = (
        100 * (star_input_reads - unassigned_genes_unstranded) / star_input_reads
    )
    stats["pct_aligned_to_genes_unstranded"] = pct_aligned_to_genes_unstranded

    # %Assigned to genes (unique; forward)
    unassigned_genes_forward = unassigned_genes_df.iloc[:, 2].sum()
    pct_aligned_to_genes_forward = (
        100 * (star_input_reads - unassigned_genes_forward) / star_input_reads
    )
    stats["pct_aligned_to_genes_forward"] = pct_aligned_to_genes_forward

    # %Assigned to genes (unique; reverse)
    unassigned_genes_reverse = unassigned_genes_df.iloc[:, 3].sum()
    pct_aligned_to_genes_reverse = (
        100 * (star_input_reads - unassigned_genes_reverse) / star_input_reads
    )
    stats["pct_aligned_to_genes_reverse"] = pct_aligned_to_genes_reverse

    # Average_mapped_length
    average_mapped_length = df_star_stats.loc[('unique_reads','Average_mapped_length')].value
    stats["average_mapped_length"] = average_mapped_length

    # Uniquely_mapped_reads_%
    pct_uniquely_mapped_reads =  df_star_stats.loc[('unique_reads','pct_Uniquely_mapped_reads')].value
    stats["pct_uniquely_mapped_reads"] = pct_uniquely_mapped_reads

    # Mismatch_rate_per_base_%
    mismatch_rate = float(df_star_stats.loc[('unique_reads','pct_Mismatch_rate_per_base')])
    stats["pct_mismatch"] = mismatch_rate

    # pct_deletion
    deletion_rate = float(df_star_stats.loc[('unique_reads','pct_Deletion_rate_per_base')])
    stats["pct_deletion"] = deletion_rate

    # pct_insertion
    insertion_rate = float(df_star_stats.loc[('unique_reads','pct_Insertion_rate_per_base')])
    stats["pct_insertion"] = insertion_rate

    series = pd.Series(stats, dtype="float")
    series.to_hdf(h5_file, key=H5Keys.STATISTICS_SHORTLIST.value)
