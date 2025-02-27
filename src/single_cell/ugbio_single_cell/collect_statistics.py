import gzip
import json
import warnings
from collections import defaultdict
from pathlib import Path

import pandas as pd
from Bio import SeqIO
from ugbio_core.sorter_utils import read_sorter_statistics_csv
from ugbio_core.trimmer_utils import merge_trimmer_histograms, read_trimmer_failure_codes
from ugbio_single_cell.sc_qc_dataclasses import H5Keys, Inputs, OutputFiles


def collect_statistics(
    input_files: Inputs,
    output_path: str,
    sample_name: str,
    star_db: str = "STAR_hg38_3_2.7.10a",
    *,
    save_trimmer_histogram: bool = False,
) -> Path:
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
    star_db : str
        DB name used when running STAR.
    save_trimmer_histogram : bool
        Save Trimmer histogram.

    Returns
    -------
    Path
        Path to the h5 file with statistics.
    """
    Path(output_path).mkdir(parents=True, exist_ok=True)

    if save_trimmer_histogram and input_files.trimmer_histogram_csv:
        try:
            # Merge Trimmer histograms from two optical paths (if needed)
            merged_histogram_csv = merge_trimmer_histograms(input_files.trimmer_histogram_csv, output_path=output_path)
            histogram = pd.read_csv(merged_histogram_csv)
        except Exception as e:  # noqa
            print(
                f"Failed to process Trimmer histograms. It will be skipped and barcode rank plot won't be added "
                f"to the report. Error details: {e}"
            )
            histogram = None

    # Read inputs into df
    trimmer_stats = pd.read_csv(input_files.trimmer_stats_csv)
    df_trimmer_failure_codes = read_trimmer_failure_codes(
        input_files.trimmer_failure_codes_csv, add_total=True, include_pretrim_filters=False
    )
    sorter_stats = read_sorter_statistics_csv(input_files.sorter_stats_csv)
    if input_files.sorter_stats_json:
        with open(input_files.sorter_stats_json) as f:
            sorter_stats_json = json.load(f)
        sorter_stats_json_df = pd.DataFrame([sorter_stats_json])
    star_stats = read_star_stats(input_files.star_stats, star_db=star_db)
    star_reads_per_gene = pd.read_csv(input_files.star_reads_per_gene, header=None, sep="\t")

    # Get insert subsample quality and lengths
    insert_quality, insert_lengths = get_insert_properties(input_files.insert)

    # Save statistics into h5
    output_filename = Path(output_path) / (sample_name + OutputFiles.H5.value)

    with pd.HDFStore(output_filename, "w") as store:
        store.put(H5Keys.TRIMMER_STATS.value, trimmer_stats, format="table")
        store.put(H5Keys.TRIMMER_FAILURE_CODES.value, df_trimmer_failure_codes, format="table")
        if save_trimmer_histogram:
            store.put(H5Keys.TRIMMER_HISTOGRAM.value, histogram, format="table")
        store.put(H5Keys.SORTER_STATS.value, sorter_stats, format="table")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
            store.put(H5Keys.STAR_STATS.value, star_stats)
        store.put(H5Keys.STAR_READS_PER_GENE.value, star_reads_per_gene, format="table")
        store.put(H5Keys.INSERT_QUALITY.value, insert_quality, format="table")
        store.put(H5Keys.INSERT_LENGTHS.value, pd.Series(insert_lengths), format="table")
        if input_files.sorter_stats_json:
            store.put(H5Keys.SORTER_STATS_JSON.value, sorter_stats_json_df)

    return output_filename


def read_star_stats(star_stats_file: str, star_db: str = "STAR_hg38_3_2.7.10a") -> pd.Series:
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
    star_stats_df = pd.read_csv(star_stats_file, header=None, sep="\t")
    star_stats_df.columns = ["metric", "value"]

    # parse metric description
    star_stats_df["metric"] = star_stats_df["metric"].str.replace("|", "").str.strip().str.replace(" ", "_")
    star_stats_df.loc[:, "metric"] = star_stats_df["metric"].str.replace(",", "").str.replace(":", "")

    # Add type (general, unique_reads, multi_mapping_reads, unmapped_reads, chimeric_reads)
    star_stats_df.loc[:, "type"] = (
        star_stats_df["metric"]
        .where(star_stats_df["value"].isna())
        .ffill()
        .fillna("general")
        .str.lower()
        .str.replace(":", "")
    )
    star_stats_df = star_stats_df.dropna(subset=["value"])

    # Add "pct_" to metric name if value ends with "%" (and remove "%" from the name)
    star_stats_df.loc[star_stats_df["value"].str.endswith("%"), "metric"] = star_stats_df.loc[
        star_stats_df["value"].str.endswith("%"), "metric"
    ].apply(lambda x: "pct_" + x.replace("_%", "").replace("%_", ""))

    # Remove "%" from value
    star_stats_df["value"] = star_stats_df["value"].str.replace("%", "")

    # Set index
    star_stats_df = star_stats_df.set_index(["type", "metric"])
    # convert df to pd.series for easier access
    s = star_stats_df["value"]

    # convert types
    s = s.apply(convert_value)

    # Add star_db to the series
    s["general", "star_db"] = star_db

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


def extract_statistics_table(h5_file: Path):  # noqa: PLR0915
    """
    Create shortlist of statistics from h5 file and append it to h5 file.

    Parameters
    ----------
    h5_file : Path
        Path to the h5 file with statistics.
    """
    stats = {}

    with pd.HDFStore(h5_file, "r") as store:
        # number of Input Reads
        num_input_reads_list = store[H5Keys.TRIMMER_STATS.value]["num input reads"].to_numpy()
        num_input_reads = next((x for x in num_input_reads_list if x != 0), None)
        if num_input_reads is None:
            raise ValueError("Number of input reads in trimmer statistics is not available.")
        stats["num_input_reads"] = num_input_reads

        # number of Failed reads due to rsq or other prefilters (e.g., subsampling)
        trimmer_stats_df = store[H5Keys.TRIMMER_STATS.value]
        trimmer_start_segment_stats_df = trimmer_stats_df[trimmer_stats_df["segment label"] == "start"]
        num_failed_reads = 0
        if len(trimmer_start_segment_stats_df) > 0:
            num_failed_reads = trimmer_start_segment_stats_df["num failures"].sum()
            stats["num_PF_reads"] = num_input_reads - num_failed_reads
            stats["pct_PF"] = 100 * (num_input_reads - num_failed_reads) / num_input_reads

        # number of Trimmed reads
        num_trimmed_reads_list = store[H5Keys.TRIMMER_STATS.value]["num trimmed reads"].to_numpy()
        num_trimmed_reads = next((x for x in num_trimmed_reads_list if x != 0), None)
        stats["num_trimmed_reads"] = num_trimmed_reads

        # pct_pass_trimmer
        pass_trimmer_rate = num_trimmed_reads / (num_input_reads - num_failed_reads)
        stats["pct_pass_trimmer"] = pass_trimmer_rate * 100

        # Mean read length
        mean_read_length = int(store[H5Keys.STAR_STATS.value].loc[("general", "Average_input_read_length")]) + 1
        stats["mean_read_length"] = mean_read_length

        # %q >= 20 for insert
        q20 = store[H5Keys.SORTER_STATS.value].loc["PCT_PF_Q20_bases"]
        stats["pct_q20"] = q20

        # %q >= 30 for insert
        q30 = store[H5Keys.SORTER_STATS.value].loc["PCT_PF_Q30_bases"]
        stats["pct_q30"] = q30

        # %Aligned to genome
        ur_tmm = float(
            store[H5Keys.STAR_STATS.value].loc[("unmapped_reads", "pct_of_reads_unmapped_too_many_mismatches")]
        )
        ur_ts = float(store[H5Keys.STAR_STATS.value].loc[("unmapped_reads", "pct_of_reads_unmapped_too_short")])
        ur_other = float(store[H5Keys.STAR_STATS.value].loc[("unmapped_reads", "pct_of_reads_unmapped_other")])
        pct_aligned_to_genome = 100 - ur_tmm - ur_ts - ur_other
        stats["pct_aligned_to_genome"] = pct_aligned_to_genome

        # %Assigned to genes (unique)
        unassigned_genes_df = store[H5Keys.STAR_READS_PER_GENE.value][
            store[H5Keys.STAR_READS_PER_GENE.value][0].astype(str).str.startswith("N_")
        ]  # unmapped, multimapping, noFeature, ambiguous
        unassigned_genes_unstranded = unassigned_genes_df.iloc[:, 1].sum()
        star_input_reads = int(store[H5Keys.STAR_STATS.value].loc[("general", "Number_of_input_reads")])

        pct_aligned_to_genes_unstranded = 100 * (star_input_reads - unassigned_genes_unstranded) / star_input_reads
        stats["pct_aligned_to_genes_unstranded"] = pct_aligned_to_genes_unstranded

        # %Assigned to genes (unique; forward)
        unassigned_genes_forward = unassigned_genes_df.iloc[:, 2].sum()
        pct_aligned_to_genes_forward = 100 * (star_input_reads - unassigned_genes_forward) / star_input_reads
        stats["pct_aligned_to_genes_forward"] = pct_aligned_to_genes_forward

        # %Assigned to genes (unique; reverse)
        unassigned_genes_reverse = unassigned_genes_df.iloc[:, 3].sum()
        pct_aligned_to_genes_reverse = 100 * (star_input_reads - unassigned_genes_reverse) / star_input_reads
        stats["pct_aligned_to_genes_reverse"] = pct_aligned_to_genes_reverse

        # Average_mapped_length
        average_mapped_length = store[H5Keys.STAR_STATS.value].loc[("unique_reads", "Average_mapped_length")]
        stats["average_mapped_length"] = average_mapped_length

        # Uniquely_mapped_reads_%
        pct_uniquely_mapped_reads = store[H5Keys.STAR_STATS.value].loc[("unique_reads", "pct_Uniquely_mapped_reads")]
        stats["pct_uniquely_mapped_reads"] = pct_uniquely_mapped_reads

        # Mismatch_rate_per_base_%
        mismatch_rate = float(store[H5Keys.STAR_STATS.value].loc[("unique_reads", "pct_Mismatch_rate_per_base")])
        stats["pct_mismatch"] = mismatch_rate

        # pct_deletion
        deletion_rate = float(store[H5Keys.STAR_STATS.value].loc[("unique_reads", "pct_Deletion_rate_per_base")])
        stats["pct_deletion"] = deletion_rate

        # pct_insertion
        insertion_rate = float(store[H5Keys.STAR_STATS.value].loc[("unique_reads", "pct_Insertion_rate_per_base")])
        stats["pct_insertion"] = insertion_rate

        # cell_barcode_filter statistics
        if H5Keys.SORTER_STATS_JSON.value in store:
            extract_cell_barcode_filter_data(stats, store)

    series = pd.Series(stats, dtype="float")
    series.to_hdf(h5_file, key=H5Keys.STATISTICS_SHORTLIST.value)


def extract_cell_barcode_filter_data(stats, store):
    sorter_stats_json_df = store[H5Keys.SORTER_STATS_JSON.value]
    if "cell_barcode_filter" in sorter_stats_json_df:
        cell_barcode_filter = sorter_stats_json_df["cell_barcode_filter"].iloc[0]  # get "cell_barcode_filter" dict
        n_failed_cbcs = cell_barcode_filter["nr_failed_cbcs"]
        n_good_cbcs_above_thresh = cell_barcode_filter["nr_good_cbcs_above_threshold"]
        n_failed_cbc_reads = cell_barcode_filter["nr_failed_reads"]
        n_total_reads = sorter_stats_json_df["total_reads"].iloc[0]

        if (n_failed_cbcs + n_good_cbcs_above_thresh) > 0:
            percent_failed_cbcs_above_threshold = 100 * n_failed_cbcs / (n_failed_cbcs + n_good_cbcs_above_thresh)
            stats["pct_failed_cbcs_above_threshold"] = percent_failed_cbcs_above_threshold

        if n_total_reads > 0:
            percent_cbc_filter_failed_reads = 100 * n_failed_cbc_reads / n_total_reads
            stats["pct_cbc_filter_failed_reads"] = percent_cbc_filter_failed_reads
