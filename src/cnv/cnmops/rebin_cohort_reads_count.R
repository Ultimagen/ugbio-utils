# Copyright 2026 Ultima Genomics Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
# DESCRIPTION
#    Re-bin an existing CNmops cohort from smaller bins to larger bins by
#    aggregating read counts. This allows users to adjust the resolution of
#    existing cohorts without regenerating from BAM files.
#    IMPORTANT: New bin end positions are calculated as the maximum of the
#    original bin ends within each group, ensuring that partial bins at
#    chromosome ends are not artificially extended.
# USAGE
#    Rscript rebin_cohort_reads_count.R \
#      --input_cohort_file cohort.rds \
#      --new_window_length 5000 \
#      --output_file rebinned_cohort.rds
#
# CHANGELOG in reverse chronological order
#    2026-02-13: Auto-detect original_window_length from max width, handle equal window case
#    2026-02-13: Fix boundary handling - use max of original ends, not rounded (BIOIN-2615)

suppressPackageStartupMessages(library(cn.mops))
suppressPackageStartupMessages(library(magrittr))
suppressPackageStartupMessages(library(argparse))
suppressPackageStartupMessages(library(rhdf5))

parser <- ArgumentParser(description = "Re-bin CNmops cohort to larger window size")
parser$add_argument("-i", "--input_cohort_file",
                    required = TRUE,
                    help = "Input cohort RDS file path")
parser$add_argument("-owl", "--original_window_length",
                    type = "integer",
                    required = FALSE,
                    default = NULL,
                    help = "Original window length (bp). DEPRECATED: Auto-detected if not provided.")
parser$add_argument("-nwl", "--new_window_length",
                    type = "integer",
                    required = TRUE,
                    help = "New window length (bp) for rebinned cohort")
parser$add_argument("-o", "--output_file",
                    default = "rebinned_cohort_reads_count.rds",
                    help = "Output RDS file path")
parser$add_argument("--save_hdf", action = "store_true",
                    help = "Save reads count data in HDF5 format")
parser$add_argument("--save_csv", action = "store_true",
                    help = "Save reads count data in CSV format")

args <- parser$parse_args()

# Load cohort first (needed for auto-detection)
cat("Loading input cohort...\n")
gr <- readRDS(args$input_cohort_file)
cat("Input cohort has", length(gr), "bins\n")

# Auto-detect or validate original window length
if (is.null(args$original_window_length)) {
  # Auto-detect: use max width (handles most cases, including partial bins)
  widths <- width(gr)
  original_window_length <- max(widths)
  cat("Auto-detected original_window_length:", original_window_length, "bp (max width)\n")

} else {
  # User provided a value - use it directly
  original_window_length <- args$original_window_length
}


# Handle case where new == original (no re-binning needed)
if (args$new_window_length == original_window_length) {
  cat("INFO: new_window_length (", args$new_window_length, " bp) equals original_window_length (",
      original_window_length, " bp)\n", sep="")
  cat("      No re-binning needed. Saving cohort to output file...\n")

  # Save outputs
  new_gr = gr
} else {
  # Validate divisibility
  if (args$new_window_length %% original_window_length != 0) {
    stop("ERROR: new_window_length (", args$new_window_length,
        " bp) must be evenly divisible by original_window_length (", original_window_length, " bp).\n",
        "       Remainder: ", args$new_window_length %% original_window_length, " bp")
  }

  # Validate new > original
  if (args$new_window_length < original_window_length) {
    stop("ERROR: new_window_length (", args$new_window_length,
        " bp) must be larger than original_window_length (", original_window_length, " bp).\n",
        "       This script only aggregates to larger bins, not splits to smaller bins.")
  }

  bin_factor <- args$new_window_length / original_window_length
  cat("=== Re-binning Parameters ===\n")
  cat("Input file:", args$input_cohort_file, "\n")
  cat("Original window length:", original_window_length, "bp")
  if (is.null(args$original_window_length)) {
    cat(" (auto-detected)\n")
  } else {
    cat(" (user-provided)\n")
  }
  cat("New window length:", args$new_window_length, "bp\n")
  cat("Bin factor:", bin_factor, "x\n")
  cat("==============================\n\n")


  # Convert GRanges to data frame
  df <- as.data.frame(gr)

  # Get sample column names from data frame (after conversion, R makes names syntactically valid)
  # Sample columns are everything except the genomic coordinate columns
  genomic_cols <- c("seqnames", "start", "end", "width", "strand")
  sample_cols <- setdiff(colnames(df), genomic_cols)
  cat("Number of samples:", length(sample_cols), "\n")

  # Create new bin assignments
  # Genomic coordinates are 1-based and right-closed (inclusive on both ends)
  # Original bins: 1-1000, 1001-2000, 2001-3000, ...
  # New bins must maintain this alignment: 1-N, (N+1)-2N, (2N+1)-3N, ...
  # Formula: new_bin_start = floor((start - 1) / new_window_length) * new_window_length + 1
  df$new_bin_start <- floor((df$start - 1) / args$new_window_length) * args$new_window_length + 1

  # Create a grouping key for aggregation (without end position, which will be calculated)
  df$bin_key <- paste(df$seqnames, df$new_bin_start, sep = "_")

  cat("Aggregating read counts...\n")

  # Get unique bin keys and calculate their end positions
  # For each bin_key, the new_bin_end is the MAX of original bin ends in that group
  bin_ends <- tapply(df$end, df$bin_key, max)

  # Get unique bin keys and their coordinates
  unique_bins <- unique(df[, c("seqnames", "new_bin_start", "bin_key")])
  unique_bins <- unique_bins[order(unique_bins$seqnames, unique_bins$new_bin_start), ]

  # Add calculated end positions
  unique_bins$new_bin_end <- bin_ends[match(unique_bins$bin_key, names(bin_ends))]

  # Initialize result data frame
  rebinned_df <- unique_bins[, c("seqnames", "new_bin_start", "new_bin_end")]

  # Aggregate each sample column using tapply
  for (col in sample_cols) {
    # Sum read counts for each bin using tapply
    aggregated <- tapply(df[[col]], df$bin_key, sum)
    # Convert to data frame and match with unique_bins order
    rebinned_df[[col]] <- aggregated[match(unique_bins$bin_key, names(aggregated))]
  }

  cat("Output cohort has", nrow(rebinned_df), "bins\n")

  # Create new GRanges object
  cat("Creating new GRanges object...\n")
  new_gr <- GRanges(
    seqnames = rebinned_df$seqnames,
    ranges = IRanges(start = rebinned_df$new_bin_start, end = rebinned_df$new_bin_end),
    strand = "*"
  )

  # Add sample read counts as metadata columns
  for (col in sample_cols) {
    mcols(new_gr)[[col]] <- rebinned_df[[col]]
  }

}

# Save output
cat("Saving to", args$output_file, "...\n")
saveRDS(new_gr, file = args$output_file)

# Optional: save CSV
if (args$save_csv) {
  csv_file <- sub("\\.rds$", ".csv", args$output_file)
  cat("Saving CSV to", csv_file, "...\n")
  write.csv(as.data.frame(new_gr), file = csv_file, row.names = FALSE, quote = FALSE)
}

# Optional: save HDF5
if (args$save_hdf) {
  hdf5_file <- sub("\\.rds$", ".hdf5", args$output_file)
  cat("Saving HDF5 to", hdf5_file, "...\n")
  h5createFile(hdf5_file)
  h5write(as.data.frame(new_gr), hdf5_file, "rebinned_cohort_reads_count")
}

cat("Re-binning complete!\n")
