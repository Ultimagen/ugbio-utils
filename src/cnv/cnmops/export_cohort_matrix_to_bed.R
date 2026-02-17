# Copyright 2025 Ultima Genomics Inc.
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
#    exports read counts to BED

suppressPackageStartupMessages(library("GenomicRanges"))
suppressPackageStartupMessages(library("rtracklayer"))
suppressPackageStartupMessages(library("argparse"))

parser <- ArgumentParser()
parser$add_argument("input_file", help = "input RDS file containing GenomicRanges object")
parser$add_argument("--mean", action = "store_true", help = "export mean coverage instead of per-sample coverage")
parser$add_argument("--sample_name", help = "export coverage for a specific sample only")
parser$add_argument("--intervals_only", action = "store_true", help = "export only the intervals (chr, start, end) without coverage data")
args <- parser$parse_args()

germline_coverage_rds <- args$input_file
gr <- readRDS(germline_coverage_rds)


if (args$intervals_only) {
  # Export only the intervals without coverage data
  # can't use export.bed since it does not export 3-column BED
  output_file <- "intervals.bed"
  bed3 <- data.frame(
    chrom = as.character(seqnames(gr)),
    start = sprintf("%d", start(gr) - 1L),
    end   = sprintf("%d", end(gr))
  )
  write.table(
    bed3,
    file = output_file,
    sep = "\t",
    quote = FALSE,
    row.names = FALSE,
    col.names = FALSE
  )
} else if (!is.null(args$sample_name)) {
  # Export only the specified sample
  sample_names <- colnames(mcols(gr))
  gr_sample <- gr
  mcols(gr_sample) <- NULL
  mcols(gr_sample)$score <- mcols(gr)[[make.names(args$sample_name)]]
  export.bed(gr_sample, paste0(args$sample_name, ".cov.bed"))
} else if (!args$mean) {
  # Export all samples
  sample_names <- colnames(mcols(gr))
  for (sample in sample_names) {
    gr_sample <- gr
    mcols(gr_sample) <- NULL
    mcols(gr_sample)$score <- mcols(gr)[[make.names(sample)]]
    export.bed(gr_sample, paste0(sample, ".cov.bed"))
  }
} else {
  # Export mean coverage
  df <- as.data.frame(gr)
  df$cohort_avg <- rowMeans(df[, 6:ncol(df)], na.rm = TRUE)
  gr_cohort <- GRanges(
    seqnames = df$seqnames,
    ranges = IRanges(start = df$start, end = df$end),
    score = df$cohort_avg
  )
  export.bed(gr_cohort, "coverage.cohort.bed")
}
