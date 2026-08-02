suppressPackageStartupMessages(library(cn.mops))
suppressPackageStartupMessages(library(magrittr))
suppressPackageStartupMessages(library(argparse))
suppressPackageStartupMessages(library(rhdf5))
suppressPackageStartupMessages(library(GenomicRanges))


parser <- ArgumentParser()

parser$add_argument("-i", "--input_bam_file",
                    help="input bam file path")
parser$add_argument("-refseq", "--refSeqNames_string",
                    help="chromosome names in comma seperated format. e.g, chr1,chr2,chrX" ,
                    default="chr1,chr2,chr3,chr4,chr5,chr6,chr7,chr8,chr9,chr10,chr11,chr12,chr13,chr14,chr15,chr16,chr17,chr18,chr19,chr20,chr21,chr22,chrX,chrY")
parser$add_argument("-wl", "--window_length",
                   help="window length (#bp) for which reads count is calculated for",
                   type="integer", default=1000)
parser$add_argument("--intervals",
                    help=paste("BED3 file of the cohort's genomic windows. Reads are counted",
                               "over these windows. Mutually exclusive with -refseq/-wl."))
parser$add_argument("-o", "--base_file_name",
                    help="out base file name")
parser$add_argument("--save_hdf", action='store_true',
                    help="whether to save reads count data-frames in hdf5 format")
parser$add_argument("--save_csv", action='store_true',
                    help="whether to save reads count data-frames in csv format")

args <- parser$parse_args()

# -refseq/-wl carry defaults, so detect whether the user explicitly passed them
# by inspecting the raw command line (needed to enforce mutual exclusivity with --intervals).
# Match both the space form ("-wl 1000") and the equals form ("--window_length=1000").
raw_args <- commandArgs(trailingOnly = TRUE)
flag_given <- function(flags) {
  any(raw_args %in% flags | grepl(paste0("^(", paste(flags, collapse = "|"), ")="), raw_args))
}
refseq_given <- flag_given(c("-refseq", "--refSeqNames_string"))
wl_given <- flag_given(c("-wl", "--window_length"))

if (!is.null(args$intervals)) {
  if (refseq_given || wl_given) {
    stop("--intervals is mutually exclusive with -refseq/-wl; provide only one.")
  }
  # BED is 0-based half-open; +1L on start recovers the 1-based windows that
  # getReadCountsFromBAM would have produced for the same bins.
  bed <- read.table(args$intervals, sep = "\t", header = FALSE,
                    colClasses = c("character", "integer", "integer"))
  gr <- GRanges(seqnames = bed[[1]],
                ranges = IRanges(start = bed[[2]] + 1L, end = bed[[3]]))
  # getSegmentReadCountsFromBAM uses the same underlying counter (.countBamInGRanges,
  # default min.mapq = 1) as getReadCountsFromBAM, so counts match on identical windows.
  # cn.mops parallelizes over BAM files, and this script always processes a single BAM,
  # so parallelism has no effect -- counting runs serially (cn.mops default parallel = 0).
  bamDataRanges_RC <- getSegmentReadCountsFromBAM(args$input_bam_file, GR = gr,
                                                  sampleNames = basename(args$input_bam_file))
} else {
  refSeqNames <- unlist(strsplit(args$refSeqNames_string, ","))
  bamDataRanges_RC <- getReadCountsFromBAM(args$input_bam_file, refSeqNames=refSeqNames, WL=args$window_length)
}
saveRDS(bamDataRanges_RC, file = paste(args$base_file_name,".ReadCounts.rds",sep = ""))

if(args$save_csv){
  write.csv(as.data.frame(bamDataRanges_RC),paste(args$base_file_name,".ReadCounts.csv",sep = ""), row.names = FALSE,quote=FALSE)
}
if(args$save_hdf){
  hdf5_out_file_name <- paste(args$base_file_name,".ReadCounts.hdf5",sep = "")
  h5createFile(hdf5_out_file_name)
  h5write(as.data.frame(bamDataRanges_RC), hdf5_out_file_name,"bamDataRanges_RC")
}
