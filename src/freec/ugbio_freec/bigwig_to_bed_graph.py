import argparse
import sys

import pyBigWig


def run(argv):
    parser = argparse.ArgumentParser(prog="bigwig_to_bed_graph", description="Convert bigwig file to BedGraph")

    parser.add_argument("--bigwig", help="general:bigwigInput", required=True)
    parser.add_argument("--bed_graph", help="general:bedGraphOutput", required=True)

    args = parser.parse_args(argv[1:])
    bigwig_to_bedgraph(args.bigwig, args.bed_graph)


def bigwig_to_bedgraph(bigwig_file, bedgraph_file):
    """
    Converts a BigWig file to a BEDGraph file using pyBigWig.

    Parameters:
        bigwig_file (str): Path to the input BigWig file.
        bedgraph_file (str): Path to the output BEDGraph file.
    """
    # Open the BigWig file
    with pyBigWig.open(bigwig_file) as bw:
        # Open the output file for writing
        with open(bedgraph_file, "w") as bedgraph:
            # Iterate over all chromosomes and their intervals
            for chrom in bw.chroms():
                intervals = bw.intervals(chrom)
                if intervals:
                    for start, end, value in intervals:
                        bedgraph.write(f"{chrom}\t{start}\t{end}\t{value}\n")


if __name__ == "__main__":
    run(sys.argv)
