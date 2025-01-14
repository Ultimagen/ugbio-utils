import pysam


def contig_lens_from_bam_header(bam_file: str, output_file: str):
    """Creates a "sizes" file from contig lengths in bam header.
    Sizes file is per the UCSC spec: contig <tab> length

    Parameters
    ----------
    bam_file: str
        Bam file
    output_file: str
        Output file

    Returns
    -------
    None, writes output_file
    """

    with pysam.AlignmentFile(bam_file) as infile:
        with open(output_file, "w", encoding="ascii") as outfile:
            lengths = infile.header.lengths
            contigs = infile.header.references
            for contig, length in zip(contigs, lengths, strict=False):
                outfile.write(f"{contig}\t{length}\n")
