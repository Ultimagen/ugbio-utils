import argparse


def parse_bases(bases):
    ref_count = 0
    nonref_count = 0
    i = 0
    while i < len(bases):
        c = bases[i]
        if c in ".,":
            ref_count += 1
        elif c in "ACGTNacgtn*":
            nonref_count += 1
        elif c == "^":
            i += 1
        elif c == "$":
            pass
        elif c in "+-":
            i += 1
            length = ""
            nonref_count += 1  # count the indel itself
            while i < len(bases) and bases[i].isdigit():
                length += bases[i]
                i += 1
            i += int(length) - 1
        i += 1
    return ref_count, nonref_count


def load_bed_regions(bed_file, distance_start_to_center=1):
    regions = []
    contigs = set()
    with open(bed_file) as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                chrom, start, end = line.strip().split()[:3]
                start = int(start)
                end = int(end)
                center = start + distance_start_to_center  # 1-based, half-open interval
                region = {
                    "chrom": chrom,
                    "start": start,
                    "end": end,
                    "center": center,
                    "ref_counts": {
                        i: float("nan") for i in range(-distance_start_to_center, distance_start_to_center + 1)
                    },
                    "nonref_counts": {
                        i: float("nan") for i in range(-distance_start_to_center, distance_start_to_center + 1)
                    },
                    "seen": {i: False for i in range(-distance_start_to_center, distance_start_to_center + 1)},
                    "ref_base": "N",
                }
                contigs.add(chrom)
                regions.append(region)
    return regions, sorted(contigs)


def build_region_index(regions, distance_start_to_center=1):
    index = {}
    for region in regions:
        chrom = region["chrom"]
        for rel in range(-distance_start_to_center, distance_start_to_center + 1):
            pos = region["center"] + rel
            index.setdefault((chrom, pos), []).append((region, rel))
    return index


def process_mpileup(mpileup_file, region_index):
    n_fields_mpileup = 5
    with open(mpileup_file) as f:
        for line in f:
            fields = line.strip().split("\t")
            if len(fields) < n_fields_mpileup:
                continue
            chrom, pos, ref, depth, bases = fields[:n_fields_mpileup]
            pos = int(pos) - 1  # mpileup is 1-based
            key = (chrom, pos)
            if key not in region_index:
                continue
            ref_ct, nonref_ct = parse_bases(bases)
            for region, rel in region_index[key]:
                region["ref_counts"][rel] = ref_ct
                region["nonref_counts"][rel] = nonref_ct
                region["seen"][rel] = True
                if rel == 0:
                    region["ref_base"] = ref.upper()


def write_vcf(output_path, regions, contigs, distance_start_to_center):
    """Writes the processed regions to a VCF file.
    Args:
        output_path (str): Path to the output VCF file.
        regions (list): List of processed regions with counts.
        contigs (set): Set of contig names.
        distance_start_to_center (int): Distance from start to center.
    """
    format_range = range(-distance_start_to_center, distance_start_to_center + 1)
    format_range_str = [str(i) for i in format_range]
    format_range_str = [n.replace("-", "M") for n in format_range_str]  # replace negative with M for VCF format
    # Define the format ID and fields
    format_id = ":".join([f"REF_{i}" for i in format_range_str] + [f"NONREF_{i}" for i in format_range_str])
    with open(output_path, "w") as out:
        out.write("##fileformat=VCFv4.2\n")
        for contig in contigs:
            out.write(f"##contig=<ID={contig}>\n")

        format_fields = [(f"REF_{i}", f"Reference counts at position {i}") for i in format_range_str] + [
            (f"NONREF_{i}", f"Non-reference counts at position {i}") for i in format_range_str
        ]
        for fmt_id, desc in format_fields:
            out.write(f'##FORMAT=<ID={fmt_id},Number=1,Type=Integer,Description="{desc}">\n')

        header_fields = ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT", "sample1"]
        out.write("\t".join(header_fields) + "\n")

        for r in regions:
            chrom = r["chrom"]
            pos = r["center"] + 1  # VCF is 1-based
            ref = r["ref_base"]
            alt = "<NONREF>"
            qual = "."
            filt = "PASS"
            info = "."
            sample_value = ":".join(
                [
                    str(int(r["ref_counts"][i])) if r["seen"][i] else "."
                    for i in range(-distance_start_to_center, distance_start_to_center + 1)
                ]
                + [
                    str(int(r["nonref_counts"][i])) if r["seen"][i] else "."
                    for i in range(-distance_start_to_center, distance_start_to_center + 1)
                ]
            )
            out.write(f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t{qual}\t{filt}\t{info}\t{format_id}\t{sample_value}\n")


def main(argv: list[str] | None = None) -> None:
    # use parser to handle command line arguments
    """
    Minimal command-line interface, e.g.:

    $ python ref_nonref_per_base_window.py \
        --input input.mpileup \
        --bed regions.bed \
        --distance_start_to_center 1 \
        --output output.vcf
    """
    parser = argparse.ArgumentParser(description="Parse samtools mpileup to vcf", allow_abbrev=True)
    parser.add_argument("--input", required=True, help="Input samtools mpileup output file")
    parser.add_argument("--bed", required=True, help="Input BED file")
    parser.add_argument("--distance_start_to_center", type=int, default=1, help="Distance from start to center")
    parser.add_argument("--output", required=True, help="Output VCF file")
    args = parser.parse_args(argv)

    regions, contigs = load_bed_regions(args.bed, distance_start_to_center=args.distance_start_to_center)
    region_index = build_region_index(regions, distance_start_to_center=args.distance_start_to_center)
    process_mpileup(args.input, region_index)
    write_vcf(args.output, regions, contigs, distance_start_to_center=args.distance_start_to_center)


if __name__ == "__main__":  # pragma: no cover
    main()
