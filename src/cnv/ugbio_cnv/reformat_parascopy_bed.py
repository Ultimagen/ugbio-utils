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
#    Reformat parascopy CNV BED files to 4-column BED with GFF3-style tags for IGV
#    See added tests for the details

import argparse
import gzip
import logging
import sys
from pathlib import Path
from urllib.parse import quote

import pandas as pd
from ugbio_core.logger import logger

INFO_COLUMNS_START = 3  # Columns after chrom, start, end are considered info columns


def run(argv):
    parser = argparse.ArgumentParser(
        prog="reformat_parascopy_bed",
        description="Convert parascopy CNV BED files to 4-column BED with GFF3-style tags for IGV",
    )
    parser.add_argument("--input_bed", required=True, help="Input BED file (res.samples.bed.gz or res.paralog.bed.gz)")
    parser.add_argument("--output_bed", required=True, help="Output BED file (e.g., NA00372.samples.bed)")
    parser.add_argument("--verbosity", default="INFO", help="Verbosity: ERROR, WARNING, INFO, DEBUG")
    args = parser.parse_args(argv[1:])
    logger.setLevel(getattr(logging, args.verbosity))

    # Validate input
    if not Path(args.input_bed).exists():
        raise FileNotFoundError(f"Input file not found: {args.input_bed}")

    # Read BED file, skipping only ## comment lines (not #chrom header)
    # Find the header line index (starts with #chrom)
    open_func = gzip.open if args.input_bed.endswith(".gz") else open
    with open_func(args.input_bed, "rt") as f:
        skip_rows = 0
        for line in f:
            if line.startswith("##"):
                skip_rows += 1
            else:
                break

    bed_df = pd.read_csv(args.input_bed, sep="\t", compression="infer", skiprows=skip_rows)
    bed_df.columns = bed_df.columns.str.lstrip("#")  # Remove leading # from chrom column

    # Write output
    with open(args.output_bed, "w") as out:
        out.write("#gffTags\n")
        for _, row in bed_df.iterrows():
            chrom, start, end = row.iloc[0], row.iloc[1], row.iloc[2]
            tags = []
            for col in bed_df.columns[INFO_COLUMNS_START:]:
                val = str(row[col])
                if col.lower() == "info" and ";" in val:
                    # Expand info column tags
                    tags.extend(val.split(";"))
                else:
                    # Regular column: escape only ; and =
                    val_escaped = val.replace(";", quote(";")).replace("=", quote("="))
                    tags.append(f"{col}={val_escaped}")
            out.write(f"{chrom}\t{start}\t{end}\t{';'.join(tags)}\n")

    logger.info(f"Output written to {args.output_bed}")


def main():
    run(sys.argv)


if __name__ == "__main__":
    main()
