""" pileuptofreq.py - Convert samtools pileup output to frequency table
    Copyright (C) 2022
    Author: Valentin Maurer <valentin.maurer@stud.uni-heidelberg.de>
"""

import fileinput
from pathlib import Path
from argparse import ArgumentParser
from collections import defaultdict
from sys import stdout

PILEUP_IGNORE   = {",", ".", "*", "#", ">", "<"}
USE_STARTEND    = True
# BASE_TO_FORWAWRD = {
#     "A" : "A", "C" : "C", "T" : "T", "N" : "N",
#     "a" : "T", "c" : "G", "t" : "A", "n" : "N"
# }

def pileup_to_freq(reference : str, pileup : str) -> (int):
    """ Counts A, C, G, T, In, Del occurence given a samtools pileup string"""

    pileup      = pileup.upper()
    frequencies = defaultdict(int)
    is_indel   = False
    indel_pos  = 0
    indel_size = ""

    is_start   = 0
    last_base  = ""

    for k in pileup:

        if k in ("+", "-"):
            is_indel    = True
            indel_pos   = 0
            indel_size  = ""
            indel       = k
            continue

        if k == "$":
            frequencies[last_base] -= not USE_STARTEND
            continue

        # Handle first base in read; Indels would come after (^)(QUAL)(BASE) structure
        if k == "^":
            is_start = 1
            continue
        if is_start > 0:
            if is_start < 2:
                is_start += 1
            elif is_start == 2:
                frequencies[k] += USE_STARTEND
                is_start = 0
            continue

        if is_indel:
            is_digit = str.isdigit(k)
            if is_digit:
                indel_size += k
                continue
            indel_pos += 1
            indel += k
            if indel_pos == int(indel_size):
                frequencies[indel] += 1
                is_indel = False
                indel_pos = 0
            continue

        last_base = k
        frequencies[k] += 1

    frequencies[reference] += frequencies[","] + frequencies["."]
    frequencies["Deletion"] = frequencies["*"] + frequencies["#"]

    return {k : v for k, v in frequencies.items() if v > 0 and k not in PILEUP_IGNORE}

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-i', type=Path,
        help = "Samtools mpileup output without extra flags. If empty, stdin is used",
        default = None)
    parser.add_argument('-o', type=Path,
        help = "Output file, if empty stdout is used.", default = None)
    args = parser.parse_args()

    if args.i is None:
        args.i = ("-", )

    fout = open(args.o, mode = "w", encoding = "utf-8") if args.o is not None else stdout
    fout.write("Depth;Chrom;Pos;Ref;Sample;Base;Count\n")
    with fileinput.input(args.i, mode = "r") as fin:
        for line in fin:
            SAMPLE    = 0
            values    = line.split("\t")
            DP = values[3]
            POSITION  = ';'.join([str(values[i]) for i in range(3)])
            # Go through all seq fields
            for i in range(4, len(values), 3):
                SAMPLE += 1
                # No reads aligned to this position
                if values[i-1] == "0":
                    continue
                freq = pileup_to_freq(values[2], values[i])
                for key, value in freq.items():
                    fout.write(f"{DP};{POSITION};{SAMPLE};{key};{value}\n")

    if fout is not stdout:
        fout.close()
