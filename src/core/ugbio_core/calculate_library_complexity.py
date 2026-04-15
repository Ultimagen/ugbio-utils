#!/usr/bin/env python3

import argparse
import math
import os
import subprocess

import polars as pl

# -----------------------------
# Argument parsing
# -----------------------------
parser = argparse.ArgumentParser(description="Calculate library complexity")
parser.add_argument("--cram", help="Path to CRAM file (optional)")
parser.add_argument("--tsv", help="Path to TSV file with MI_Z data")
parser.add_argument("--N", type=int, help="Total number of reads (optional if TSV provided)")
parser.add_argument("--C", type=int, help="Number of non-duplicate reads (optional if TSV provided)")
args = parser.parse_args()

# -----------------------------
# Validate input
# -----------------------------
if not (args.tsv or args.cram or (args.N is not None and args.C is not None)):
    parser.error("You must provide either --tsv OR --cram OR both --N and --C")

if args.tsv and (args.cram or args.N or args.C):
    print("⚠️ Multiple input modes provided — TSV will be used")
elif args.cram and (args.N or args.C):
    print("⚠️ CRAM + N/C provided — CRAM will be used")


# -----------------------------
# Extract tsv from cram file function
# -----------------------------
def extract_tsv_from_cram(cram_path, output_tsv, threads=16):
    print(f"Extracting TSV from CRAM: {cram_path}")

    cmd = ["samtools", "view", "-@", str(threads), cram_path]

    with subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True) as proc, open(output_tsv, "w") as out:
        for line in proc.stdout:
            fields = line.strip().split("\t")

            read_name = fields[0]
            flag = int(fields[1])

            # extract MI:Z tag
            mi = "NA"
            for f in fields[11:]:
                if f.startswith("MI:Z:"):
                    mi = f[5:]
                    break

            dup = 1 if (flag & 1024) else 0
            is_unmapped = 1 if (flag & 4) else 0
            is_secondary = 1 if (flag & 256) else 0
            is_supplementary = 1 if (flag & 2048) else 0

            out.write(f"{read_name}\t{dup}\t{mi}\t" f"{is_unmapped}\t{is_secondary}\t{is_supplementary}\n")

    print(f"TSV written to: {output_tsv}")


# -----------------------------
# Compute N and C
# -----------------------------
if args.tsv:
    tsv_path = args.tsv

elif args.cram:
    tsv_path_1 = args.cram.replace(".cram", "_MI_Z.tsv")
    tsv_path_2 = args.cram.replace(".cram", "_read_name_duplicate_mapped_secondary_supplementary_flag_MI_Z_tag.tsv")

    # Avoid regenerating TSV if it already exists:
    if not os.path.exists(tsv_path_1) and not os.path.exists(tsv_path_2):
        tsv_path = tsv_path_1
        extract_tsv_from_cram(args.cram, tsv_path)
    else:
        if os.path.exists(tsv_path_1):
            tsv_path = tsv_path_1
        if os.path.exists(tsv_path_2):
            tsv_path = tsv_path_2
        print(f"TSV already exists, skipping extraction: {tsv_path}")

else:
    print("Using provided N and C values")
    n = args.N
    c = args.C

# -----------------------------
# If we have a TSV → compute N and C
# -----------------------------
if args.tsv or args.cram:
    print(f"Loading TSV: {tsv_path}")

    MI_Z_df = pl.scan_csv(
        tsv_path,
        separator="\t",
        has_header=False,
        new_columns=["read_name", "dup", "MI_Z", "is_unmapped", "is_secondary", "is_supplementary"],
    ).collect()

    filtered = MI_Z_df.filter(
        (pl.col("is_unmapped") == 0) & (pl.col("is_secondary") == 0) & (pl.col("is_supplementary") == 0)
    )

    n = filtered.shape[0]

    c = filtered.filter(pl.col("dup") == 0).shape[0]

print(f"N = {n}")
print(f"C = {c}")

# -----------------------------
# Newton's method
# -----------------------------
print("Calculating library size X using Newton’s method")


def f(x, n, c):
    return x * (1 - math.exp(-n / x)) - c


def df(x, n):
    return (1 - math.exp(-n / x)) - (n / x) * math.exp(-n / x)


# initial guess
x = max(c, n * 0.1)

for i in range(50):
    fx = f(x, n, c)
    dfx = df(x, n)

    print(f"Iteration {i}: X={x}, f(X)={fx}, df(X)={dfx}")

    if dfx == 0:
        print("Derivative is zero, stopping.")
        break

    x_new = x - fx / dfx

    if abs(x_new - x) < 1e-6 * x:
        x = x_new
        print(f"Converged at iteration {i}")
        break

    x = x_new

print(f"Estimated library size X = {x}")
