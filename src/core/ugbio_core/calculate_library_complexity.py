#!/usr/bin/env python3

import argparse
import math
import os
import subprocess

import pandas as pd
import polars as pl


def parse_args() -> argparse.Namespace:
    """Parse CLI args and enforce exactly one supported input mode."""
    parser = argparse.ArgumentParser(
        description=(
            "Calculate library complexity. Provide exactly one mode: "
            "--cram OR --csv OR (--N and --C) OR "
            "(--PF_Barcode_reads --PCT_PF_Reads_aligned --pct_duplication)."
        )
    )
    parser.add_argument("--cram", help="Path to CRAM file")
    parser.add_argument("--csv", help="Path to sorter CSV file")
    parser.add_argument("--N", type=int, help="Total number of reads (must be used with --C)")
    parser.add_argument("--C", type=int, help="Number of non-duplicate reads (must be used with --N)")
    parser.add_argument("--PF_Barcode_reads", type=float, help="Total PF barcode reads")
    parser.add_argument("--PCT_PF_Reads_aligned", type=float, help="Percent aligned reads (0-100)")
    parser.add_argument("--pct_duplication", type=float, help="Duplication percentage (0-100)")
    args = parser.parse_args()

    has_cram = bool(args.cram)
    has_csv = bool(args.csv)
    has_nc = args.N is not None or args.C is not None
    has_pf = (
        args.PF_Barcode_reads is not None or args.PCT_PF_Reads_aligned is not None or args.pct_duplication is not None
    )

    selected_modes = sum([has_cram, has_csv, has_nc, has_pf])
    if selected_modes == 0:
        parser.error(
            "No input mode selected. Provide exactly one mode: --cram OR --csv OR "
            "(--N and --C) OR (--PF_Barcode_reads --PCT_PF_Reads_aligned --pct_duplication)."
        )
    if selected_modes > 1:
        parser.error(
            "Multiple input modes were provided. Choose exactly one: --cram OR --csv OR "
            "(--N and --C) OR (--PF_Barcode_reads --PCT_PF_Reads_aligned --pct_duplication)."
        )

    if args.N is not None and args.C is None:
        parser.error("The N/C mode requires both --N and --C.")
    if args.N is None and args.C is not None:
        parser.error("--C can only be used with --N.")

    if args.PF_Barcode_reads is not None and (args.PCT_PF_Reads_aligned is None or args.pct_duplication is None):
        parser.error(
            "The PF metrics mode requires all three parameters: --PF_Barcode_reads "
            "--PCT_PF_Reads_aligned --pct_duplication."
        )
    if args.PF_Barcode_reads is None and (args.PCT_PF_Reads_aligned is not None or args.pct_duplication is not None):
        parser.error("--PCT_PF_Reads_aligned and --pct_duplication can only be used with --PF_Barcode_reads.")

    return args


# -----------------------------
# Extract tsv from cram file function
# -----------------------------
def extract_tsv_from_cram(cram_path, output_tsv, threads=16):
    print(f"Extracting TSV from CRAM: {cram_path}")

    cmd = ["samtools", "view", "-@", str(threads), cram_path]

    with subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True) as proc, open(output_tsv, "w") as out:
        for line in proc.stdout or []:
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


def estimate_library_size(n, c):
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
    return int(x)


def determine_mode(args):
    # -----------------------------
    # input mode (priority logic)
    # -----------------------------
    if args.csv:
        mode = "csv"

    elif (
        args.PF_Barcode_reads is not None and args.PCT_PF_Reads_aligned is not None and args.pct_duplication is not None
    ):
        if not (0 <= args.PCT_PF_Reads_aligned <= 100):  # noqa: PLR2004
            raise ValueError("PCT_PF_Reads_aligned must be between 0 and 100")

        if not (0 <= args.pct_duplication <= 100):  # noqa: PLR2004
            raise ValueError("pct_duplication must be between 0 and 100")

        mode = "pf_metrics"

    elif args.N is not None and args.C is not None:
        mode = "n_c"

    elif args.cram:
        mode = "cram"

    else:
        raise ValueError(
            "Provide one of: --cram | --csv | (--N and --C) | "
            "(--PF_Barcode_reads --PCT_PF_Reads_aligned --pct_duplication)"
        )

    return mode


def main():
    args = parse_args()
    mode = determine_mode(args)

    # -----------------------------
    # Compute N and C
    # -----------------------------
    if mode == "csv":
        print(f"Processing CSV: {args.csv}")

        sorter_csv = pd.read_csv(args.csv, header=None, names=["metric", "value"])

        pf_barcode_reads = sorter_csv.loc[sorter_csv["metric"] == "PF_Barcode_reads", "value"].to_numpy()[0]
        print(f"PF_Barcode_reads {pf_barcode_reads}")

        pct_pf_reads_aligned = sorter_csv.loc[sorter_csv["metric"] == "PCT_PF_Reads_aligned", "value"].to_numpy()[0]
        print(f"PCT_PF_Reads_aligned {pct_pf_reads_aligned}")

        pct_duplication = sorter_csv.loc[sorter_csv["metric"] == "% duplicates", "value"].to_numpy()[0]
        print(f"pct_duplication {pct_duplication}")

        n = int(pf_barcode_reads * (pct_pf_reads_aligned / 100))
        d = int((pct_duplication / 100) * n)
        c = n - d
        print(f"N={n}, D={d}, C={c}")

    elif mode == "pf_metrics":
        print("Using PF metrics to compute N and C")

        pf_barcode_reads = args.PF_Barcode_reads
        pct_pf_reads_aligned = args.PCT_PF_Reads_aligned
        pct_duplication = args.pct_duplication

        n = int(pf_barcode_reads * (pct_pf_reads_aligned / 100))
        d = int((pct_duplication / 100) * n)
        c = n - d

    elif mode == "n_c":
        print("Using provided N and C")

        n = args.N
        c = args.C

    elif mode == "cram":
        print(f"Processing CRAM: {args.cram}")

        tsv_path = args.cram.replace(".cram", "_MI_Z.tsv")

        if not os.path.exists(tsv_path):
            extract_tsv_from_cram(args.cram, tsv_path)
        else:
            print(f"TSV already exists: {tsv_path}")

        mi_z_df = pl.scan_csv(
            tsv_path,
            separator="\t",
            has_header=False,
            new_columns=["read_name", "dup", "MI_Z", "is_unmapped", "is_secondary", "is_supplementary"],
        ).collect()

        filtered = mi_z_df.filter(
            (pl.col("is_unmapped") == 0) & (pl.col("is_secondary") == 0) & (pl.col("is_supplementary") == 0)
        )

        n = filtered.shape[0]
        c = filtered.filter(pl.col("dup") == 0).shape[0]
    else:
        raise ValueError(
            "Invalid parameters provided. Provide one of: --cram | --csv | (--N and --C) | "
            "(--PF_Barcode_reads and --PCT_PF_Reads_aligned and --pct_duplication)"
        )

    print(f"N = {n}")
    print(f"C = {c}")

    x = estimate_library_size(n, c)
    print(f"Estimated library size X = {x}")
