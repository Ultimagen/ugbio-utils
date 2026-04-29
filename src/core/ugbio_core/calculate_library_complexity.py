#!/usr/bin/env python3

import argparse
import math
import subprocess
import sys

import tqdm
from ugbio_core import sorter_utils


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI args and enforce exactly one supported input mode."""
    parser = argparse.ArgumentParser(
        description=(
            "Calculate library complexity. Provide exactly one mode: "
            "--cram OR --csv OR "
            "(--PF_Barcode_reads --PCT_PF_Reads_aligned --pct_duplication)."
        )
    )
    parser.add_argument("--cram", help="Path to CRAM file")
    parser.add_argument("--csv", help="Path to sorter CSV file")
    parser.add_argument("--PF_Barcode_reads", type=int, help="Total PF barcode reads")
    parser.add_argument("--PCT_PF_Reads_aligned", type=float, help="Percent aligned reads (0-100)")
    parser.add_argument("--pct_duplication", type=float, help="Duplication percentage (0-100)")
    args = parser.parse_args(argv)
    return args


def extract_n_c_from_cram(cram_path, threads=16):
    print(f"Parsing CRAM: {cram_path}")

    cmd = ["samtools", "view", "-@", str(threads), cram_path]
    n = 0
    c = 0
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True) as proc:
        for line in tqdm.tqdm(proc.stdout):
            fields = line.strip().split("\t")
            flag = int(fields[1])
            dup = 1 if (flag & 1024) else 0
            is_unmapped = 1 if (flag & 4) else 0
            is_secondary = 1 if (flag & 256) else 0
            is_supplementary = 1 if (flag & 2048) else 0
            if not is_unmapped and not is_secondary and not is_supplementary:
                n += 1
                if dup == 0:
                    c += 1
    return n, c


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

    elif args.cram:
        mode = "cram"
    else:
        raise ValueError(
            "Provide one of: --cram | --csv | " "(--PF_Barcode_reads --PCT_PF_Reads_aligned --pct_duplication)"
        )

    return mode


def run(argv):
    """Calculate library complexity from various input sources."""
    args = parse_args(argv[1:])
    mode = determine_mode(args)

    # -----------------------------
    # Compute N and C
    # -----------------------------
    if mode == "csv":
        print(f"Processing CSV: {args.csv}")

        sorter_csv = sorter_utils.read_and_parse_sorter_statistics_csv(args.csv)

        pf_barcode_reads = sorter_csv["PF_Barcode_reads"]
        print(f"PF_Barcode_reads {pf_barcode_reads}")

        pct_pf_reads_aligned = sorter_csv["PCT_PF_Reads_aligned"]
        print(f"PCT_PF_Reads_aligned {pct_pf_reads_aligned}")

        # Handle both possible names for duplication metric
        pct_duplication = sorter_csv["PCT_duplicates"]
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

    elif mode == "cram":
        print(f"Processing CRAM: {args.cram}")
        n, c = extract_n_c_from_cram(args.cram)
    else:
        raise ValueError(
            "Invalid parameters provided. Provide one of: --cram | --csv |"
            "(--PF_Barcode_reads and --PCT_PF_Reads_aligned and --pct_duplication)"
        )

    print(f"N = {n}")
    print(f"C = {c}")
    if n < c:
        raise ValueError(
            f"Invalid input: N (total reads) must be greater than C (non-duplicate reads). Got N={n}, C={c}."
        )
    if n == c:
        raise ValueError(
            f"Invalid input: N (total reads) cannot be equal to C (non-duplicate reads). Got N={n}, C={c}."
        )

    x = estimate_library_size(n, c)
    print(f"Estimated library size X = {x}")


def main():
    run(sys.argv)
