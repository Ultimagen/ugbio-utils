"""HTML report of a sequins analysis: detected spikes, SNP/indel precision/recall/F1 and discordant variants."""

import argparse
import csv
import html
import subprocess
import sys

CSS = (
    "body{font-family:sans-serif;margin:2em}table{border-collapse:collapse;margin:1em 0}"
    "th,td{border:1px solid #ccc;padding:4px 10px;text-align:left}th{background:#eee}"
    ".yes{color:#1a7f37;font-weight:bold}.no{color:#888}"
)
# bcftools type selection for each variant class
TYPE_OPTIONS = {"snps": ["-v", "snps"], "indels": ["-V", "snps"]}


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="sequins_report", description=__doc__)
    ap.add_argument("--base-file-name", required=True, help="Prefix of the output file")
    ap.add_argument("--min-median-depth", type=int, required=True, help="Detection threshold on the median depth")
    ap.add_argument("--genomes", nargs="+", required=True, help="Genome names")
    ap.add_argument("--median-depths", nargs="+", type=int, required=True, help="Median depth per genome")
    ap.add_argument("--mean-depths", nargs="+", type=int, required=True, help="Mean depth per genome")
    ap.add_argument("--detected", nargs="+", required=True, choices=["true", "false"], help="Detected, per genome")
    # Result files exist only for the detected genomes, in the order of --genomes
    for name in ("tp-baseline-vcfs", "tp-vcfs", "fp-vcfs", "fn-vcfs", "discordant-tables"):
        ap.add_argument(f"--{name}", nargs="*", default=[], help=f"{name} of the detected genomes")
    args = ap.parse_args(argv)

    n = len(args.genomes)
    if not (len(args.median_depths) == len(args.mean_depths) == len(args.detected) == n):
        ap.error("--genomes, --median-depths, --mean-depths and --detected must have the same length")
    n_detected = sum(d == "true" for d in args.detected)
    for name in ("tp_baseline_vcfs", "tp_vcfs", "fp_vcfs", "fn_vcfs", "discordant_tables"):
        if len(getattr(args, name)) != n_detected:
            ap.error(f"--{name.replace('_', '-')} must have one file per detected genome ({n_detected})")
    return args


def count_variants(vcf: str, variant_class: str) -> int:
    """Number of records of the given class ("snps" or "indels", the latter being everything but SNPs)."""
    cmd = ["bcftools", "view", "-H", *TYPE_OPTIONS[variant_class], vcf]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout  # noqa: S603
    return len(out.splitlines())


def precision_recall_f1(tp_baseline: int, tp_call: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp_call / (tp_call + fp) if (tp_call + fp) else 0.0
    recall = tp_baseline / (tp_baseline + fn) if (tp_baseline + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["<table><tr>" + "".join(f"<th>{html.escape(h)}</th>" for h in header) + "</tr>"]
    lines += ["<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows]
    lines.append("</table>")
    return lines


def _genome_section(genome: str, files: dict[str, str]) -> list[str]:
    rows = []
    for variant_class in TYPE_OPTIONS:
        a = count_variants(files["tp_baseline"], variant_class)
        b = count_variants(files["tp"], variant_class)
        p = count_variants(files["fp"], variant_class)
        n = count_variants(files["fn"], variant_class)
        precision, recall, f1 = precision_recall_f1(a, b, p, n)
        rows.append([variant_class, str(a), str(b), str(p), str(n), f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}"])
    out = _table(["type", "TP (truth)", "TP (call)", "FP", "FN", "precision", "recall", "F1"], rows)

    out += [
        "<h3>Discordant variants</h3>",
        "<p>FP: called but not in the truth, FN: truth variants that were not called.</p>",
    ]
    with open(files["discordant"], newline="") as fh:
        table = list(csv.reader(fh, delimiter="\t"))
    if len(table) > 1:
        out += _table(table[0], [[html.escape(c) for c in row] for row in table[1:]])
    else:
        out.append("<p>None.</p>")
    return out


def build_report(args: argparse.Namespace) -> str:
    title = html.escape(args.base_file_name)
    detected = [d == "true" for d in args.detected]
    names = [g for g, d in zip(args.genomes, detected, strict=True) if d]

    out = [
        f'<!DOCTYPE html>\n<html><head><meta charset="utf-8"><title>Sequins report: {title}</title>',
        f"<style>{CSS}</style></head><body>",
        f"<h1>Sequins report: {title}</h1>",
        "<p>A sequins spike is detected when its median depth at the truth-variant sites (duplicates counted) "
        f"is above {args.min_median_depth}.</p>",
        "<h2>Detected spikes</h2>",
        f"<p>Detected: {html.escape(' '.join(names))}</p>" if names else "<p>No sequins spike was detected.</p>",
    ]
    rows = [
        [html.escape(g), str(med), str(mean), '<span class="yes">yes</span>' if d else '<span class="no">no</span>']
        for g, med, mean, d in zip(args.genomes, args.median_depths, args.mean_depths, detected, strict=True)
    ]
    out += _table(["genome", "median depth", "mean depth", "detected"], rows)

    k = 0  # indexes the per-genome result lists, which hold only the detected genomes
    for genome, median, is_detected in zip(args.genomes, args.median_depths, detected, strict=True):
        out.append(f"<h2>{html.escape(genome)}</h2>")
        if not is_detected:
            out.append(
                f"<p>Not detected (median depth {median}, required above {args.min_median_depth}); "
                "no variant calling was done.</p>"
            )
            continue
        files = {
            "tp_baseline": args.tp_baseline_vcfs[k],
            "tp": args.tp_vcfs[k],
            "fp": args.fp_vcfs[k],
            "fn": args.fn_vcfs[k],
            "discordant": args.discordant_tables[k],
        }
        out += _genome_section(genome, files)
        k += 1
    out.append("</body></html>")
    return "\n".join(out) + "\n"


def run(argv: list[str]) -> str:
    args = parse_args(argv[1:])
    report_path = f"{args.base_file_name}.sequins_report.html"
    with open(report_path, "w") as fh:
        fh.write(build_report(args))
    return report_path


def main() -> None:
    run(sys.argv)


if __name__ == "__main__":
    main()
