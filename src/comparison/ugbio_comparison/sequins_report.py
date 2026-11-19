"""HTML report of a sequins analysis: detected spikes, SNP/indel precision/recall/F1 and discordant variants."""

import argparse
import html
import sys

import pandas as pd

from ugbio_comparison.sequins_vcfeval import H5_KEY

CSS = (
    "body{font-family:sans-serif;margin:2em}table{border-collapse:collapse;margin:1em 0}"
    "th,td{border:1px solid #ccc;padding:4px 10px;text-align:left}th{background:#eee}"
    ".yes{color:#1a7f37;font-weight:bold}.no{color:#888}"
)
GENOME_TITLES = {
    "sequins_clinical": "Sequins WGS Clinical Control Set",
    "sequins_core": "Sequins WGS Core Control Set",
}
DISCORDANT_COLUMNS = ["class", "type", "chrom", "pos", "ref", "alt", "gt", "qual", "filter", "DP"]


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="sequins_report", description=__doc__)
    ap.add_argument("--base-file-name", required=True, help="Sample name, shown in the report title")
    ap.add_argument("--output", default="report.html", help="Output HTML file (default: %(default)s)")
    ap.add_argument("--input-crams", nargs="+", required=True, help="Paths of the input CRAMs the reads came from")
    ap.add_argument("--target-coverage", type=int, required=True, help="Coverage the detected genomes were set to")
    ap.add_argument("--min-median-depth", type=int, required=True, help="Detection threshold on the median depth")
    ap.add_argument("--genomes", nargs="+", required=True, help="Genome names")
    ap.add_argument("--median-depths", nargs="+", type=int, required=True, help="Median depth per genome")
    ap.add_argument("--detected", nargs="+", required=True, choices=["true", "false"], help="Detected, per genome")
    ap.add_argument(
        "--concordance-h5s", nargs="*", default=[], help="sequins_vcfeval outputs of the detected genomes, in order"
    )
    args = ap.parse_args(argv)
    if not len(args.genomes) == len(args.median_depths) == len(args.detected):
        ap.error("--genomes, --median-depths and --detected must have the same length")
    if len(args.concordance_h5s) != args.detected.count("true"):
        ap.error("--concordance-h5s must have one file per detected genome")
    return args


def _table(conc: pd.DataFrame) -> str:
    return conc.to_html(index=False, escape=False, border=0)


def _esc(conc: pd.DataFrame) -> pd.DataFrame:
    return conc.astype(str).apply(lambda col: col.map(html.escape))


def metrics_table(conc: pd.DataFrame) -> pd.DataFrame:
    """TP/FP/FN, precision, recall and F1 per variant type, from the vcfeval CALL/BASE labels."""
    rows = []
    for name, is_indel in (("snps", False), ("indels", True)):
        d = conc[conc["indel"] == is_indel]
        tp_base, tp_call = (d["base"] == "TP").sum(), (d["call"] == "TP").sum()
        fp, fn = d["call"].isin(["FP", "FP_CA"]).sum(), d["base"].isin(["FN", "FN_CA"]).sum()
        precision = tp_call / (tp_call + fp) if tp_call + fp else 0.0
        recall = tp_base / (tp_base + fn) if tp_base + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append([name, tp_base, tp_call, fp, fn, f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}"])
    return pd.DataFrame(rows, columns=["type", "TP (truth)", "TP (call)", "FP", "FN", "precision", "recall", "F1"])


def discordant_table(conc: pd.DataFrame) -> pd.DataFrame:
    """The FP and FN records (FP first), each sorted by position."""
    rows = []
    for variant_class, mask in (
        ("FP", conc["call"].isin(["FP", "FP_CA"])),
        ("FN", conc["base"].isin(["FN", "FN_CA"])),
    ):
        for _, v in conc[mask].sort_values(["chrom", "pos"]).iterrows():
            ref, alts = v["alleles"][0], v["alleles"][1:]
            gt = v["gt_ultima"] if variant_class == "FP" else v["gt_ground_truth"]
            rows.append(
                [
                    variant_class,
                    "indels" if v["indel"] else "snps",
                    v["chrom"],
                    v["pos"],
                    ref,
                    ",".join(alts),
                    "/".join("." if a is None else str(a) for a in gt),
                    v.get("qual", "."),
                    v.get("filter", "."),
                    v.get("dp", "."),
                ]
            )
    return pd.DataFrame(rows, columns=DISCORDANT_COLUMNS)


def build_report(args: argparse.Namespace) -> str:
    title = html.escape(args.base_file_name)
    detected = [d == "true" for d in args.detected]
    names = [g for g, d in zip(args.genomes, detected, strict=True) if d]
    summary = pd.DataFrame(
        {
            "genome": [html.escape(g) for g in args.genomes],
            "median depth": args.median_depths,
            "detected": ['<span class="yes">yes</span>' if d else '<span class="no">no</span>' for d in detected],
        }
    )
    out = [
        f'<!DOCTYPE html>\n<html><head><meta charset="utf-8"><title>Sequins report: {title}</title>',
        f"<style>{CSS}</style></head><body>",
        f"<h1>Sequins report: {title}</h1>",
        "<h2>Input</h2>",
        "<p>Input CRAMs (unmapped reads and reads with an unmapped mate were re-aligned to the sequins genomes):</p>",
        "<ul>" + "".join(f"<li>{html.escape(c)}</li>" for c in args.input_crams) + "</ul>",
        f"<p>Downsampled to {args.target_coverage}x</p>",
        "<h2>Detected spikes</h2>",
        "<p>A sequins spike is detected when its median depth at the truth-variant sites (duplicates counted) "
        f"is above {args.min_median_depth}.</p>",
        f"<p>Detected: {html.escape(' '.join(names))}</p>" if names else "<p>No sequins spike was detected.</p>",
        _table(summary),
    ]
    h5_files = iter(args.concordance_h5s)  # only the detected genomes have one
    for genome, is_detected in zip(args.genomes, detected, strict=True):
        out.append(f"<h2>{html.escape(GENOME_TITLES.get(genome, genome))}</h2>")
        if not is_detected:
            out.append("<p>Not detected</p>")
            continue
        conc = pd.read_hdf(next(h5_files), key=H5_KEY)
        discordant = discordant_table(conc)
        out += [
            _table(metrics_table(conc)),
            "<h3>Discordant variants</h3>",
            _table(_esc(discordant)) if len(discordant) else "<p>None.</p>",
        ]
    out.append("</body></html>")
    return "\n".join(out) + "\n"


def run(argv: list[str]) -> str:
    args = parse_args(argv[1:])
    with open(args.output, "w") as fh:
        fh.write(build_report(args))
    return args.output


def main() -> None:
    run(sys.argv)


if __name__ == "__main__":
    main()
