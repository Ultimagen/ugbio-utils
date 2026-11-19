"""Run ``pypgx run-ngs-pipeline`` for many genes of one sample and collect their outputs.

Ports the per-gene file glue of terra_pipeline wdls/pypgx.wdl NGSPipeline and its ConcatResults TSV:
pypgx archives are renamed to ``<kind>.<gene>.zip`` (empty when pypgx skipped them), the first profile
PNGs become ``<profile>.<gene>.png``, and the TSV holds the last data.tsv row of each gene's results.zip.
A failed gene still lets the other genes finish, but then nothing is written to the output dir: like
EDV, a failed sample has no final output.
"""

import argparse
import shutil
import subprocess
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ZIP_KINDS = (
    "alleles",
    "cnv-calls",
    "consolidated-variants",
    "copy-number",
    "genotypes",
    "imported-variants",
    "phased-variants",
    "phenotypes",
    "read-depth",
)
PROFILE_KINDS = ("copy-number-profile", "allele-fraction-profile")
TSV_HEADER = "Symbol\tSampleName\tGenotype\tPhenotype\tHaplotype1\tHaplotype2\tAlternativePhase\tVariantData\tCNV"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="pgx_ngs_pipeline", description=__doc__.splitlines()[0])
    parser.add_argument("--genes", nargs="+", required=True)
    parser.add_argument("--parallel", type=int, required=True, help="genes run at once")
    parser.add_argument("--assembly", required=True)
    parser.add_argument("--variants", required=True, help="PASS-only VCF")
    parser.add_argument("--depth-of-coverage", required=True)
    parser.add_argument("--control-statistics", required=True)
    parser.add_argument("--ngs-dir", type=Path, required=True, help="pypgx writes each gene into <ngs-dir>/<gene>")
    parser.add_argument("--log-dir", type=Path, required=True, help="gets ngs.<gene>.log")
    parser.add_argument("--output-dir", type=Path, required=True, help="gets the WDL-named zips and PNGs")
    parser.add_argument("--tsv", type=Path, required=True)
    args = parser.parse_args(argv)
    duplicates = sorted({gene for gene in args.genes if args.genes.count(gene) > 1})
    if duplicates:
        parser.error(f"duplicate genes: {' '.join(duplicates)}")
    return args


def run_gene(gene: str, args: argparse.Namespace) -> int:
    command = [
        "pypgx",
        "run-ngs-pipeline",
        gene,
        str(args.ngs_dir / gene),
        *["--assembly", args.assembly, "--variants", args.variants],
        *["--depth-of-coverage", args.depth_of_coverage, "--control-statistics", args.control_statistics],
        # replaces a failed earlier attempt's dir, which pypgx otherwise refuses
        "--force",
    ]
    with open(args.log_dir / f"ngs.{gene}.log", "wb") as log:
        return subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False).returncode


def collect_gene_outputs(gene: str, gene_dir: Path, output_dir: Path):
    for kind in ZIP_KINDS:
        source, target = gene_dir / f"{kind}.zip", output_dir / f"{kind}.{gene}.zip"
        if source.is_file():
            shutil.move(source, target)
        else:
            target.write_bytes(b"")
    for kind in PROFILE_KINDS:
        pngs, target = sorted((gene_dir / kind).glob("*.png")), output_dir / f"{kind}.{gene}.png"
        if pngs:
            shutil.copyfile(pngs[0], target)
        else:
            target.write_bytes(b"")


def last_row(results_zip: Path) -> str:
    # What the WDL's `pypgx print-data results.zip | tail -1` prints.
    if not zipfile.is_zipfile(results_zip):
        raise ValueError(f"{results_zip} is not a zip")
    with zipfile.ZipFile(results_zip) as archive:
        member = next((name for name in archive.namelist() if name.endswith("data.tsv")), None)
        if member is None:
            raise ValueError(f"no data.tsv in {results_zip}")
        rows = [line for line in archive.read(member).decode().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty data.tsv in {results_zip}")
    return rows[-1]


def run(argv: list[str]) -> int:
    args = parse_args(argv[1:])
    for directory in (args.ngs_dir, args.log_dir, args.output_dir, args.tsv.parent):
        directory.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        return_codes = dict(zip(args.genes, pool.map(lambda gene: run_gene(gene, args), args.genes), strict=True))
    failed = [gene for gene, code in return_codes.items() if code != 0]
    if failed:
        print("pypgx run-ngs-pipeline failed for:", file=sys.stderr)
        for gene in failed:
            print(f"  {gene}: {args.log_dir / f'ngs.{gene}.log'}", file=sys.stderr)
        return 1

    rows = [TSV_HEADER, *(f"{gene}\t{last_row(args.ngs_dir / gene / 'results.zip')}" for gene in args.genes)]
    for gene in args.genes:
        collect_gene_outputs(gene, args.ngs_dir / gene, args.output_dir)
    args.tsv.write_text("\n".join(rows) + "\n")
    return 0


def main():
    sys.exit(run(sys.argv))


if __name__ == "__main__":
    main()
