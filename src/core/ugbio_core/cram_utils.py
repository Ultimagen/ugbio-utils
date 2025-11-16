from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable

import pysam


def _extract_sample_names(cram_path: str) -> set[str]:
    with pysam.AlignmentFile(cram_path, "rc") as cram_file:
        read_groups = cram_file.header.get("RG", [])
    samples = {rg.get("SM") for rg in read_groups if rg.get("SM")}
    return samples


def check_cram_samples(
    cram_files: Iterable[str],
    *,
    require_unique: bool,
    raise_on_failure: bool,
) -> None:
    sample_to_files: dict[str, set[str]] = {}
    missing_samples: list[str] = []

    for path in cram_files:
        samples = _extract_sample_names(path)
        if not samples:
            missing_samples.append(path)
            continue
        for sample in samples:
            sample_to_files.setdefault(sample, set()).add(path)

    errors: list[str] = []

    if missing_samples:
        errors.append("Files with no sample names found: " + ", ".join(sorted(missing_samples)))

    if require_unique:
        duplicates = {s: sorted(paths) for s, paths in sample_to_files.items() if len(paths) > 1}
        if duplicates:
            formatted = "; ".join(f"{sample}: {', '.join(paths)}" for sample, paths in sorted(duplicates.items()))
            errors.append(f"Duplicate sample names detected across files: {formatted}")
    else:
        unique_samples = set(sample_to_files)
        if len(unique_samples) > 1:
            formatted = "; ".join(
                f"{sample}: {', '.join(sorted(paths))}" for sample, paths in sorted(sample_to_files.items())
            )
            errors.append(f"Multiple sample names detected; expected all identical. Details: {formatted}")

    if errors:
        combined_message = " | ".join(errors)
        if raise_on_failure:
            raise ValueError(combined_message)
        print(combined_message)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="cram-sample-check",
        description=("Validate CRAM sample names by ensuring they are unique across files or identical across files."),
    )
    parser.add_argument(
        "cram_files",
        nargs="+",
        help="Paths to CRAM files to inspect.",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--require-unique",
        action="store_true",
        help="Require that sample names are unique across CRAM files.",
    )
    mode_group.add_argument(
        "--require-identical",
        action="store_true",
        help="Require that sample names are identical across CRAM files (default).",
    )
    parser.add_argument(
        "--raise-on-failure",
        action="store_true",
        help="Raise an exception instead of printing an error message when validation fails.",
    )
    return parser.parse_args(argv[1:])


def run(argv: list[str]) -> None:
    args = _parse_args(argv)
    require_unique = args.require_unique and not args.require_identical
    check_cram_samples(
        args.cram_files,
        require_unique=require_unique,
        raise_on_failure=args.raise_on_failure,
    )


def main() -> None:
    run(sys.argv)


if __name__ == "__main__":
    main()
