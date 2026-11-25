"""Check a DeepVariant ONNX model against the make_examples settings of a variant-calling run.

The model input shape is (batch, height, width, channels). Models exported by
deepvariant-training also carry a ``metadata_props`` entry ``ultima_make_examples``
(JSON) with the make_examples settings they were trained on.

Checks
------
1. channels         FAIL  channel selection changes the channel count (needs metadata)
2. dimensions       FAIL  sum(sample_heights) / image width differ from the model input
3. sample_heights   WARN  rows per sample differ from training (needs metadata)
4. coverage         WARN  optimal_coverages / cap_at_optimal_coverage differ (needs metadata)
5. prioritize_alt   WARN  prioritize_alt_supporting_reads differs (needs metadata)
6. reference_rows   WARN  ensemble_reference_rows differs from training reference rows (needs metadata)
7. channel_names    WARN  selected channels differ (needs metadata)

WARN becomes FAIL with --strict. Checks that need metadata are SKIP for models without it.
Exit code is 1 when any check fails.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from dataclasses import asdict, dataclass

import onnx

METADATA_KEY = "ultima_make_examples"
SUPPORTED_SCHEMA_VERSION = 1
DEFAULT_IMAGE_WIDTH = 221

PASS, WARN, FAIL, SKIP = "PASS", "WARN", "FAIL", "SKIP"
METADATA_CHECKS = ("channels", "sample_heights", "coverage", "prioritize_alt", "reference_rows", "channel_names")


@dataclass
class CheckResult:
    name: str
    status: str
    message: str


def read_model(model_path: str) -> tuple[list, dict | None]:
    """Return (first input shape, ultima_make_examples metadata or None) of an ONNX model."""
    model = onnx.load(model_path, load_external_data=False)
    dims = model.graph.input[0].type.tensor_type.shape.dim
    shape = [d.dim_value if d.HasField("dim_value") else (d.dim_param or None) for d in dims]
    metadata = None
    for prop in model.metadata_props:
        if prop.key == METADATA_KEY:
            metadata = json.loads(prop.value)
    return shape, metadata


def parse_extra_args(extra_args: str | None) -> dict:
    """Split make_examples extra args into {flag: value-or-True}."""
    tokens = shlex.split(extra_args or "")
    out, i = {}, 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("--"):
            if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                out[tok] = tokens[i + 1]
                i += 2
                continue
            out[tok] = True
        i += 1
    return out


def _number(value):
    f = float(value)
    return int(f) if f.is_integer() else f


def _coverages(value) -> list | None:
    if value is None or value in ("", []):
        return None
    if isinstance(value, str):
        return [_number(v) for v in value.replace(",", ";").split(";") if v.strip()]
    if isinstance(value, list | tuple):
        return [_number(v) for v in value]
    return [_number(value)]


def normalize_run_settings(run: dict) -> dict:
    """Run settings in the metadata vocabulary; extra make_examples args override named inputs."""
    flags = parse_extra_args(run.get("make_examples_extra_args"))
    channels = flags.get("--channels", run.get("channels"))
    if isinstance(channels, str):
        channels = channels.split(",")
    reference_rows = run.get("ensemble_reference_rows")
    return {
        "sample_heights": [int(h) for h in run["sample_heights"]],
        "image_width": int(flags.get("--image-width", run.get("image_width") or DEFAULT_IMAGE_WIDTH)),
        "optimal_coverages": _coverages(flags.get("--optimal-coverages", run.get("optimal_coverages"))),
        "cap_at_optimal_coverage": bool(
            run.get("cap_at_optimal_coverage", False) or "--cap-at-optimal-coverage" in flags
        ),
        "prioritize_alt_supporting_reads": bool(
            run.get("prioritize_alt_supporting_reads", False) or "--prioritize-alt-supporting-reads" in flags
        ),
        "reference_rows": int(reference_rows) if reference_rows is not None else None,
        "add_ins_size_channel": bool(run.get("add_ins_size_channel", False) or "--add-ins-size-channel" in flags),
        "channels": channels or None,
    }


def _fixed(dim) -> bool:
    return isinstance(dim, int) and dim > 0


def _check_dimensions(shape: list, run: dict) -> CheckResult:
    height = sum(run["sample_heights"])
    problems = []
    if _fixed(shape[1]) and shape[1] != height:
        problems.append(f"model height {shape[1]} != sum(sample_heights {run['sample_heights']}) = {height}")
    if _fixed(shape[2]) and shape[2] != run["image_width"]:
        problems.append(f"model width {shape[2]} != run image width {run['image_width']}")
    if problems:
        return CheckResult("dimensions", FAIL, "; ".join(problems))
    if not (_fixed(shape[1]) and _fixed(shape[2])):
        return CheckResult("dimensions", WARN, f"model height/width not fixed in the ONNX input {shape}; not checked")
    return CheckResult("dimensions", PASS, f"height {height}, width {run['image_width']}")


def _check_channel_count(shape: list, metadata: dict, run: dict) -> CheckResult:
    trained_channels = metadata.get("image", {}).get("channels")
    problems = []
    if bool(metadata.get("add_ins_size_channel")) != run["add_ins_size_channel"]:
        problems.append(
            f"add_ins_size_channel: model {metadata.get('add_ins_size_channel')}, run {run['add_ins_size_channel']}"
        )
    if trained_channels and run["channels"] and len(trained_channels) != len(run["channels"]):
        problems.append(f"--channels: model {len(trained_channels)} channels, run {len(run['channels'])}")
    num_channels = metadata.get("image", {}).get("num_channels")
    if _fixed(shape[3]) and num_channels and shape[3] != num_channels:
        problems.append(f"model input has {shape[3]} channels, metadata says {num_channels}")
    if problems:
        return CheckResult("channels", FAIL, "; ".join(problems))
    return CheckResult("channels", PASS, f"{shape[3]} channels")


def run_checks(shape: list, metadata: dict | None, run: dict, *, strict: bool = False) -> list[CheckResult]:
    """Compare a model input shape and metadata with normalized run settings."""
    warn = FAIL if strict else WARN
    results = []

    if metadata is not None and metadata.get("schema_version") != SUPPORTED_SCHEMA_VERSION:
        results.append(
            CheckResult(
                "metadata",
                WARN,
                f"{METADATA_KEY} schema_version {metadata.get('schema_version')} not supported; shape checks only",
            )
        )
        metadata = None

    if len(shape) != 4:  # noqa: PLR2004
        message = f"expected a 4D (batch, height, width, channels) input: {shape}"
        results.append(CheckResult("dimensions", FAIL, message))
        return results
    results.append(_check_dimensions(shape, run))

    if metadata is None:
        results.extend(CheckResult(name, SKIP, f"model has no {METADATA_KEY} metadata") for name in METADATA_CHECKS)
        return results

    results.append(_check_channel_count(shape, metadata, run))

    def compare(name, model_value, run_value, label):
        if model_value == run_value:
            results.append(CheckResult(name, PASS, f"{label}: {run_value}"))
        else:
            results.append(CheckResult(name, warn, f"{label}: model {model_value}, run {run_value}"))

    compare("sample_heights", metadata.get("sample_heights"), run["sample_heights"], "sample_heights")
    compare(
        "coverage",
        [metadata.get("optimal_coverages"), bool(metadata.get("cap_at_optimal_coverage"))],
        [run["optimal_coverages"], run["cap_at_optimal_coverage"]],
        "[optimal_coverages, cap_at_optimal_coverage]",
    )
    compare(
        "prioritize_alt",
        bool(metadata.get("prioritize_alt_supporting_reads")),
        run["prioritize_alt_supporting_reads"],
        "prioritize_alt_supporting_reads",
    )
    if run["reference_rows"] is None:
        results.append(CheckResult("reference_rows", SKIP, "run did not report ensemble_reference_rows"))
    else:
        compare("reference_rows", metadata.get("reference_rows"), run["reference_rows"], "reference rows")
    compare(
        "channel_names",
        metadata.get("image", {}).get("channels"),
        run["channels"],
        "channels (null = make_examples default)",
    )
    return results


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="validate_model_geometry",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="ONNX model")
    parser.add_argument(
        "--run-settings",
        required=True,
        help="JSON with the run's make_examples settings: sample_heights (required), optimal_coverages, "
        "cap_at_optimal_coverage, prioritize_alt_supporting_reads, ensemble_reference_rows, "
        "add_ins_size_channel, make_examples_extra_args, image_width, channels",
    )
    parser.add_argument("--strict", action="store_true", help="treat warnings as failures")
    parser.add_argument("--output-json", help="write the check results here")
    return parser.parse_args(argv[1:])


def run(argv: list[str]) -> int:
    """Check an ONNX model against a run's make_examples settings; return the exit code."""
    args = parse_args(argv)
    with open(args.run_settings, encoding="utf-8") as f:
        settings = normalize_run_settings(json.load(f))
    shape, metadata = read_model(args.model)
    results = run_checks(shape, metadata, settings, strict=args.strict)

    print(f"Model input shape: {shape}")
    print(f"Model {METADATA_KEY}: {json.dumps(metadata, sort_keys=True) if metadata else 'none'}")
    print(f"Run settings: {json.dumps(settings, sort_keys=True)}")
    for r in results:
        print(f"{r.status:4}  {r.name:15} {r.message}")
    failed = [r.name for r in results if r.status == FAIL]
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_input_shape": shape,
                    "model_metadata": metadata,
                    "run_settings": settings,
                    "strict": args.strict,
                    "passed": not failed,
                    "checks": [asdict(r) for r in results],
                },
                f,
                indent=2,
            )
    if failed:
        print(f"ERROR: model does not match the run settings ({', '.join(failed)})", file=sys.stderr)
        return 1
    return 0


def main():
    sys.exit(run(sys.argv))


if __name__ == "__main__":
    main()
