"""Vocabulary configuration and channel constants for deep SRSNV tensors."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# Canonical channel order for x_num — positional channels first, then per-read constants.
# The model sees len(NUM_CHANNELS_POS) + len(NUM_CHANNELS_CONST) = NUMERIC_CHANNELS total.
NUM_CHANNELS_POS: list[str] = ["qual", "tp", "mask", "focus", "softclip_mask", "t0"]
NUM_CHANNELS_CONST: list[str] = ["strand", "mapq", "rq", "mixed"]
CHANNEL_ORDER: list[str] = NUM_CHANNELS_POS + NUM_CHANNELS_CONST
NUMERIC_CHANNELS: int = len(CHANNEL_ORDER)

_VOCAB_CONFIG_PATH = Path(__file__).parent.parent / "vocab_config.json"


@dataclass
class Encoders:
    base_vocab: dict[str, int]
    t0_vocab: dict[str, int]
    tm_vocab: dict[str, int]
    st_vocab: dict[str, int]
    et_vocab: dict[str, int]


def load_vocab_config(path: str | Path | None = None) -> Encoders:
    """Load static vocabulary mappings from a JSON config file.

    Falls back to the bundled ``vocab_config.json`` next to this module
    when *path* is ``None``.
    """
    cfg_path = Path(path) if path is not None else _VOCAB_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"Vocab config not found: {cfg_path}")
    try:
        raw = json.loads(cfg_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in vocab config {cfg_path}: {exc}") from exc

    required_keys = {"base_vocab", "t0_vocab", "tm_vocab", "st_vocab", "et_vocab"}
    missing = required_keys - set(raw.keys())
    if missing:
        raise ValueError(f"Vocab config missing keys: {sorted(missing)}")

    return Encoders(
        base_vocab=raw["base_vocab"],
        t0_vocab=raw["t0_vocab"],
        tm_vocab=raw["tm_vocab"],
        st_vocab=raw["st_vocab"],
        et_vocab=raw["et_vocab"],
    )
