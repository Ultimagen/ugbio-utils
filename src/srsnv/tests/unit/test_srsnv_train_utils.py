import json
from pathlib import Path
from typing import Any

import pytest
from ugbio_srsnv.srsnv_training import _parse_model_params


@pytest.fixture
def resources_dir():
    return Path(__file__).parent.parent / "resources"


@pytest.mark.parametrize(
    "raw, expected",
    [
        (None, {}),
        (
            "eta=0.1:max_depth=8:n_estimators=200",
            {"eta": 0.1, "max_depth": 8, "n_estimators": 200},
        ),
        ("verbosity=debug:subsample=0.9", {"verbosity": "debug", "subsample": 0.9}),
    ],
)
def test_parse_model_params_inline(raw: str | None, expected: dict[str, Any]) -> None:
    assert _parse_model_params(raw) == expected


def test_parse_model_params_json(tmp_path: Path) -> None:  # noqa: D103
    json_path = tmp_path / "xgb_params.json"
    payload = {"eta": 0.05, "max_depth": 6, "enable_categorical": True}
    json_path.write_text(json.dumps(payload))

    parsed = _parse_model_params(str(json_path))
    assert parsed == payload


def test_parse_model_params_invalid() -> None:  # noqa: D103
    with pytest.raises(ValueError):
        _parse_model_params("eta=0.1:max_depth")  # uneven tokens
    with pytest.raises(ValueError):
        _parse_model_params("eta")  # missing '='
