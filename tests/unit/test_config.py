from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from conftest import tiny_config_dict

from aurora.config import ConfigError, ExperimentConfig


def write_config(path: Path, payload: dict[str, object]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_config_yaml_round_trip_has_stable_hash(tmp_path: Path) -> None:
    source = tmp_path / "source.yaml"
    write_config(source, tiny_config_dict())
    config = ExperimentConfig.from_yaml(source)
    saved = tmp_path / "saved.yaml"
    config.to_yaml(saved)

    loaded = ExperimentConfig.from_yaml(saved)

    assert loaded == config
    assert loaded.sha256() == config.sha256()
    assert len(config.sha256()) == 64


def test_unknown_top_level_config_key_is_rejected(tmp_path: Path) -> None:
    payload = tiny_config_dict()
    payload["unknown"] = True
    path = tmp_path / "bad.yaml"
    write_config(path, payload)

    with pytest.raises(ConfigError, match="unknown top-level keys: unknown"):
        ExperimentConfig.from_yaml(path)


def test_unknown_nested_config_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    write_config(path, tiny_config_dict(model={"mystery_width": 12}))

    with pytest.raises(ConfigError, match=r"model.*mystery_width"):
        ExperimentConfig.from_yaml(path)


def test_invalid_grouped_query_attention_shape_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    write_config(
        path,
        tiny_config_dict(model={"d_model": 30, "num_heads": 3, "num_kv_heads": 2}),
    )

    with pytest.raises(ConfigError, match=r"num_heads.*num_kv_heads"):
        ExperimentConfig.from_yaml(path)


def test_byte_tokenizer_cannot_be_selected_in_production_config(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    write_config(path, tiny_config_dict(tokenizer={"kind": "byte"}))

    with pytest.raises(ConfigError, match=r"tokenizer\.kind must be 'bpe'"):
        ExperimentConfig.from_yaml(path)
