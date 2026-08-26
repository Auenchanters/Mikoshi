from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from conftest import tiny_config_dict

from aurora.config import ExperimentConfig
from aurora.experiment import append_jsonl, create_run


def config_from_payload(tmp_path: Path) -> ExperimentConfig:
    path = tmp_path / "input.yaml"
    path.write_text(yaml.safe_dump(tiny_config_dict()), encoding="utf-8")
    return ExperimentConfig.from_yaml(path)


def test_create_run_writes_resolved_config_and_refuses_overwrite(tmp_path: Path) -> None:
    config = config_from_payload(tmp_path)

    run = create_run(config, tmp_path / "runs")

    resolved = yaml.safe_load(run.config_path.read_text(encoding="utf-8"))
    assert resolved["config_hash"] == config.sha256()
    assert resolved["experiment"]["run_id"] == "EXP-0001-dense-sanity"
    assert run.checkpoints_dir.is_dir()
    assert run.tokenizer_dir.is_dir()
    with pytest.raises(FileExistsError):
        create_run(config, tmp_path / "runs")


def test_append_jsonl_preserves_each_machine_readable_record(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"

    append_jsonl(path, {"step": 1, "loss": 4.0})
    append_jsonl(path, {"step": 2, "loss": 3.5})

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert records == [{"loss": 4.0, "step": 1}, {"loss": 3.5, "step": 2}]
