from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from aurora.config import ExperimentConfig


@dataclass(frozen=True)
class RunArtifacts:
    root: Path
    config_path: Path
    metadata_path: Path
    metrics_path: Path
    checkpoints_dir: Path
    tokenizer_dir: Path
    result_path: Path


def create_run(config: ExperimentConfig, root: Path) -> RunArtifacts:
    root.mkdir(parents=True, exist_ok=True)
    run_root = root / config.experiment.run_id
    run_root.mkdir(exist_ok=False)
    checkpoints_dir = run_root / "checkpoints"
    tokenizer_dir = run_root / "tokenizer"
    checkpoints_dir.mkdir()
    tokenizer_dir.mkdir()
    artifacts = RunArtifacts(
        root=run_root,
        config_path=run_root / "config.yaml",
        metadata_path=run_root / "metadata.json",
        metrics_path=run_root / "metrics.jsonl",
        checkpoints_dir=checkpoints_dir,
        tokenizer_dir=tokenizer_dir,
        result_path=run_root / "result.json",
    )
    config.to_yaml(artifacts.config_path, include_hash=True)
    artifacts.metadata_path.write_text(
        json.dumps(
            {
                "config_hash": config.sha256(),
                "dataset_version": config.experiment.dataset_version,
                "run_id": config.experiment.run_id,
                "status": "initialized",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    artifacts.metrics_path.write_text("", encoding="utf-8")
    return artifacts


def append_jsonl(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True, separators=(",", ":")))
        handle.write("\n")
