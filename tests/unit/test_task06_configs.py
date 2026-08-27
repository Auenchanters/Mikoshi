from __future__ import annotations

from pathlib import Path

import pytest

from aurora.recurrent_config import RecurrentExperimentConfig
from aurora.stability_config import StabilityExperimentConfig

_ROOT = Path(__file__).resolve().parents[2]
_TASK05_CONFIGS = {
    4: _ROOT / "configs" / "task05" / "recurrent_4.yaml",
    8: _ROOT / "configs" / "task05" / "recurrent_8.yaml",
}
_INDEPENDENT_RUNS = [
    ("s1_anchor_r4.yaml", "EXP-0614-anchor-r4", 4, "S1"),
    ("s1_anchor_r8.yaml", "EXP-0618-anchor-r8", 8, "S1"),
    ("s2_gate_r4.yaml", "EXP-0624-gate-r4", 4, "S2"),
    ("s2_gate_r8.yaml", "EXP-0628-gate-r8", 8, "S2"),
    ("s3_initial_rms_r4.yaml", "EXP-0634-initial-rms-r4", 4, "S3"),
    ("s3_initial_rms_r8.yaml", "EXP-0638-initial-rms-r8", 8, "S3"),
]


@pytest.mark.parametrize(
    ("filename", "run_id", "depth", "variant"),
    _INDEPENDENT_RUNS,
)
def test_task06_independent_config_preserves_matching_task05_protocol(
    filename: str,
    run_id: str,
    depth: int,
    variant: str,
) -> None:
    control = RecurrentExperimentConfig.from_yaml(_TASK05_CONFIGS[depth])
    candidate = StabilityExperimentConfig.from_yaml(_ROOT / "configs" / "task06" / filename)

    assert candidate.experiment.run_id == run_id
    assert candidate.experiment.dataset_version == control.experiment.dataset_version
    assert candidate.tokenizer == control.tokenizer
    assert candidate.data == control.data
    assert candidate.model.as_recurrent_config() == control.model
    assert candidate.optimizer == control.optimizer
    assert candidate.scheduler == control.scheduler
    assert candidate.runtime == control.runtime
    assert candidate.checkpoint == control.checkpoint
    assert candidate.model.variant == variant
