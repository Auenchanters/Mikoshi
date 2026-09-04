from __future__ import annotations

from copy import deepcopy

import pytest

from aurora.config import ConfigError
from aurora.recurrent_config import RecurrentModelConfig
from aurora.stability_config import StabilityExperimentConfig
from tests.stability_helpers import stability_experiment_payload, stability_model_config


@pytest.mark.parametrize(
    ("settings", "expected_variant"),
    [
        ({}, "S0"),
        ({"input_anchoring": True}, "S1"),
        ({"gated_update": True}, "S2"),
        ({"state_stabilization": "initial_rms"}, "S3"),
        ({"input_anchoring": True, "gated_update": True}, "S4"),
    ],
)
def test_stability_model_config_resolves_each_allowed_variant(
    settings: dict[str, object], expected_variant: str
) -> None:
    config = stability_model_config(**settings)

    assert config.variant == expected_variant
    recurrent = config.as_recurrent_config()
    assert type(recurrent) is RecurrentModelConfig
    assert recurrent.num_iterations == 4
    assert recurrent.max_iterations == 8


@pytest.mark.parametrize(
    "settings",
    [
        {"input_anchoring": True, "state_stabilization": "initial_rms"},
        {"gated_update": True, "state_stabilization": "initial_rms"},
        {
            "input_anchoring": True,
            "gated_update": True,
            "state_stabilization": "initial_rms",
        },
        {"state_stabilization": "unknown"},
    ],
)
def test_stability_experiment_rejects_unrequested_variant_combinations(
    settings: dict[str, object],
) -> None:
    with pytest.raises(ConfigError, match="stability"):
        StabilityExperimentConfig.from_dict(stability_experiment_payload(**settings))


def test_stability_experiment_rejects_training_depth_three() -> None:
    with pytest.raises(ConfigError, match="num_iterations"):
        StabilityExperimentConfig.from_dict(stability_experiment_payload(num_iterations=3))


@pytest.mark.parametrize("field", ["input_anchoring", "gated_update"])
def test_stability_experiment_requires_boolean_mechanism_flags(field: str) -> None:
    with pytest.raises(ConfigError, match=field):
        StabilityExperimentConfig.from_dict(stability_experiment_payload(**{field: 1}))


def test_stability_experiment_retains_strict_unknown_key_and_hash_validation() -> None:
    payload = stability_experiment_payload()
    payload["model"]["unrequested"] = True  # type: ignore[index]

    with pytest.raises(ConfigError, match="unknown"):
        StabilityExperimentConfig.from_dict(payload)

    resolved = StabilityExperimentConfig.from_dict(stability_experiment_payload())
    hashed = deepcopy(stability_experiment_payload())
    hashed["config_hash"] = "not-the-resolved-hash"

    with pytest.raises(ConfigError, match="config_hash"):
        StabilityExperimentConfig.from_dict(hashed)

    assert resolved.model.variant == "S0"


def test_inference_depth_three_is_not_a_stability_configuration_value() -> None:
    payload = stability_experiment_payload()
    payload["model"]["num_iterations_override"] = 3  # type: ignore[index]

    with pytest.raises(ConfigError, match="unknown"):
        StabilityExperimentConfig.from_dict(payload)
