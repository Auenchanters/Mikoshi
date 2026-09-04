from __future__ import annotations

from aurora.training.stability_metrics import estimate_stability_transformer_flops
from tests.stability_helpers import stability_model_config


def test_stability_flops_add_only_enabled_control_matmuls() -> None:
    common = {"batch_size": 2, "sequence_length": 8, "training": False}
    s0 = estimate_stability_transformer_flops(stability_model_config(num_iterations=2), **common)
    s1 = estimate_stability_transformer_flops(
        stability_model_config(num_iterations=2, input_anchoring=True), **common
    )
    s2 = estimate_stability_transformer_flops(
        stability_model_config(num_iterations=2, gated_update=True), **common
    )
    s3 = estimate_stability_transformer_flops(
        stability_model_config(num_iterations=2, state_stabilization="initial_rms"),
        **common,
    )
    s4 = estimate_stability_transformer_flops(
        stability_model_config(
            num_iterations=2,
            input_anchoring=True,
            gated_update=True,
        ),
        **common,
    )

    assert s0 == 1_310_720
    assert s1 == 1_376_256
    assert s2 == 1_314_816
    assert s3 == s0
    assert s4 == 1_380_352


def test_stability_training_flops_use_three_forward_equivalents() -> None:
    config = stability_model_config(num_iterations=4, gated_update=True)

    forward = estimate_stability_transformer_flops(
        config, batch_size=1, sequence_length=8, training=False
    )
    training = estimate_stability_transformer_flops(
        config, batch_size=1, sequence_length=8, training=True
    )

    assert training == 3 * forward
