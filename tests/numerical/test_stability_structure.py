from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from aurora.model.recurrent_stability import ControlledRecurrentTransformer
from tests.stability_helpers import stability_model_config

VARIANTS = ("S0", "S1", "S2", "S3", "S4")
DEPTHS = (1, 2, 4, 8)


def _model(variant: str, *, num_iterations: int) -> ControlledRecurrentTransformer:
    if variant == "S0":
        config = stability_model_config(num_iterations=num_iterations)
    elif variant == "S1":
        config = stability_model_config(num_iterations=num_iterations, input_anchoring=True)
    elif variant == "S2":
        config = stability_model_config(num_iterations=num_iterations, gated_update=True)
    elif variant == "S3":
        config = stability_model_config(
            num_iterations=num_iterations, state_stabilization="initial_rms"
        )
    elif variant == "S4":
        config = stability_model_config(
            num_iterations=num_iterations, input_anchoring=True, gated_update=True
        )
    else:
        raise AssertionError(f"unknown test variant {variant}")
    model = ControlledRecurrentTransformer(config)
    model.cpu()
    model.eval()
    return model


def _storage_pointers(parameters: tuple[nn.Parameter, ...]) -> tuple[int, ...]:
    return tuple(parameter.untyped_storage().data_ptr() for parameter in parameters)


def test_s3_applies_stabilization_after_each_core_proposal() -> None:
    model = _model("S3", num_iterations=4)
    events: list[str] = []
    core_inputs: list[Tensor] = []
    core_outputs: list[Tensor] = []
    stabilizer_inputs: list[tuple[Tensor, Tensor]] = []
    stabilizer_outputs: list[Tensor] = []

    def record_core(
        _module: nn.Module, inputs: tuple[Tensor, ...], output: Tensor
    ) -> None:
        events.append("core")
        core_inputs.append(inputs[0])
        core_outputs.append(output)

    def record_stabilizer(
        _module: nn.Module, inputs: tuple[Tensor, Tensor], output: Tensor
    ) -> None:
        events.append("stabilizer")
        stabilizer_inputs.append(inputs)
        stabilizer_outputs.append(output)

    core_hook = model.core.register_forward_hook(record_core)
    stabilizer_hook = model.state_stabilizer.register_forward_hook(record_stabilizer)
    try:
        output = model(
            torch.tensor([[1, 4, 7, 3, 9, 2]], dtype=torch.long),
            retain_iteration_states=True,
        )
    finally:
        core_hook.remove()
        stabilizer_hook.remove()

    assert events == ["core", "stabilizer"] * 4
    assert all(inputs[0] is output.recurrent_states[0] for inputs in stabilizer_inputs)
    assert all(
        inputs[1] is proposal
        for inputs, proposal in zip(stabilizer_inputs, core_outputs, strict=True)
    )
    assert all(
        stabilized is retained
        for stabilized, retained in zip(
            stabilizer_outputs, output.recurrent_states[1:], strict=True
        )
    )
    assert core_inputs[0] is output.recurrent_states[0]
    assert all(
        stabilized is next_core_input
        for stabilized, next_core_input in zip(
            stabilizer_outputs[:-1], core_inputs[1:], strict=True
        )
    )
    initial_rms = torch.linalg.vector_norm(
        output.recurrent_states[0].float(), dim=-1
    ) / model.config.d_model**0.5
    for state in output.recurrent_states[1:]:
        state_rms = torch.linalg.vector_norm(state.float(), dim=-1) / model.config.d_model**0.5
        assert torch.allclose(state_rms, initial_rms, atol=1e-5, rtol=1e-5)


def test_s4_composes_anchor_core_gate_without_stabilization() -> None:
    model = _model("S4", num_iterations=2)
    events: list[str] = []
    anchor_inputs: list[tuple[Tensor, Tensor]] = []
    anchor_outputs: list[Tensor] = []
    core_inputs: list[Tensor] = []
    core_outputs: list[Tensor] = []
    gate_inputs: list[tuple[Tensor, Tensor]] = []
    gate_outputs: list[tuple[Tensor, Tensor]] = []

    def record_anchor(
        _module: nn.Module, inputs: tuple[Tensor, Tensor], output: Tensor
    ) -> None:
        events.append("anchor")
        anchor_inputs.append(inputs)
        anchor_outputs.append(output)

    def record_core_input(_module: nn.Module, inputs: tuple[Tensor, ...]) -> None:
        events.append("core")
        core_inputs.append(inputs[0])

    def record_core_output(
        _module: nn.Module, _inputs: tuple[Tensor, ...], output: Tensor
    ) -> None:
        core_outputs.append(output)

    def record_gate(
        _module: nn.Module,
        inputs: tuple[Tensor, Tensor],
        outputs: tuple[Tensor, Tensor],
    ) -> None:
        events.append("gate")
        gate_inputs.append(inputs)
        gate_outputs.append(outputs)

    hooks = (
        model.input_anchor.register_forward_hook(record_anchor),
        model.core.register_forward_pre_hook(record_core_input),
        model.core.register_forward_hook(record_core_output),
        model.gated_update.register_forward_hook(record_gate),
    )
    try:
        output = model(
            torch.tensor([[1, 4, 7, 3]], dtype=torch.long),
            retain_iteration_states=True,
        )
    finally:
        for hook in hooks:
            hook.remove()

    assert not hasattr(model, "state_stabilizer")
    assert events == ["anchor", "core", "gate"] * 2
    assert all(
        core_input is anchor_output
        for core_input, anchor_output in zip(core_inputs, anchor_outputs, strict=True)
    )
    assert all(
        gate_input[1] is core_output
        for gate_input, core_output in zip(gate_inputs, core_outputs, strict=True)
    )
    assert all(
        gate_input[0] is retained_previous
        for gate_input, retained_previous in zip(
            gate_inputs, output.recurrent_states[:-1], strict=True
        )
    )
    assert all(
        gate_output[0] is retained
        for gate_output, retained in zip(
            gate_outputs, output.recurrent_states[1:], strict=True
        )
    )
    assert all(
        anchor_input[1] is retained_previous
        for anchor_input, retained_previous in zip(
            anchor_inputs, output.recurrent_states[:-1], strict=True
        )
    )
    assert all(
        gate_output[0] is next_anchor_input[1]
        for gate_output, next_anchor_input in zip(
            gate_outputs[:-1], anchor_inputs[1:], strict=True
        )
    )


@pytest.mark.parametrize("variant", VARIANTS)
def test_parameter_count_and_state_keys_are_invariant_with_depth(variant: str) -> None:
    models = [_model(variant, num_iterations=depth) for depth in DEPTHS]
    expected_parameter_count = sum(parameter.numel() for parameter in models[0].parameters())
    expected_state_keys = set(models[0].state_dict())

    assert all(
        sum(parameter.numel() for parameter in model.parameters()) == expected_parameter_count
        for model in models
    )
    assert all(set(model.state_dict()) == expected_state_keys for model in models)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("depth", DEPTHS)
def test_one_shared_core_object_and_storage_are_reused_each_iteration(
    variant: str, depth: int
) -> None:
    model = _model(variant, num_iterations=depth)
    core = model.core
    core_parameters = tuple(core.parameters())
    expected_parameter_ids = tuple(id(parameter) for parameter in core_parameters)
    expected_storage_pointers = _storage_pointers(core_parameters)
    observations: list[tuple[int, tuple[int, ...], tuple[int, ...]]] = []

    def record_core(module: nn.Module, _inputs: tuple[Tensor, ...]) -> None:
        parameters = tuple(module.parameters())
        observations.append(
            (
                id(module),
                tuple(id(parameter) for parameter in parameters),
                _storage_pointers(parameters),
            )
        )

    hook = core.register_forward_pre_hook(record_core)
    try:
        model(torch.tensor([[1, 4, 7, 3]], dtype=torch.long))
    finally:
        hook.remove()

    assert observations == [
        (id(core), expected_parameter_ids, expected_storage_pointers)
    ] * depth


@pytest.mark.parametrize("variant", VARIANTS)
def test_optimizer_contains_every_trainable_parameter_exactly_once(variant: str) -> None:
    model = _model(variant, num_iterations=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model_parameter_ids = [id(parameter) for parameter in model.parameters()]
    optimizer_parameter_ids = [
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]

    assert len(model_parameter_ids) == len(set(model_parameter_ids))
    assert len(optimizer_parameter_ids) == len(set(optimizer_parameter_ids))
    assert set(optimizer_parameter_ids) == set(model_parameter_ids)


@pytest.mark.parametrize("variant", VARIANTS)
def test_absent_inference_override_uses_configured_depth(variant: str) -> None:
    model = _model(variant, num_iterations=4)
    input_ids = torch.tensor([[1, 4, 7, 3]], dtype=torch.long)

    configured = model(input_ids, retain_iteration_states=True)

    assert len(configured.recurrent_states) == 5


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("accepted_depth", range(1, 9))
def test_inference_override_accepts_each_integer_through_maximum(
    variant: str, accepted_depth: int
) -> None:
    model = _model(variant, num_iterations=4)
    input_ids = torch.tensor([[1, 4, 7, 3]], dtype=torch.long)

    overridden = model(
        input_ids,
        retain_iteration_states=True,
        num_iterations_override=accepted_depth,
    )

    assert len(overridden.recurrent_states) == accepted_depth + 1


@pytest.mark.parametrize("invalid_depth", [0, 9, True, 1.0, "4"])
def test_inference_override_rejects_non_integer_or_out_of_range_depth(
    invalid_depth: object,
) -> None:
    model = _model("S4", num_iterations=4)

    with pytest.raises(ValueError, match="num_iterations_override"):
        model(
            torch.tensor([[1, 4, 7, 3]], dtype=torch.long),
            num_iterations_override=invalid_depth,
        )
