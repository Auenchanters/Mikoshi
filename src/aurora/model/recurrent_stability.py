from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from aurora.model.recurrent import FixedLoopRecurrentTransformer, RecurrentLMOutput
from aurora.stability_config import StabilityModelConfig


@dataclass(frozen=True)
class StabilityLMOutput(RecurrentLMOutput):
    gate_diagnostics: tuple[object, ...] = ()


class ControlledRecurrentTransformer(FixedLoopRecurrentTransformer):
    """Task 06 recurrent controls with an exact delegated S0 path."""

    def __init__(self, config: StabilityModelConfig) -> None:
        super().__init__(config.as_recurrent_config())
        self.config = config
        self.stability_config = config

    def forward(
        self,
        input_ids: Tensor,
        targets: Tensor | None = None,
        *,
        collect_diagnostics: bool = False,
        retain_iteration_states: bool = False,
        num_iterations_override: int | None = None,
    ) -> StabilityLMOutput:
        if self.stability_config.variant != "S0" or num_iterations_override is not None:
            raise NotImplementedError(
                "controlled recurrent mechanisms are implemented in later tasks"
            )
        output: RecurrentLMOutput = super().forward(
            input_ids,
            targets=targets,
            collect_diagnostics=collect_diagnostics,
            retain_iteration_states=retain_iteration_states,
        )
        return StabilityLMOutput(
            logits=output.logits,
            loss=output.loss,
            iteration_diagnostics=output.iteration_diagnostics,
            recurrent_states=output.recurrent_states,
        )
