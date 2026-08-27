from __future__ import annotations

from dataclasses import dataclass

import torch.nn.functional as functional
from torch import Tensor, nn

from aurora.model.recurrent import FixedLoopRecurrentTransformer, RecurrentLMOutput
from aurora.model.recurrent_diagnostics import measure_recurrent_state
from aurora.model.transformer import DenseTransformer
from aurora.stability_config import StabilityModelConfig


@dataclass(frozen=True)
class StabilityLMOutput(RecurrentLMOutput):
    gate_diagnostics: tuple[object, ...] = ()


class InputAnchor(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.projection = nn.Linear(d_model, d_model, bias=False)
        DenseTransformer._initialize_weights(self.projection)

    def forward(self, initial_state: Tensor, recurrent_state: Tensor) -> Tensor:
        return recurrent_state + self.projection(initial_state)


class ControlledRecurrentTransformer(FixedLoopRecurrentTransformer):
    """Task 06 recurrent controls with an exact delegated S0 path."""

    def __init__(self, config: StabilityModelConfig) -> None:
        super().__init__(config.as_recurrent_config())
        self.config = config
        self.stability_config = config
        if config.input_anchoring:
            self.input_anchor = InputAnchor(config.d_model)

    def forward(
        self,
        input_ids: Tensor,
        targets: Tensor | None = None,
        *,
        collect_diagnostics: bool = False,
        retain_iteration_states: bool = False,
        num_iterations_override: int | None = None,
    ) -> StabilityLMOutput:
        if self.stability_config.variant == "S0" and num_iterations_override is None:
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
        if self.stability_config.variant != "S1" or num_iterations_override is not None:
            raise NotImplementedError(
                "controlled recurrent mechanisms are implemented in later tasks"
            )
        self._validate_inputs(input_ids, targets)
        hidden = self.embedding_dropout(self.token_embedding(input_ids))
        hidden = self.prelude(hidden)
        initial_state = hidden
        diagnostics = []
        recurrent_states = [hidden] if retain_iteration_states else []
        for iteration in range(self.config.num_iterations):
            previous = hidden
            core_input = self.input_anchor(initial_state, previous)
            proposal = self.core(core_input, iteration=iteration)
            hidden = proposal
            if collect_diagnostics:
                diagnostics.append(
                    measure_recurrent_state(
                        initial_state,
                        previous,
                        hidden,
                        iteration=iteration + 1,
                    )
                )
            if retain_iteration_states:
                recurrent_states.append(hidden)
        hidden = self.coda(hidden)
        logits = self.lm_head(self.final_norm(hidden))
        loss = None
        if targets is not None:
            loss = functional.cross_entropy(
                logits.float().reshape(-1, self.config.vocab_size),
                targets.reshape(-1),
            )
        output = RecurrentLMOutput(
            logits=logits,
            loss=loss,
            iteration_diagnostics=tuple(diagnostics),
            recurrent_states=tuple(recurrent_states),
        )
        return StabilityLMOutput(
            logits=output.logits,
            loss=output.loss,
            iteration_diagnostics=output.iteration_diagnostics,
            recurrent_states=output.recurrent_states,
        )
