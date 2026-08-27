from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from aurora.model.norms import RMSNorm
from aurora.model.recurrent_diagnostics import (
    RecurrentStateDiagnostics,
    measure_recurrent_state,
)
from aurora.model.transformer import DecoderBlock, DenseTransformer
from aurora.recurrent_config import RecurrentModelConfig


@dataclass(frozen=True)
class RecurrentLMOutput:
    logits: Tensor
    loss: Tensor | None
    iteration_diagnostics: tuple[RecurrentStateDiagnostics, ...] = ()
    recurrent_states: tuple[Tensor, ...] = ()


class TransformerStage(nn.Module):
    def __init__(self, config: RecurrentModelConfig, num_layers: int) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("TransformerStage requires at least one layer")
        block_config = config.as_block_config()
        self.blocks = nn.ModuleList([DecoderBlock(block_config) for _ in range(num_layers)])

    def forward(self, hidden: Tensor) -> Tensor:
        for block in self.blocks:
            hidden = block(hidden)
        return hidden


class RecurrentCore(nn.Module):
    """One parameter set invoked repeatedly at fixed recurrent depth."""

    def __init__(self, config: RecurrentModelConfig) -> None:
        super().__init__()
        self.max_iterations = config.max_iterations
        self.step_embedding = nn.Embedding(config.max_iterations, config.d_model)
        self.stage = TransformerStage(config, config.num_core_layers)

    def forward(self, hidden: Tensor, iteration: int = 0) -> Tensor:
        if type(iteration) is not int or not 0 <= iteration < self.max_iterations:
            raise ValueError("iteration must be within the configured recurrent range")
        step = self.step_embedding.weight[iteration].view(1, 1, -1)
        return self.stage(hidden + step)


class FixedLoopRecurrentTransformer(nn.Module):
    """Prelude/shared-core/coda causal LM with an externally fixed loop count."""

    def __init__(self, config: RecurrentModelConfig) -> None:
        super().__init__()
        if config.num_iterations > config.max_iterations:
            raise ValueError("num_iterations cannot exceed max_iterations")
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.prelude = TransformerStage(config, config.num_prelude_layers)
        self.core = RecurrentCore(config)
        self.coda = TransformerStage(config, config.num_coda_layers)
        self.final_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(DenseTransformer._initialize_weights)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    def _validate_inputs(self, input_ids: Tensor, targets: Tensor | None) -> None:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.dtype != torch.long:
            raise ValueError("input_ids must use torch.long token IDs")
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("input sequence length exceeds model.max_seq_len")
        if targets is not None:
            if targets.shape != input_ids.shape:
                raise ValueError("targets must have the same shape as input_ids")
            if targets.dtype != torch.long:
                raise ValueError("targets must use torch.long token IDs")

    def forward(
        self,
        input_ids: Tensor,
        targets: Tensor | None = None,
        *,
        collect_diagnostics: bool = False,
        retain_iteration_states: bool = False,
    ) -> RecurrentLMOutput:
        self._validate_inputs(input_ids, targets)
        hidden = self.embedding_dropout(self.token_embedding(input_ids))
        hidden = self.prelude(hidden)
        initial_state = hidden
        diagnostics: list[RecurrentStateDiagnostics] = []
        recurrent_states = [hidden] if retain_iteration_states else []
        for iteration in range(self.config.num_iterations):
            previous = hidden
            hidden = self.core(previous, iteration=iteration)
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
        return RecurrentLMOutput(
            logits=logits,
            loss=loss,
            iteration_diagnostics=tuple(diagnostics),
            recurrent_states=tuple(recurrent_states),
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_k: int | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        if input_ids.ndim != 2 or input_ids.dtype != torch.long:
            raise ValueError("generation input_ids must be a [batch, sequence] long tensor")
        if input_ids.shape[1] == 0:
            raise ValueError("generation requires at least one prompt token")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be nonnegative")
        if temperature < 0.0:
            raise ValueError("temperature must be nonnegative")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive")
        generated = input_ids.clone()
        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                context = generated[:, -self.config.max_seq_len :]
                next_logits = self(context).logits[:, -1, :].float()
                if temperature == 0.0:
                    next_token = next_logits.argmax(dim=-1, keepdim=True)
                else:
                    scaled = next_logits / temperature
                    if top_k is not None:
                        resolved_top_k = min(top_k, scaled.shape[-1])
                        threshold = torch.topk(scaled, resolved_top_k, dim=-1).values[:, -1:]
                        scaled = scaled.masked_fill(scaled < threshold, float("-inf"))
                    probabilities = functional.softmax(scaled, dim=-1)
                    next_token = torch.multinomial(
                        probabilities,
                        num_samples=1,
                        generator=generator,
                    )
                generated = torch.cat((generated, next_token), dim=1)
        finally:
            self.train(was_training)
        return generated
