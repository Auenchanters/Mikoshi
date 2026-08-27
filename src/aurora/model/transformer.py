from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from aurora.config import ModelConfig
from aurora.model.attention import GroupedQueryAttention
from aurora.model.mlp import SwiGLU
from aurora.model.norms import RMSNorm


@dataclass(frozen=True)
class LMOutput:
    logits: Tensor
    loss: Tensor | None


class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.attention = GroupedQueryAttention(config)
        self.mlp_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.mlp = SwiGLU(config.d_model, config.d_ff, dropout=config.dropout)

    def forward(self, hidden: Tensor) -> Tensor:
        hidden = hidden + self.attention(self.attention_norm(hidden))
        return hidden + self.mlp(self.mlp_norm(hidden))


class DenseTransformer(nn.Module):
    """Modern dense decoder-only Transformer used as the research control."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.vocab_size <= 0 or config.max_seq_len <= 0:
            raise ValueError("vocab_size and max_seq_len must be positive")
        if config.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.num_layers)])
        self.final_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(self._initialize_weights)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: Tensor, targets: Tensor | None = None) -> LMOutput:
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
        hidden = self.embedding_dropout(self.token_embedding(input_ids))
        for block in self.blocks:
            hidden = block(hidden)
        logits = self.lm_head(self.final_norm(hidden))
        loss = None
        if targets is not None:
            loss = functional.cross_entropy(
                logits.float().reshape(-1, self.config.vocab_size),
                targets.reshape(-1),
            )
        return LMOutput(logits=logits, loss=loss)

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
