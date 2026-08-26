"""Dense-control and fixed-loop recurrent Transformer modules."""

from aurora.model.attention import GroupedQueryAttention
from aurora.model.mlp import SwiGLU
from aurora.model.norms import RMSNorm
from aurora.model.recurrent import FixedLoopRecurrentTransformer, RecurrentLMOutput
from aurora.model.rope import RotaryEmbedding
from aurora.model.transformer import DenseTransformer, LMOutput

__all__ = [
    "DenseTransformer",
    "FixedLoopRecurrentTransformer",
    "GroupedQueryAttention",
    "LMOutput",
    "RMSNorm",
    "RecurrentLMOutput",
    "RotaryEmbedding",
    "SwiGLU",
]
