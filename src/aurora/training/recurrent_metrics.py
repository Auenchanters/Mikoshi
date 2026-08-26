from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class CoreGradientStatistics:
    l2_norm: float
    mean_abs: float
    max_abs: float
    nonzero_fraction: float
    finite: bool
    parameters_with_grad: int
    trainable_parameters: int


def collect_core_gradient_statistics(core: nn.Module) -> CoreGradientStatistics:
    squared_sum = 0.0
    absolute_sum = 0.0
    maximum = 0.0
    nonzero = 0
    parameters_with_grad = 0
    trainable_parameters = 0
    finite = True
    for parameter in core.parameters():
        if not parameter.requires_grad:
            continue
        trainable_parameters += parameter.numel()
        gradient = parameter.grad
        if gradient is None:
            finite = False
            continue
        detached = gradient.detach().float()
        parameters_with_grad += detached.numel()
        if not bool(torch.isfinite(detached).all()):
            finite = False
        squared_sum += float(detached.square().sum().item())
        absolute_sum += float(detached.abs().sum().item())
        maximum = max(maximum, float(detached.abs().max().item()))
        nonzero += int(torch.count_nonzero(detached).item())
    if trainable_parameters == 0:
        raise ValueError("core has no trainable parameters")
    mean_abs = absolute_sum / trainable_parameters
    nonzero_fraction = nonzero / trainable_parameters
    return CoreGradientStatistics(
        l2_norm=math.sqrt(squared_sum),
        mean_abs=mean_abs,
        max_abs=maximum,
        nonzero_fraction=nonzero_fraction,
        finite=finite and parameters_with_grad == trainable_parameters,
        parameters_with_grad=parameters_with_grad,
        trainable_parameters=trainable_parameters,
    )
