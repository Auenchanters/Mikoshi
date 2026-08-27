from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import cast

import torch
from torch import Tensor

from aurora.cli.recurrent_tiny import _dataset, _tiny_corpus
from aurora.model.recurrent_stability import ControlledRecurrentTransformer
from aurora.stability_config import StabilityExperimentConfig
from aurora.tokenization.base import TokenizerProtocol
from aurora.tokenization.bpe import BPETokenizer
from aurora.training.checkpoint import load_checkpoint
from aurora.training.metrics import count_parameters
from aurora.training.stability_metrics import estimate_stability_transformer_flops

_STATE_FIELDS = (
    "hidden_state_rms",
    "recurrent_update_rms",
    "state_cosine_similarity",
    "initial_state_cosine_similarity",
    "relative_update_magnitude",
)
_GATE_FIELDS = ("mean", "population_std", "minimum", "maximum")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _device(config: StabilityExperimentConfig) -> torch.device:
    requested = config.runtime.device
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    device = torch.device(requested)
    if device.type == "cpu" and config.runtime.precision != "fp32":
        raise ValueError("CPU evaluation supports fp32 only")
    return device


def _autocast(config: StabilityExperimentConfig, device: torch.device) -> torch.autocast:
    enabled = device.type == "cuda" and config.runtime.precision != "fp32"
    dtype = torch.bfloat16 if config.runtime.precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def _mean_diagnostics(
    sums: list[dict[str, float]], total_tokens: int, fields: Sequence[str]
) -> list[dict[str, object]]:
    averaged: list[dict[str, object]] = []
    for iteration, values in enumerate(sums, start=1):
        item: dict[str, object] = {"iteration": iteration}
        for name in fields:
            item[name] = values[name] / total_tokens
        if fields == _GATE_FIELDS:
            mean = cast(float, item["mean"])
            item["collapse"] = "zero" if mean <= 0.05 else "one" if mean >= 0.95 else "none"
        averaged.append(item)
    return averaged


@torch.no_grad()
def evaluate_cross_depth(
    *,
    config: StabilityExperimentConfig,
    tokenizer: TokenizerProtocol,
    eval_text: str,
    checkpoint_path: Path,
    output_path: Path | None = None,
    depths: Sequence[int] = (1, 2, 3, 4, 6, 8),
    max_batches: int | None = 8,
) -> dict[str, object]:
    if not depths:
        raise ValueError("at least one inference depth is required")
    if len(set(depths)) != len(depths):
        raise ValueError("inference depths must be unique")
    if any(
        type(depth) is not int or not 1 <= depth <= config.model.max_iterations for depth in depths
    ):
        raise ValueError("inference depths must be integers within [1, max_iterations]")
    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be positive")

    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    if checkpoint["config_hash"] != config.sha256():
        raise ValueError("checkpoint configuration does not match evaluation configuration")
    checkpoint_tokenizer = cast(Mapping[str, object], checkpoint["tokenizer"])
    if checkpoint_tokenizer.get("sha256") != tokenizer.identity.sha256:
        raise ValueError("checkpoint tokenizer does not match evaluation tokenizer")
    experiment = cast(Mapping[str, object], checkpoint["experiment"])
    if experiment.get("run_id") != config.experiment.run_id:
        raise ValueError("checkpoint run identity does not match evaluation configuration")

    device = _device(config)
    model = ControlledRecurrentTransformer(config.model).to(device)
    model.load_state_dict(cast(Mapping[str, Tensor], checkpoint["model"]))
    model.eval()
    dataset = _dataset(tokenizer, eval_text, config)
    batch_indices = [
        list(range(start, min(start + config.data.batch_size, len(dataset))))
        for start in range(0, len(dataset), config.data.batch_size)
    ]
    if max_batches is not None:
        batch_indices = batch_indices[:max_batches]
    if not batch_indices:
        raise ValueError("evaluation dataset produced no batches")

    results: list[dict[str, object]] = []
    tokens_per_forward = config.data.batch_size * config.data.sequence_length
    for depth in depths:
        total_loss = 0.0
        total_tokens = 0
        state_sums = [{name: 0.0 for name in _STATE_FIELDS} for _ in range(depth)]
        gate_sums = (
            [{name: 0.0 for name in _GATE_FIELDS} for _ in range(depth)]
            if config.model.gated_update
            else []
        )
        for indices in batch_indices:
            examples = [dataset[index] for index in indices]
            input_ids = torch.stack([example[0] for example in examples]).to(device)
            targets = torch.stack([example[1] for example in examples]).to(device)
            with _autocast(config, device):
                output = model(
                    input_ids,
                    targets=targets,
                    collect_diagnostics=True,
                    num_iterations_override=depth,
                )
            if output.loss is None or not bool(torch.isfinite(output.loss).item()):
                raise FloatingPointError("non-finite cross-depth evaluation loss")
            token_count = int(targets.numel())
            total_loss += float(output.loss.item()) * token_count
            total_tokens += token_count
            if len(output.iteration_diagnostics) != depth:
                raise RuntimeError("model returned the wrong number of state diagnostics")
            for index, diagnostic in enumerate(output.iteration_diagnostics):
                for name in _STATE_FIELDS:
                    value = float(getattr(diagnostic, name))
                    if not math.isfinite(value):
                        raise FloatingPointError("non-finite cross-depth state diagnostics")
                    state_sums[index][name] += value * token_count
            if config.model.gated_update:
                if len(output.gate_diagnostics) != depth:
                    raise RuntimeError("model returned the wrong number of gate diagnostics")
                for index, diagnostic in enumerate(output.gate_diagnostics):
                    for name in _GATE_FIELDS:
                        value = float(getattr(diagnostic, name))
                        if not math.isfinite(value):
                            raise FloatingPointError("non-finite cross-depth gate diagnostics")
                        gate_sums[index][name] += value * token_count
        evaluation_loss = total_loss / total_tokens
        if not math.isfinite(evaluation_loss):
            raise FloatingPointError("non-finite aggregate cross-depth evaluation loss")
        results.append(
            {
                "active_flops_per_token": estimate_stability_transformer_flops(
                    replace(config.model, num_iterations=depth),
                    batch_size=config.data.batch_size,
                    sequence_length=config.data.sequence_length,
                    training=False,
                )
                / tokens_per_forward,
                "batches": len(batch_indices),
                "evaluation_loss": evaluation_loss,
                "gate_diagnostics": _mean_diagnostics(gate_sums, total_tokens, _GATE_FIELDS),
                "inference_depth": depth,
                "recurrent_diagnostics": _mean_diagnostics(state_sums, total_tokens, _STATE_FIELDS),
                "target_tokens": total_tokens,
            }
        )

    counts = count_parameters(model)
    result: dict[str, object] = {
        "active_parameters": counts.active,
        "configured_training_depth": config.model.num_iterations,
        "evaluated_depths": list(depths),
        "results": results,
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": _sha256(checkpoint_path),
        "source_config_hash": config.sha256(),
        "source_run_id": config.experiment.run_id,
        "tokenizer_sha256": tokenizer.identity.sha256,
        "total_parameters": counts.total,
        "trainable_parameters": counts.trainable,
        "variant": config.model.variant,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one Task 06 checkpoint by depth")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8])
    parser.add_argument("--max-batches", type=int, default=8)
    arguments = parser.parse_args()
    config = StabilityExperimentConfig.from_yaml(arguments.config)
    tokenizer = BPETokenizer.load(arguments.tokenizer_dir)
    _, eval_text = _tiny_corpus()
    result = evaluate_cross_depth(
        config=config,
        tokenizer=tokenizer,
        eval_text=eval_text,
        checkpoint_path=arguments.checkpoint,
        output_path=arguments.output,
        depths=arguments.depths,
        max_batches=arguments.max_batches,
    )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
