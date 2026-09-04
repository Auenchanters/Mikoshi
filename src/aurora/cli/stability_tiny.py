from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch

from aurora.cli.recurrent_tiny import _dataset, _dataset_hash, _tiny_corpus
from aurora.experiment import RunArtifacts, create_run
from aurora.model.recurrent_stability import ControlledRecurrentTransformer
from aurora.stability_config import StabilityExperimentConfig
from aurora.tokenization.base import TokenizerProtocol
from aurora.tokenization.bpe import train_bpe
from aurora.training.metrics import count_parameters
from aurora.training.reproducibility import seed_everything
from aurora.training.stability_metrics import estimate_stability_transformer_flops
from aurora.training.stability_trainer import StabilityTrainer


def _models_are_equal(
    left: ControlledRecurrentTransformer,
    right: ControlledRecurrentTransformer,
) -> bool:
    left_state = left.state_dict()
    right_state = right.state_dict()
    return set(left_state) == set(right_state) and all(
        torch.equal(left_state[name], right_state[name]) for name in left_state
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_stability_pipeline(
    *,
    config: StabilityExperimentConfig,
    tokenizer: TokenizerProtocol,
    train_text: str,
    eval_text: str,
    run: RunArtifacts,
    generation_prompt: str,
    generation_tokens: int,
) -> dict[str, object]:
    seed_everything(config.runtime.seed, config.runtime.deterministic)
    train_dataset = _dataset(tokenizer, train_text, config)
    eval_dataset = _dataset(tokenizer, eval_text, config)
    model = ControlledRecurrentTransformer(config.model)
    trainer = StabilityTrainer(
        config=config,
        model=model,
        tokenizer_identity=tokenizer.identity,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        run=run,
    )
    training = trainer.train()
    if not training.final_loss < training.initial_loss:
        raise RuntimeError(
            "tiny stability invariant failed: final loss did not decrease below initial loss"
        )
    evaluation = trainer.evaluate(max_batches=min(8, len(eval_dataset)))
    checkpoint_path = run.checkpoints_dir / "final.pt"
    trainer.save(checkpoint_path)

    seed_everything(config.runtime.seed + 1, config.runtime.deterministic)
    restored_model = ControlledRecurrentTransformer(config.model)
    restored = StabilityTrainer(
        config=config,
        model=restored_model,
        tokenizer_identity=tokenizer.identity,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        run=run,
    )
    restored.resume(checkpoint_path)
    restored_evaluation = restored.evaluate(max_batches=min(8, len(eval_dataset)))
    replay_verified = _models_are_equal(trainer.model, restored.model) and (
        restored_evaluation.loss == evaluation.loss
    )
    if not replay_verified:
        raise RuntimeError("fresh stability checkpoint reload did not reproduce evaluation state")

    prompt_ids = tokenizer.encode(generation_prompt, add_bos=True)
    prompt = torch.tensor([prompt_ids], dtype=torch.long, device=restored.device)
    generated_ids = restored.model.generate(prompt, max_new_tokens=generation_tokens)
    generated_text = tokenizer.decode(generated_ids[0].tolist())
    if not generated_text:
        raise RuntimeError("generation produced empty decoded text")

    counts = count_parameters(restored.model)
    training_flops_per_step = estimate_stability_transformer_flops(
        config.model,
        batch_size=config.data.batch_size,
        sequence_length=config.data.sequence_length,
        training=True,
    )
    forward_flops_per_step = estimate_stability_transformer_flops(
        config.model,
        batch_size=config.data.batch_size,
        sequence_length=config.data.sequence_length,
        training=False,
    )
    tokens_per_step = config.data.batch_size * config.data.sequence_length
    peak_vram = (
        int(torch.cuda.max_memory_allocated(restored.device))
        if restored.device.type == "cuda"
        else None
    )
    final_core_gradient = training.core_gradient_statistics[-1]
    final_gate_diagnostics = [asdict(diagnostic) for diagnostic in training.last_gate_diagnostics]
    gate_collapse = (
        training.last_gate_diagnostics[-1].collapse
        if training.last_gate_diagnostics
        else "not_applicable"
    )
    result: dict[str, object] = {
        "active_flops_per_token": forward_flops_per_step / tokens_per_step,
        "active_parameters": counts.active,
        "checkpoint_reloaded": restored.step == training.steps_completed,
        "checkpoint_sha256": _sha256(checkpoint_path),
        "config_hash": config.sha256(),
        "dataset_sha256": _dataset_hash(train_text, eval_text),
        "deterministic_replay_verified": replay_verified,
        "estimated_training_flops": training_flops_per_step * training.steps_completed,
        "evaluation_loss": evaluation.loss,
        "final_core_gradient_statistics": asdict(final_core_gradient),
        "final_gate_diagnostics": final_gate_diagnostics,
        "final_gradient_norm": training.gradient_norms[-1],
        "final_loss": training.final_loss,
        "final_recurrent_diagnostics": [
            asdict(diagnostic) for diagnostic in training.last_iteration_diagnostics
        ],
        "gate_collapse_status": gate_collapse,
        "gated_update": config.model.gated_update,
        "generated_text": generated_text,
        "initial_loss": training.initial_loss,
        "input_anchoring": config.model.input_anchoring,
        "loss_decreased": training.final_loss < training.initial_loss,
        "mean_gradient_norm": sum(training.gradient_norms) / len(training.gradient_norms),
        "num_iterations": config.model.num_iterations,
        "peak_vram_bytes": peak_vram,
        "run_id": config.experiment.run_id,
        "state_stabilization": config.model.state_stabilization,
        "tokenizer_sha256": tokenizer.identity.sha256,
        "tokens_per_second": training.tokens_seen / training.wall_time_seconds,
        "total_parameters": counts.total,
        "trainable_parameters": counts.trainable,
        "training_tokens": training.tokens_seen,
        "variant": config.model.variant,
        "wall_time_seconds": training.wall_time_seconds,
    }
    run.result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    metadata = json.loads(run.metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "active_flops_per_token": result["active_flops_per_token"],
            "checkpoint_sha256": result["checkpoint_sha256"],
            "estimated_training_flops": result["estimated_training_flops"],
            "num_iterations": config.model.num_iterations,
            "peak_vram_bytes": peak_vram,
            "status": "completed",
            "training_tokens": training.tokens_seen,
            "variant": config.model.variant,
            "wall_time_seconds": training.wall_time_seconds,
        }
    )
    run.metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return result


def run_stability_experiment(config_path: Path, runs_dir: Path) -> dict[str, object]:
    config = StabilityExperimentConfig.from_yaml(config_path)
    run = create_run(config, runs_dir)
    train_text, eval_text = _tiny_corpus()
    corpus_path = run.root / "training_corpus.txt"
    corpus_path.write_text(train_text, encoding="utf-8", newline="\n")
    tokenizer = train_bpe([corpus_path], run.tokenizer_dir, config.tokenizer)
    if tokenizer.vocab_size != config.model.vocab_size:
        raise RuntimeError(
            f"trained tokenizer vocabulary {tokenizer.vocab_size} does not match "
            f"model vocabulary {config.model.vocab_size}"
        )
    return run_stability_pipeline(
        config=config,
        tokenizer=tokenizer,
        train_text=train_text,
        eval_text=eval_text,
        run=run,
        generation_prompt="AURORA learns",
        generation_tokens=12,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Task 06 recurrent stability experiment")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    arguments = parser.parse_args()
    result = run_stability_experiment(arguments.config, arguments.runs_dir)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
