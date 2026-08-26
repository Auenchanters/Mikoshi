from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch

from aurora.data.dataset import TokenBlockDataset
from aurora.experiment import RunArtifacts, create_run
from aurora.model.recurrent import FixedLoopRecurrentTransformer
from aurora.recurrent_config import RecurrentExperimentConfig
from aurora.tokenization.base import TokenizerProtocol
from aurora.tokenization.bpe import train_bpe
from aurora.training.metrics import (
    count_parameters,
    estimate_recurrent_transformer_flops,
)
from aurora.training.recurrent_trainer import RecurrentTrainer
from aurora.training.reproducibility import seed_everything


def _dataset_hash(train_text: str, eval_text: str) -> str:
    digest = hashlib.sha256(b"AURORA-TASK05-DATASET-v1\0")
    for text in (train_text, eval_text):
        encoded = text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big"))
        digest.update(encoded)
    return digest.hexdigest()


def _dataset(
    tokenizer: TokenizerProtocol,
    text: str,
    config: RecurrentExperimentConfig,
) -> TokenBlockDataset:
    tokens = tokenizer.encode(text, add_bos=True, add_eos=True)
    if not tokens:
        raise ValueError("tokenizer produced an empty token stream")
    if min(tokens) < 0 or max(tokens) >= config.model.vocab_size:
        raise ValueError("tokenizer emitted IDs outside model.vocab_size")
    return TokenBlockDataset(
        tokens,
        sequence_length=config.data.sequence_length,
        stride=config.data.stride,
    )


def _models_are_equal(
    left: FixedLoopRecurrentTransformer,
    right: FixedLoopRecurrentTransformer,
) -> bool:
    left_state = left.state_dict()
    right_state = right.state_dict()
    return set(left_state) == set(right_state) and all(
        torch.equal(left_state[name], right_state[name]) for name in left_state
    )


def run_recurrent_pipeline(
    *,
    config: RecurrentExperimentConfig,
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
    model = FixedLoopRecurrentTransformer(config.model)
    trainer = RecurrentTrainer(
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
            "tiny recurrent invariant failed: final loss did not decrease below initial loss"
        )
    evaluation = trainer.evaluate(max_batches=min(8, len(eval_dataset)))
    checkpoint_path = run.checkpoints_dir / "final.pt"
    trainer.save(checkpoint_path)

    seed_everything(config.runtime.seed + 1, config.runtime.deterministic)
    restored_model = FixedLoopRecurrentTransformer(config.model)
    restored = RecurrentTrainer(
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
        raise RuntimeError("fresh recurrent checkpoint reload did not reproduce evaluation state")

    prompt_ids = tokenizer.encode(generation_prompt, add_bos=True)
    prompt = torch.tensor([prompt_ids], dtype=torch.long, device=restored.device)
    generated_ids = restored.model.generate(prompt, max_new_tokens=generation_tokens)
    generated_text = tokenizer.decode(generated_ids[0].tolist())
    if not generated_text:
        raise RuntimeError("generation produced empty decoded text")

    counts = count_parameters(restored.model)
    training_flops_per_step = estimate_recurrent_transformer_flops(
        config.model,
        batch_size=config.data.batch_size,
        sequence_length=config.data.sequence_length,
        training=True,
    )
    forward_flops_per_step = estimate_recurrent_transformer_flops(
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
    result: dict[str, object] = {
        "active_flops_per_token": forward_flops_per_step / tokens_per_step,
        "active_parameters": counts.active,
        "checkpoint_reloaded": restored.step == training.steps_completed,
        "config_hash": config.sha256(),
        "dataset_sha256": _dataset_hash(train_text, eval_text),
        "deterministic_replay_verified": replay_verified,
        "estimated_training_flops": training_flops_per_step * training.steps_completed,
        "evaluation_loss": evaluation.loss,
        "final_core_gradient_statistics": asdict(final_core_gradient),
        "final_gradient_norm": training.gradient_norms[-1],
        "final_loss": training.final_loss,
        "final_recurrent_diagnostics": [
            asdict(diagnostic) for diagnostic in training.last_iteration_diagnostics
        ],
        "generated_text": generated_text,
        "initial_loss": training.initial_loss,
        "loss_decreased": training.final_loss < training.initial_loss,
        "mean_gradient_norm": sum(training.gradient_norms) / len(training.gradient_norms),
        "num_iterations": config.model.num_iterations,
        "peak_vram_bytes": peak_vram,
        "run_id": config.experiment.run_id,
        "tokenizer_sha256": tokenizer.identity.sha256,
        "tokens_per_second": training.tokens_seen / training.wall_time_seconds,
        "total_parameters": counts.total,
        "training_tokens": training.tokens_seen,
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
            "estimated_training_flops": result["estimated_training_flops"],
            "num_iterations": config.model.num_iterations,
            "peak_vram_bytes": peak_vram,
            "status": "completed",
            "training_tokens": training.tokens_seen,
            "wall_time_seconds": training.wall_time_seconds,
        }
    )
    run.metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return result


def _tiny_corpus() -> tuple[str, str]:
    lines = [
        "AURORA learns a small repeated language pattern from random initialization.",
        "Dense controls make later recurrent experiments scientifically interpretable.",
        "Every run records configuration, tokenizer, seed, loss, and checkpoint lineage.",
        "Alpha follows beta; beta follows gamma; gamma returns to alpha.",
        "Numbers 0 1 2 3 4 5 6 7 8 9 and punctuation remain reproducible.",
        "Unicode remains explicit: café, naïve, Δ, हिंदी, 日本語, and 🚀.",
    ]
    train = ("\n".join(lines) + "\n") * 128
    evaluation = ("\n".join(lines[:4]) + "\n") * 16
    return train, evaluation


def run_recurrent_experiment(config_path: Path, runs_dir: Path) -> dict[str, object]:
    config = RecurrentExperimentConfig.from_yaml(config_path)
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
    return run_recurrent_pipeline(
        config=config,
        tokenizer=tokenizer,
        train_text=train_text,
        eval_text=eval_text,
        run=run,
        generation_prompt="AURORA learns",
        generation_tokens=12,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Task 05 fixed-loop experiment")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/task05/recurrent_1.yaml"),
    )
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    arguments = parser.parse_args()
    result = run_recurrent_experiment(arguments.config, arguments.runs_dir)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
