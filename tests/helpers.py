from __future__ import annotations

from pathlib import Path

from conftest import tiny_config_dict

from aurora.config import ExperimentConfig
from aurora.data.dataset import TokenBlockDataset
from aurora.experiment import create_run
from aurora.model.transformer import DenseTransformer
from aurora.training.reproducibility import seed_everything
from aurora.training.trainer import Trainer
from tests.fixtures.byte_tokenizer import ByteTokenizer


def make_tiny_trainer(
    root: Path,
    *,
    initialization_seed: int = 17,
    total_steps: int = 30,
) -> Trainer:
    payload = tiny_config_dict(
        data={"sequence_length": 8, "batch_size": 4, "stride": 4},
        model={
            "max_seq_len": 8,
            "d_model": 24,
            "num_layers": 1,
            "num_heads": 4,
            "num_kv_heads": 2,
            "d_ff": 48,
        },
        optimizer={"learning_rate": 0.01, "weight_decay": 0.0},
        scheduler={"warmup_steps": 0, "total_steps": total_steps, "min_lr_ratio": 0.2},
        checkpoint={"save_every": total_steps, "keep_last": 1},
    )
    config = ExperimentConfig.from_dict(payload)
    seed_everything(initialization_seed, deterministic=True)
    model = DenseTransformer(config.model)
    pattern = [4, 5, 6, 7, 8, 7, 6, 5]
    tokens = pattern * 80
    dataset = TokenBlockDataset(
        tokens,
        sequence_length=config.data.sequence_length,
        stride=config.data.stride,
    )
    run = create_run(config, root / "runs")
    return Trainer(
        config=config,
        model=model,
        tokenizer_identity=ByteTokenizer().identity,
        train_dataset=dataset,
        eval_dataset=dataset,
        run=run,
    )
