# Progress

## Current phase

Tasks 01–05 from `plan.md` are implemented and verified on the Task 05 feature
branch.

### Task 01 — Repository skeleton and configuration

- installable `src`-layout package and `pyproject.toml`;
- strict nested YAML dataclasses with unknown-key rejection;
- canonical config serialization and SHA-256 identity;
- immutable `EXP-NNNN-slug` run directories;
- generated data, runs, checkpoints, environments, and secrets ignored by Git.

### Task 02 — Tokenizer and deterministic data

- production Hugging Face BPE training, save/load, settings metadata, ordered
  corpus hash, and artifact hash;
- BPE determinism and UTF-8 round-trip tests;
- fixed byte tokenizer confined to `tests/fixtures`;
- shifted token-block dataset;
- seeded epoch permutations with serialized epoch/order/cursor and exact resume.

### Task 03 — Modern dense Transformer control

- RMSNorm, interleaved RoPE, grouped-query causal attention, QK normalization,
  SwiGLU, pre-normalized residuals, tied embeddings, aligned cross-entropy, and
  deterministic generation;
- unit/numerical tests for equations, shapes, causal masking, finite gradients,
  loss, and generation.

### Task 04 — Train/eval/checkpoint infrastructure

- single-device AdamW trainer with warmup/cosine schedule, gradient clipping,
  finite-value aborts, throughput/FLOP/VRAM metrics, token-weighted evaluation,
  and generation;
- atomic checkpoints containing model, optimizer, scheduler, optional scaler,
  RNG, sampler cursor, counters, config, tokenizer, and experiment identity;
- exact uninterrupted-versus-resumed CPU fp32 replay;
- complete offline integration and production-BPE tiny experiment.

### Task 05 — Fixed-loop recurrent prelude/core/coda

- separate recurrent configuration path with fixed counts 1, 2, 4, and 8;
- one fixed prelude stage, one repeatedly invoked shared core object, one fixed
  coda stage, and fixed-size recurrent-step embeddings;
- exact R=1 expanded-path logits/loss and finite nonzero prelude/core/coda
  gradients on deterministic CPU fp32;
- hook-observed core object identity, stable parameter IDs/storage pointers,
  unique optimizer membership, and R=1/2/4/8 parameter/state-key invariance;
- finite nonzero gradients through every retained state in an R=4 trajectory;
- bitwise deterministic recurrent states, logits, and loss on repeated CPU fp32
  forwards;
- output-neutral per-iteration hidden/update norms, successive-state cosine,
  cosine to the initial recurrent state, relative-update magnitude, and
  pre-clip core-gradient statistics;
- production-BPE B0/R1/R2/R4/R8 controlled sweep with exact checkpoint reload.
- negative experimental result retained: deeper naïve R=2/4/8 recurrence used
  more active compute and degraded evaluation loss monotonically versus R=1.

## Verification snapshot

- Ruff formatter: clean
- Ruff linter: clean
- mypy: clean across `src`
- pytest: 66 passed
- `EXP-0001-dense-sanity`: loss 5.7461218834 → 3.8214008808
- Task 05 B0: loss 5.7461218834 → 3.8214008808, eval 3.7949758768
- Task 05 R=1/2/4/8: all training losses decreased; evaluation losses were
  3.7306981683 / 3.8169981837 / 4.0504400730 / 4.1997586489
- R=1/2/4/8: identical 91,152 parameter counts and state-dict key sets
- fresh checkpoint reload and exact replay: verified
- unexplained NaN/Inf: none observed

## Deliberately deferred

- broader Task 06 recurrence analysis/instrumentation beyond the recorded Task
  05 norm/update/cosine-to-previous/cosine-to-h0/relative-update and
  core-gradient observations;
- recurrence-stability experiments and regularization;
- scratch/state highways;
- persistent neural memory;
- Experience-Adaptive Compute and learned halting;
- distributed training;
- mixed-precision CUDA validation;
- Triton/CUDA kernels;
- large training runs.

## Stopping boundary

Task 05 is complete. Task 06 has not started. No adaptive halting, neural
memory, EAC, routing, STARS/JSRR, randomized depth, scratch/state highway, or
other later-plan mechanism was implemented.
