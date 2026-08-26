# Progress

## Current phase

Tasks 01–04 from `plan.md` are implemented and verified on the feature branch.

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

## Verification snapshot

- Ruff formatter: clean
- Ruff linter: clean
- mypy: clean across `src`
- pytest: 41 passed
- `EXP-0001-dense-sanity`: loss 5.7461218834 → 3.8214008808
- fresh checkpoint reload and exact replay: verified
- unexplained NaN/Inf: none observed

## Deliberately deferred

- recurrent prelude/core/coda architecture;
- recurrence instrumentation and stability experiments;
- scratch/state highways;
- persistent neural memory;
- Experience-Adaptive Compute and learned halting;
- distributed training;
- mixed-precision CUDA validation;
- Triton/CUDA kernels;
- large training runs.

## Next phase: Task 05

Task 05 will add a recurrent prelude/core/coda model with externally fixed loop
counts, while preserving this dense model unchanged as the control. It will not
add neural memory, EAC, or learned halting. The first comparisons must be
parameter- and FLOP-aware and must retain the existing checkpoint/test
contracts.
