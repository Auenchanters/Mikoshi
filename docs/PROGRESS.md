# Progress

## Current phase

Tasks 01–06 are implemented and verified through the Task 06 recurrent-stability
branch. Research Task 07 has not started.

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

### Task 06 — Recurrent-stability controls

- exact delegated S0 compatibility with the frozen Task 05 model and results;
- independently configurable S1 projected additive input anchor, S2 scalar
  gated interpolation, S3 parameter-free initial-RMS stabilization, and S4
  anchor-plus-gate composition;
- immutable `h0` anchor sourcing, one shared core object, parameter/state-key
  invariance across depth, unique optimizer membership, retained-state and
  mechanism gradients, deterministic CPU fp32 execution, and output-neutral
  state/gate diagnostics;
- predeclared gate-collapse thresholds of mean at most 0.05 or at least 0.95;
- mechanism-aware matmul-only parameter/FLOP accounting and Task 06-specific
  trainer, production pipeline, checkpoint replay, and cross-depth evaluator;
- eight immutable production-BPE R=4/R=8 runs under the frozen Task 05 tiny
  protocol; every run completed with finite metrics, 40 metric rows, exact
  checkpoint reload, deterministic replay, and matching config/data/tokenizer/
  checkpoint identity;
- complete S0–S4 R=4/R=8 result matrix: S1/S2/S4 reduced final directional
  drift at both depths, every Task 06 mechanism reduced final RMS, S3 held RMS
  essentially constant, and no S2/S4 gate collapsed;
- negative/mixed result retained: no mechanism prevented near-collinear
  successive states; S3 had worse raw loss than S0 at both depths; raw loss
  differences from unmatched parameter/compute conditions are not improvement
  claims;
- predeclared R=4 selection applied mechanically against frozen S0 RMS
  0.3238864541 and cosine to `h0` 0.6662002206: S1/S2/S4 qualified, S3 did not,
  and lowest-loss qualifying S4/4 was selected;
- exact unchanged S4/4 checkpoint evaluated at inference R=1/2/3/4/6/8. Loss
  reached its minimum at trained depth 4, then rose at R=6/R=8 while remaining
  finite with no gate collapse; checkpoint/config bytes and configured training
  depth remained unchanged.

## Verification snapshot

- Ruff formatter: clean
- Ruff linter: clean
- mypy: clean across `src`
- pytest: 201 passed
- `EXP-0001-dense-sanity`: loss 5.7461218834 → 3.8214008808
- Task 05 B0: loss 5.7461218834 → 3.8214008808, eval 3.7949758768
- Task 05 R=1/2/4/8: all training losses decreased; evaluation losses were
  3.7306981683 / 3.8169981837 / 4.0504400730 / 4.1997586489
- R=1/2/4/8: identical 91,152 parameter counts and state-dict key sets
- fresh checkpoint reload and exact replay: verified
- Task 06 S1/S2/S3/S4 R=4 evaluation losses: 3.8888302743 / 3.8293781579 /
  4.2306061983 / 3.7531955242
- Task 06 S1/S2/S3/S4 R=8 evaluation losses: 3.9617625475 / 4.0581546426 /
  4.2106192112 / 3.9005704820
- selected S4/4 cross-depth R=1/2/3/4/6/8 evaluation losses:
  3.8521460295 / 3.7836607397 / 3.7593515813 / 3.7531955242 /
  3.7674068511 / 3.7994184196
- unexplained NaN/Inf: none observed

## Deliberately deferred

- scratch/state highways;
- persistent neural memory;
- Experience-Adaptive Compute and learned halting;
- randomized recurrent depth and routing;
- Jacobian/STARS stability regularization;
- distributed training;
- mixed-precision CUDA validation;
- Triton/CUDA kernels;
- large training runs.

## Stopping boundary

Task 06 is complete. Research Task 07 has not started. No adaptive halting,
neural memory, EAC, routing, STARS/JSRR, randomized depth, scratch/state
highway, or other Task 07+ mechanism was implemented. The Task 06 results do
not establish a controlled model-quality improvement or that AURORA is
successful.
