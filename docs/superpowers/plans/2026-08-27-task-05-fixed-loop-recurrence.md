# Task 05 Fixed-Loop Recurrence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and validate a fixed-loop prelude/shared-core/coda language model while preserving the Tasks 01–04 dense B0 control unchanged.

**Architecture:** Add separate recurrent configuration and model modules. Reuse the existing decoder block inside fixed prelude/coda stages and one repeatedly invoked core object, then extend the single-device trainer only for recurrent metrics and run controlled B0/R1/R2/R4/R8 experiments.

**Tech Stack:** Python 3.10+, PyTorch 2.5+, Hugging Face `tokenizers`, PyYAML, NumPy, pytest, Ruff, and mypy.

**Spec:** `docs/superpowers/specs/2026-08-27-task-05-fixed-loop-recurrence-design.md`

## Global Constraints

- `DenseTransformer`, `ModelConfig`, and existing B0 configs retain their architecture and behavior.
- Fixed `num_iterations` values are exactly 1, 2, 4, or 8; `max_iterations` is 8 in Task 05 configs.
- One shared core object and parameter set is invoked at every iteration.
- Additive learned step embeddings are allowed; gates, anchors, scratch state, memory, adaptive halting, EAC, routing, randomized depth, and stability regularizers are forbidden.
- Forward diagnostics are observational and must not change logits or loss.
- All scientific records contain measured artifact values only.

---

### Task 1: Separate recurrent configuration

**Files:**
- Create: `src/aurora/recurrent_config.py`
- Create: `tests/unit/test_recurrent_config.py`

**Interfaces:**
- Produces: `RecurrentModelConfig`
- Produces: `RecurrentExperimentConfig.from_yaml(path: Path) -> RecurrentExperimentConfig`
- Produces: canonical `to_dict()`, `to_yaml()`, `sha256()`, and strict `validate()` behavior.

- [ ] Write tests that accept only fixed depths 1/2/4/8, reject unknown keys and invalid stage/max-depth settings, and prove config round-trip/hash stability.
- [ ] Run `python -m pytest tests/unit/test_recurrent_config.py -q` and confirm import failure because `aurora.recurrent_config` is absent.
- [ ] Implement frozen recurrent schemas without changing `src/aurora/config.py`.
- [ ] Run the focused tests, Ruff, and mypy.
- [ ] Commit as `feat: add fixed-loop recurrent configuration`.

### Task 2: Shared recurrent model and architectural invariants

**Files:**
- Create: `src/aurora/model/recurrent.py`
- Modify: `src/aurora/model/__init__.py`
- Create: `tests/unit/test_recurrent_model.py`
- Create: `tests/numerical/test_recurrent_equivalence.py`

**Interfaces:**
- Produces: `TransformerStage(config, num_layers)`.
- Produces: `RecurrentCore(config)` with `forward(hidden, iteration=0)`.
- Produces: `FixedLoopRecurrentTransformer(config)` returning `RecurrentLMOutput`.

- [ ] Write a failing one-loop test comparing model forward with the explicit embedding → prelude → core → coda → final norm/head path and hand-computed aligned cross-entropy using exact tensor equality.
- [ ] Write failing tests for hook-observed core identity, unchanged parameter IDs/storage pointers, unique `core.*` state keys, and identical R=1/2/4/8 parameter counts/key sets.
- [ ] Run the focused tests and confirm missing recurrent model imports.
- [ ] Implement the minimal stages, one core object, fixed Python loop, loss, and generation.
- [ ] Run focused tests and the inherited dense-model tests.
- [ ] Commit as `feat: add shared fixed-loop recurrent model`.

### Task 3: Gradient-through-time, determinism, and observational diagnostics

**Files:**
- Modify: `src/aurora/model/recurrent.py`
- Create: `src/aurora/training/recurrent_metrics.py`
- Create: `tests/numerical/test_recurrent_gradients.py`
- Create: `tests/numerical/test_recurrent_determinism.py`
- Create: `tests/unit/test_recurrent_metrics.py`

**Interfaces:**
- Produces: per-iteration `RecurrentStateDiagnostics`.
- Produces: optional retained recurrent states in `RecurrentLMOutput`.
- Produces: `collect_core_gradient_statistics(core: nn.Module) -> CoreGradientStatistics`.

- [ ] Write failing tests requiring finite nonzero prelude/core/coda gradients for R=1 and finite nonzero retained-state gradients for every state at R=4.
- [ ] Write a failing deterministic replay test for states, logits, loss, and diagnostics.
- [ ] Write a failing diagnostic noninterference test requiring exact logits/loss equality with collection disabled/enabled.
- [ ] Write failing literal-fixture tests for state and core-gradient statistic formulas.
- [ ] Implement detached state diagnostics, retained-state output, and pre-clip core-gradient aggregation.
- [ ] Run focused numerical tests and commit as `feat: add recurrent numerical diagnostics`.

### Task 4: Recurrent FLOPs and trainer integration

**Files:**
- Modify: `src/aurora/training/trainer.py`
- Modify: `src/aurora/training/metrics.py`
- Create: `src/aurora/training/recurrent_trainer.py`
- Modify: `src/aurora/training/__init__.py`
- Create: `tests/unit/test_recurrent_trainer.py`

**Interfaces:**
- Produces: `estimate_recurrent_transformer_flops(config, batch_size, sequence_length, training)`.
- Produces: `RecurrentTrainer` and `RecurrentTrainingResult`.

- [ ] Write a failing hand-derived recurrent FLOP test proving active block count is `prelude + coda + R*core`.
- [ ] Write a failing real-trainer test proving optimizer parameter IDs are unique and exactly equal the model's unique trainable parameter IDs.
- [ ] Write a failing trainer test proving recurrent diagnostics and gradient statistics are appended to `metrics.jsonl`.
- [ ] Generalize only trainer type boundaries needed to accept the recurrent model/config; do not alter the dense training path.
- [ ] Implement the recurrent training override and run focused plus dense trainer/resume tests.
- [ ] Commit as `feat: train and measure fixed-loop recurrence`.

### Task 5: Recurrent CLI, configs, and integration

**Files:**
- Create: `src/aurora/cli/recurrent_tiny.py`
- Modify: `pyproject.toml`
- Create: `configs/task05/b0_dense.yaml`
- Create: `configs/task05/recurrent_1.yaml`
- Create: `configs/task05/recurrent_2.yaml`
- Create: `configs/task05/recurrent_4.yaml`
- Create: `configs/task05/recurrent_8.yaml`
- Create: `tests/integration/test_recurrent_pipeline.py`

**Interfaces:**
- Produces: `aurora-recurrent --config PATH --runs-dir PATH`.
- Produces: `run_recurrent_experiment(config_path, runs_dir) -> dict[str, object]`.

- [ ] Write a failing offline byte-tokenizer integration test covering train, evaluation, diagnostics, checkpoint reload, and result artifacts.
- [ ] Run it and confirm failure because the recurrent pipeline is absent.
- [ ] Implement the production-BPE recurrent CLI and measured result artifact.
- [ ] Add five matched-corpus fixed-depth configs and the console entry point.
- [ ] Run the integration test and full static/test gate.
- [ ] Commit as `feat: add Task 05 controlled experiment pipeline`.

### Task 6: Controlled runs and research records

**Files:**
- Modify: `docs/DECISIONS.md`
- Modify: `docs/EXPERIMENTS.md`
- Modify: `docs/PROGRESS.md`

**Interfaces:**
- Consumes: B0 `aurora-tiny` and recurrent `aurora-recurrent` commands.
- Produces: immutable ignored run artifacts and measured Task 05 records.

- [ ] Run B0 and R=1/2/4/8 on CPU fp32 from a clean committed source state.
- [ ] Validate each result/config/metadata/metrics artifact and confirm finite values, exact checkpoint reload, and recorded recurrent diagnostics.
- [ ] Compute only transparent descriptive comparisons: active FLOPs/token, total training FLOPs, throughput, and loss beside compute; make no raw-loss superiority claim.
- [ ] Update the three research records with exact artifact values and explicit limitations.
- [ ] Run Ruff format/check, mypy, full pytest, diff checks, artifact-exclusion checks, and a tracked-secret scan.
- [ ] Commit as `docs: record Task 05 fixed-loop results` and stop without Task 06 work.
