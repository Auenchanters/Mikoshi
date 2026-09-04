# Task 06 Recurrent Stability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and evaluate S1 input anchoring, S2 gated recurrent updates,
S3 initial-RMS stabilization, and S4 anchor+gate against the frozen Task 05 S0
failure controls.

**Architecture:** New Task 06 classes subclass and reuse the frozen Task 05
prelude/shared-core/coda implementation. Mechanisms are conditional modules
around the shared core, with an exact delegated S0 path, separate configuration,
trainer/metrics, immutable run IDs, and an inference-only cross-depth evaluator.

**Tech Stack:** Python 3.10, PyTorch 2.5 CPU fp32, pytest, Ruff, mypy, Hugging
Face `tokenizers`, YAML experiment configurations.

**Spec:** `docs/superpowers/specs/2026-08-27-task-06-recurrent-stability-design.md`

## Global Constraints

- Do not modify `src/aurora/model/recurrent.py`, `src/aurora/recurrent_config.py`,
  `configs/task05/`, or existing Task 05 result text in the research records.
- S0 must delegate to the Task 05 forward path and match parameters, recurrent
  states, diagnostics, logits, and loss exactly under identical CPU fp32 inputs.
- Training recurrence remains fixed in `{1, 2, 4, 8}`; only cross-depth
  inference accepts `1..8`.
- S1/S2/S3 are independently configurable; S4 combines only S1 and S2.
- The shared core stays one module object and every optimizer parameter appears
  exactly once.
- Diagnostics are detached, deterministic, output-neutral, and finite for the
  controlled runs.
- Gate collapse thresholds are fixed at mean `<= 0.05` (toward zero) or
  `>= 0.95` (toward one).
- Active FLOPs use the Task 05 matmul-only convention plus explicit anchor and
  scalar-gate projection matmuls.
- Use the frozen Task 05 S0 R=4/R=8 records; never overwrite their artifacts.
- Run S4 only after independent S1/S2 tests and experiment artifacts exist.
- Do not implement any Task 07+ mechanism listed in the specification.

---

### Task 1: Stability configuration and exact S0 compatibility

**Files:**
- Create: `src/aurora/stability_config.py`
- Create: `src/aurora/model/recurrent_stability.py`
- Create: `tests/stability_helpers.py`
- Create: `tests/unit/test_stability_config.py`
- Create: `tests/numerical/test_stability_s0.py`

**Interfaces:**
- Consumes: `RecurrentModelConfig`, `RecurrentExperimentConfig`, and
  `FixedLoopRecurrentTransformer`.
- Produces: `StabilityModelConfig.as_recurrent_config()`,
  `StabilityModelConfig.variant`, `StabilityExperimentConfig`, and
  `ControlledRecurrentTransformer.forward(..., num_iterations_override=None)`.

- [ ] **Step 1: Write failing configuration tests**

  Add literal configurations for S0–S4 and reject every unrequested
  combination. The helper signature is:

  ```python
  def stability_model_config(
      *,
      num_iterations: int = 4,
      input_anchoring: bool = False,
      gated_update: bool = False,
      state_stabilization: str = "none",
  ) -> StabilityModelConfig: ...
  ```

  Assert variants resolve exactly to `"S0"` through `"S4"`, training depth 3
  is rejected, and an inference override of 3 remains a model-forward concern
  rather than a configuration value.

- [ ] **Step 2: Verify the configuration tests fail**

  Run:

  ```text
  python -m pytest tests/unit/test_stability_config.py -q
  ```

  Expected: import failure because `aurora.stability_config` does not exist.

- [ ] **Step 3: Implement the minimal stability configuration**

  Define:

  ```python
  @dataclass(frozen=True)
  class StabilityModelConfig(RecurrentModelConfig):
      input_anchoring: bool = False
      gated_update: bool = False
      state_stabilization: str = "none"

      def as_recurrent_config(self) -> RecurrentModelConfig: ...

      @property
      def variant(self) -> str: ...

  @dataclass(frozen=True)
  class StabilityExperimentConfig(RecurrentExperimentConfig):
      model: StabilityModelConfig
  ```

  `from_dict` must parse `StabilityModelConfig`, retain strict unknown-key and
  config-hash validation, call the inherited Task 05 validation through an
  equivalent base configuration, type-check the two booleans, and accept only
  the five combinations in the specification.

- [ ] **Step 4: Write the failing S0 identity test**

  Seed once, instantiate the Task 05 and controlled S0 models without reseeding,
  load the base state into S0, and assert identical state-dict keys/tensors,
  retained states, diagnostics, logits, and loss. Also assert S0 adds no
  trainable parameters.

- [ ] **Step 5: Verify the S0 test fails**

  Run:

  ```text
  python -m pytest tests/numerical/test_stability_s0.py -q
  ```

  Expected: import failure because the controlled model does not exist.

- [ ] **Step 6: Implement the delegated S0 model shell**

  Subclass `FixedLoopRecurrentTransformer`, initialize it from
  `config.as_recurrent_config()`, restore `self.config = config`, and delegate
  directly to `super().forward(...)` when `variant == "S0"` and no inference
  override is supplied. Wrap the result only to add an empty gate-diagnostic
  tuple; do not recompute tensors.

- [ ] **Step 7: Run Task 1 tests and inherited Task 05 gates**

  ```text
  python -m pytest tests/unit/test_stability_config.py tests/numerical/test_stability_s0.py tests/unit/test_recurrent_model.py tests/numerical/test_recurrent_equivalence.py tests/numerical/test_recurrent_gradients.py tests/numerical/test_recurrent_determinism.py -q
  ```

- [ ] **Step 8: Commit Task 1**

  ```text
  git add src/aurora/stability_config.py src/aurora/model/recurrent_stability.py tests/stability_helpers.py tests/unit/test_stability_config.py tests/numerical/test_stability_s0.py
  git commit -m "feat: add exact Task 06 S0 control"
  ```

### Task 2: S1 projected input anchoring

**Files:**
- Modify: `src/aurora/model/recurrent_stability.py`
- Create: `tests/numerical/test_stability_anchor.py`

**Interfaces:**
- Consumes: immutable prelude output `h_0` and shared core.
- Produces: `InputAnchor.forward(initial_state, recurrent_state) -> Tensor`.

- [ ] **Step 1: Write failing anchor tests**

  Use a literal projection matrix to verify:

  ```python
  anchored = recurrent_state + anchor.projection(initial_state)
  ```

  Register a prelude forward hook and an `InputAnchor` forward pre-hook during
  an R=4 model call. Assert all four anchor calls receive the exact prelude
  output object by Python identity and never a prior recurrent state.

- [ ] **Step 2: Verify the anchor tests fail**

  ```text
  python -m pytest tests/numerical/test_stability_anchor.py -q
  ```

  Expected: `InputAnchor` is missing.

- [ ] **Step 3: Implement S1**

  Add one bias-free `nn.Linear(d_model, d_model)` only when anchoring is enabled.
  Initialize it with `DenseTransformer._initialize_weights`. In every loop call:

  ```python
  core_input = self.input_anchor(initial_state, previous)
  proposal = self.core(core_input, iteration=iteration)
  hidden = proposal
  ```

  Never reassign `initial_state`.

- [ ] **Step 4: Verify S1 and S0**

  ```text
  python -m pytest tests/numerical/test_stability_anchor.py tests/numerical/test_stability_s0.py -q
  ```

- [ ] **Step 5: Commit Task 2**

  ```text
  git add src/aurora/model/recurrent_stability.py tests/numerical/test_stability_anchor.py
  git commit -m "feat: add projected recurrent input anchoring"
  ```

### Task 3: S2 gated interpolation and gate diagnostics

**Files:**
- Modify: `src/aurora/model/recurrent_stability.py`
- Create: `tests/unit/test_stability_gate.py`
- Create: `tests/numerical/test_stability_determinism.py`

**Interfaces:**
- Produces: `GatedRecurrentUpdate.forward(previous, proposal) -> tuple[Tensor,
  Tensor]`, `RecurrentGateDiagnostics`, and populated
  `StabilityLMOutput.gate_diagnostics`.

- [ ] **Step 1: Write a failing hand-derived interpolation test**

  Set the scalar gate projection weights and bias to literal values. Assert the
  returned gate equals `sigmoid(linear(concat(previous, proposal)))`, every gate
  is finite and in `[0,1]`, and the state equals:

  ```python
  expected = (1.0 - gate) * previous + gate * proposal
  ```

- [ ] **Step 2: Verify the gate test fails**

  ```text
  python -m pytest tests/unit/test_stability_gate.py -q
  ```

  Expected: `GatedRecurrentUpdate` is missing.

- [ ] **Step 3: Implement the minimal scalar gate**

  Create `nn.Linear(2 * d_model, 1, bias=True)`, concatenate previous/proposal
  along the hidden dimension, apply sigmoid, and broadcast the scalar gate.
  Initialize with the existing model initializer. Do not feed the gate into
  loop continuation or halting logic.

- [ ] **Step 4: Add failing diagnostics tests**

  Require per-iteration detached gate mean, population std, min, max, and a
  collapse label in `{"zero", "one", "none"}` using the fixed 0.05/0.95
  thresholds. Compare diagnostics-on/off forward calls for exact state, logits,
  and loss equality, and repeat deterministic CPU fp32 calls for exact gate
  diagnostics.

- [ ] **Step 5: Implement gate observations and verify**

  ```text
  python -m pytest tests/unit/test_stability_gate.py tests/numerical/test_stability_determinism.py tests/numerical/test_stability_s0.py -q
  ```

- [ ] **Step 6: Commit Task 3**

  ```text
  git add src/aurora/model/recurrent_stability.py tests/unit/test_stability_gate.py tests/numerical/test_stability_determinism.py
  git commit -m "feat: add gated recurrent interpolation"
  ```

### Task 4: S3 stabilization, S4 composition, and structural gates

**Files:**
- Modify: `src/aurora/model/recurrent_stability.py`
- Create: `tests/unit/test_stability_normalization.py`
- Create: `tests/numerical/test_stability_structure.py`
- Create: `tests/numerical/test_stability_gradients.py`

**Interfaces:**
- Produces: `InitialRMSStabilizer.forward(initial, candidate) -> Tensor` and the
  complete S0–S4 controlled model.

- [ ] **Step 1: Write failing RMS stabilization tests**

  Use nonzero literal tensors and assert per-token output RMS equals initial RMS.
  Use zero initial/candidate and extreme finite candidate fixtures and require
  finite outputs with no NaN/Inf.

- [ ] **Step 2: Verify the stabilization tests fail**

  ```text
  python -m pytest tests/unit/test_stability_normalization.py -q
  ```

  Expected: `InitialRMSStabilizer` is missing.

- [ ] **Step 3: Implement S3**

  Compute hidden-dimension RMS in fp32, clamp the candidate denominator at
  `norm_eps`, apply the scale in the candidate dtype, and add no parameters.
  Apply it after the proposal only for S3.

- [ ] **Step 4: Write S4 and structural failing tests**

  For S4, prove call order is anchor -> shared core -> gate and stabilization is
  absent. Across S0–S4 and R=1/2/4/8, assert each variant's parameter count and
  state-dict keys are invariant with R. Use hooks to prove the same core object
  is called R times, verify core parameter IDs/storage pointers, and verify an
  optimizer built from the model contains every trainable parameter once.

  Retain every state for S1/S2/S3/S4 at R=4, backpropagate loss, and require
  finite nonzero gradients on every state and every enabled mechanism.

- [ ] **Step 5: Implement S4 and inference overrides**

  S4 enables only the already-tested anchor and gate. Resolve
  `num_iterations_override` to the configured depth when absent; otherwise
  require an integer in `[1, max_iterations]`. No training caller supplies the
  override.

- [ ] **Step 6: Verify all model correctness gates**

  ```text
  python -m pytest tests/unit/test_stability_normalization.py tests/numerical/test_stability_structure.py tests/numerical/test_stability_gradients.py tests/numerical/test_stability_determinism.py tests/numerical/test_stability_anchor.py tests/unit/test_stability_gate.py tests/numerical/test_stability_s0.py -q
  ```

- [ ] **Step 7: Commit Task 4**

  ```text
  git add src/aurora/model/recurrent_stability.py tests/unit/test_stability_normalization.py tests/numerical/test_stability_structure.py tests/numerical/test_stability_gradients.py
  git commit -m "feat: add recurrent RMS stabilization controls"
  ```

### Task 5: Task 06 trainer, FLOPs, pipeline, and cross-depth evaluator

**Files:**
- Create: `src/aurora/training/stability_metrics.py`
- Create: `src/aurora/training/stability_trainer.py`
- Create: `src/aurora/cli/stability_tiny.py`
- Create: `src/aurora/cli/stability_cross_depth.py`
- Modify: `pyproject.toml`
- Create: `tests/unit/test_stability_metrics.py`
- Create: `tests/unit/test_stability_trainer.py`
- Create: `tests/integration/test_stability_pipeline.py`
- Create: `tests/integration/test_stability_cross_depth.py`

**Interfaces:**
- Produces: `estimate_stability_transformer_flops`, `StabilityTrainer`,
  `run_stability_pipeline`, `run_stability_experiment`, and
  `evaluate_cross_depth`.

- [ ] **Step 1: Write failing FLOP and trainer-record tests**

  Hand-derive tiny-config FLOPs. Start from Task 05 active layers and add, per
  iteration, `2*tokens*d_model^2` for S1/S4 and
  `4*tokens*d_model` for S2/S4. Assert S3 equals S0 in the matmul-only estimate.
  Train one step and require state diagnostics, core gradients, gate statistics,
  collapse labels, parameter counts, and active FLOPs in JSONL.

- [ ] **Step 2: Verify these tests fail**

  ```text
  python -m pytest tests/unit/test_stability_metrics.py tests/unit/test_stability_trainer.py -q
  ```

- [ ] **Step 3: Implement metrics and trainer**

  Copy the Task 05 training semantics into a Task 06-specific trainer without
  altering the frozen trainer. Add gate diagnostics to
  `StabilityTrainingResult` and JSONL. Preserve pre-clip whole-model/core
  gradients, finite aborts, scheduler order, checkpoint state, and optimizer
  uniqueness.

- [ ] **Step 4: Write a failing end-to-end pipeline test**

  With the byte-tokenizer fixture only, run a short S2 pipeline and require loss
  decrease, evaluation, exact fresh checkpoint reload, nonempty generation,
  parameter/FLOP fields, recurrent and gate diagnostics, and a recorded result
  identical to the returned dictionary.

- [ ] **Step 5: Implement the training CLI/pipeline**

  Reuse the frozen Task 05 corpus/dataset helpers and production BPE policy. Add
  console entry point:

  ```text
  aurora-stability = aurora.cli.stability_tiny:main
  ```

  Result JSON includes `variant`, mechanism settings, train/eval loss, unique
  parameters, active FLOPs/token, total training FLOPs, throughput, peak memory,
  gradient norms, final state/gate diagnostics, collapse status, checkpoint
  reload, tokenizer/data/config hashes, and generated text.

- [ ] **Step 6: Write a failing cross-depth test**

  Train/save an R=4 fixture, evaluate the exact checkpoint at
  `[1,2,3,4,6,8]`, and assert finite losses, matching diagnostic lengths,
  unchanged parameter tensors/configured training depth, and source
  run/config/checkpoint identity in the output.

- [ ] **Step 7: Implement cross-depth evaluation**

  Add console entry point:

  ```text
  aurora-stability-cross-depth = aurora.cli.stability_cross_depth:main
  ```

  Aggregate token-weighted loss and mean per-iteration state/gate diagnostics
  over the same evaluation batches for every inference depth. Record source
  checkpoint SHA-256. Do not step an optimizer or mutate weights.

- [ ] **Step 8: Verify Task 5**

  ```text
  python -m pytest tests/unit/test_stability_metrics.py tests/unit/test_stability_trainer.py tests/integration/test_stability_pipeline.py tests/integration/test_stability_cross_depth.py -q
  ```

- [ ] **Step 9: Commit Task 5**

  ```text
  git add src/aurora/training/stability_metrics.py src/aurora/training/stability_trainer.py src/aurora/cli/stability_tiny.py src/aurora/cli/stability_cross_depth.py pyproject.toml tests/unit/test_stability_metrics.py tests/unit/test_stability_trainer.py tests/integration/test_stability_pipeline.py tests/integration/test_stability_cross_depth.py
  git commit -m "feat: add Task 06 controlled experiment pipeline"
  ```

### Task 6: Immutable experiment configurations and independent runs

**Files:**
- Create: `configs/task06/s1_anchor_r4.yaml`
- Create: `configs/task06/s1_anchor_r8.yaml`
- Create: `configs/task06/s2_gate_r4.yaml`
- Create: `configs/task06/s2_gate_r8.yaml`
- Create: `configs/task06/s3_initial_rms_r4.yaml`
- Create: `configs/task06/s3_initial_rms_r8.yaml`
- Create after S1/S2 runs: `configs/task06/s4_anchor_gate_r4.yaml`
- Create after S1/S2 runs: `configs/task06/s4_anchor_gate_r8.yaml`

**Interfaces:**
- Consumes: Task 05 tiny protocol and `aurora-stability`.
- Produces: eight ignored immutable `EXP-06xx-*` run directories.

- [ ] **Step 1: Add and validate S1/S2/S3 configs**

  Copy every non-mechanism setting from `configs/task05/recurrent_4.yaml` and
  `recurrent_8.yaml`. Use run IDs:

  ```text
  EXP-0614-anchor-r4
  EXP-0618-anchor-r8
  EXP-0624-gate-r4
  EXP-0628-gate-r8
  EXP-0634-initial-rms-r4
  EXP-0638-initial-rms-r8
  ```

  Parse each config and assert shared seed, tokenizer, dataset, batch/context,
  optimizer, scheduler, and runtime fields match the frozen controls.

- [ ] **Step 2: Commit independent configs before running**

  ```text
  git add configs/task06/s1_anchor_r4.yaml configs/task06/s1_anchor_r8.yaml configs/task06/s2_gate_r4.yaml configs/task06/s2_gate_r8.yaml configs/task06/s3_initial_rms_r4.yaml configs/task06/s3_initial_rms_r8.yaml
  git commit -m "config: define independent Task 06 stability runs"
  ```

- [ ] **Step 3: Run S1/S2/S3 at R=4/R=8**

  Run `aurora-stability --config <path> --runs-dir runs` once per config. Verify
  every result is completed, finite, checkpoint-reloaded, and records the
  expected shared dataset/tokenizer identity. Do not edit the architecture in
  response to outcomes.

- [ ] **Step 4: Add S4 configs only after S1/S2 artifacts exist**

  Use run IDs:

  ```text
  EXP-0644-anchor-gate-r4
  EXP-0648-anchor-gate-r8
  ```

  Set anchoring and gating true and stabilization `none`; keep every other
  setting matched.

- [ ] **Step 5: Commit and run S4**

  ```text
  git add configs/task06/s4_anchor_gate_r4.yaml configs/task06/s4_anchor_gate_r8.yaml
  git commit -m "config: define Task 06 anchor-gate runs"
  ```

  Then execute both S4 configs and validate artifacts exactly as above.

### Task 7: Cross-depth diagnostic and research records

**Files:**
- Modify: `docs/EXPERIMENTS.md`
- Modify: `docs/DECISIONS.md`
- Modify: `docs/PROGRESS.md`

**Interfaces:**
- Consumes: frozen S0 R=4/R=8 records and all eight new Task 06 result files.
- Produces: one ignored cross-depth result and the final evidence-backed Task 06
  record.

- [ ] **Step 1: Select the R=4 checkpoint by the predeclared rule**

  Compare S1–S4 R=4 final hidden RMS and cosine-to-`h_0` with frozen S0 R=4.
  Among variants meeting both stability conditions and finite diagnostics,
  choose the lowest evaluation loss. If none qualifies, choose the lowest
  finite evaluation loss and mark the filter unmet. Record the selection inputs
  before evaluating cross-depth.

- [ ] **Step 2: Run one cross-depth diagnostic**

  Evaluate the selected unchanged checkpoint at `R_eval=1,2,3,4,6,8`. Write an
  ignored immutable analysis result with source run/config/checkpoint hashes,
  losses, state diagnostics, gate diagnostics where applicable, parameter
  count, and active FLOPs/token per depth.

- [ ] **Step 3: Update experiments with actual values only**

  Add explicit S0–S4 hypotheses, the complete R=4/R=8 matrix, compute and
  parameter counts, train/eval loss, throughput, memory availability,
  gradient/state/gate diagnostics, collapse decisions, and the cross-depth
  table. State negative or mixed results without tuning explanations.

- [ ] **Step 4: Update decisions and progress**

  Record the projected additive anchor, scalar interpolation gate, initial-RMS
  stabilizer, rejected more-complex alternatives, selection rule, limitations,
  and five requested research answers. `PROGRESS.md` contains validated results
  only and states that Task 07 has not started.

- [ ] **Step 5: Commit research records**

  ```text
  git add docs/EXPERIMENTS.md docs/DECISIONS.md docs/PROGRESS.md
  git commit -m "docs: record Task 06 recurrent stability results"
  ```

### Task 8: Final verification and scope audit

**Files:**
- Verify all Task 06 changes; do not add Task 07 code.

- [ ] **Step 1: Run the complete quality gate**

  ```text
  python -m pytest -q
  python -m ruff format --check .
  python -m ruff check .
  python -m mypy src
  ```

- [ ] **Step 2: Audit frozen controls and generated artifacts**

  Require no diff to frozen Task 05 implementation/config files, no tracked
  `runs/`, identical Task 05 result text, a clean worktree, and no forbidden
  Task 07+ mechanism. Compare documentation numbers mechanically with ignored
  JSON results.

- [ ] **Step 3: Review the complete branch diff**

  Review from merge base `f2ecdac7ac7bf2a613e86e7453c376532566c81e`
  through HEAD for specification compliance, mathematical correctness,
  deterministic/output-neutral diagnostics, experiment validity, and honest
  conclusions. Fix Critical/Important findings under covering tests.

- [ ] **Step 4: Stop at Task 06**

  Report the five requested research answers, test evidence, experiment matrix,
  cross-depth result, final commit/diff, limitations, and explicit Task 07 stop.
