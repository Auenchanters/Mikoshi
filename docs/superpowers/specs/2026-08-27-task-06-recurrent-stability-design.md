# Task 06 Recurrent Stability Design

## Scope and frozen controls

Task 06 tests whether deliberately controlled recurrent-state updates mitigate
the failure pattern measured in Task 05. The dense B0 implementation, the Task
05 `FixedLoopRecurrentTransformer`, its configurations, and its recorded
B0/R=1/2/4/8 results remain unchanged. Task 06 code lives in new stability
configuration, model, trainer, metric, CLI, config, and test files. Research
records may append Task 06 sections but must not rewrite Task 05 measurements.

The frozen failure controls are the Task 05 R=4 and R=8 runs. In particular,
R=8 ended with evaluation loss 4.1997586489, hidden-state RMS 1.0111789703,
`cos(h_t,h_0)` 0.4045824409, relative update 0.1330823600, and successive-state
cosine 0.9992300272. Task 06 does not rerun or overwrite those run IDs.

No Task 07+ feature is permitted: no persistent or Titans/ATLAS memory, EAC,
adaptive halting, randomized recurrent depth, routing, STARS/Jacobian
regularization, tokenizer-free model, or custom CUDA/Triton work.

## Configuration and variants

`StabilityModelConfig` extends the Task 05 architectural fields with three
independent controls:

- `input_anchoring: bool`;
- `gated_update: bool`;
- `state_stabilization: "none" | "initial_rms"`.

Validation accepts exactly the requested combinations:

| Variant | Anchor | Gate | Stabilization |
|---|---|---|---|
| S0 | off | off | none |
| S1 | on | off | none |
| S2 | off | on | none |
| S3 | off | off | initial_rms |
| S4 | on | on | none |

Training depth remains fixed in `{1, 2, 4, 8}` with `max_iterations = 8`.
Inference-only cross-depth evaluation may override the loop count with any
integer from 1 through 8. The override is never sampled during training.

## Controlled recurrent step

`ControlledRecurrentTransformer` subclasses the frozen Task 05 model. It builds
the same embedding, prelude, single shared core object, coda, final norm, and
head from an equivalent `RecurrentModelConfig`. With every mechanism disabled,
its forward method delegates directly to the Task 05 forward method. Under the
same seed and inputs, parameters, recurrent states, diagnostics, logits, and
loss must therefore match exactly.

For enabled mechanisms, the prelude output is stored once as immutable `h_0`.
At recurrent iteration `t`:

```text
core_input_t = h_t + AnchorProjection(h_0)   if anchoring is enabled
               h_t                           otherwise

proposal_t = SharedCore(core_input_t, t)

candidate_t = (1 - g_t) * h_t + g_t * proposal_t   if gating is enabled
              proposal_t                            otherwise

h_(t+1) = InitialRMSStabilizer(h_0, candidate_t)   for S3
          candidate_t                               otherwise
```

The core is the exact same Python module object on every call. Anchoring always
reads the original prelude tensor, never the prior anchored input or recurrent
state.

### S1 — input anchoring

The anchor is one bias-free `d_model -> d_model` linear projection added to the
current recurrent state before the shared core. It is intentionally the
simplest trainable projected/additive formulation. The projection is computed
from `h_0` at every iteration. There is no learned anchor gate or depth-specific
anchor parameter.

### S2 — gated update

The gate is one affine projection from concatenated `[h_t; proposal_t]` to one
scalar per batch/token position. A sigmoid constrains the gate to `[0,1]`; the
scalar broadcasts over the hidden dimension. Bias starts at zero under the
existing initializer, so the implementation does not hand-tune a preference
for copying or updating. This is interpolation only, never a halting decision.

### S3 — recurrent-state stabilization

The stabilizer has no parameters. For each batch/token position it rescales the
candidate so its hidden-dimension RMS equals the corresponding `h_0` RMS:

```text
h_(t+1) = candidate_t * RMS(h_0) / max(RMS(candidate_t), norm_eps)
```

This targets the measured state-magnitude failure directly while remaining
separate from anchoring and gating. The clamped denominator keeps zero or tiny
candidates finite. It is not combined with S1/S2 during Task 06.

### S4 — anchor plus gate

S4 composes S1's core input with S2's post-core interpolation and nothing else.
It is implemented and run only after the independent S1 and S2 unit/numerical
tests and controlled experiments pass.

## Diagnostics and correctness

The existing Task 05 hidden-state RMS, update RMS, relative update,
successive-state cosine, cosine-to-`h_0`, and core-gradient observations remain
the state diagnostics. They measure the final controlled state, not the raw
proposal, so variants are comparable at the recurrent-state boundary.

Gated variants additionally record detached per-iteration gate mean, population
standard deviation, minimum, and maximum. A run reports gate collapse toward
zero when mean is at most 0.05 and toward one when mean is at least 0.95; any
other mean is recorded as no collapse. These thresholds are fixed before the
experiments. Diagnostic collection must not change recurrent states, logits, or
loss.

All Task 05 gates remain active. New tests additionally prove disabled-path
identity, immutable anchor sourcing, exact interpolation arithmetic, finite
bounded gates, finite stabilization, shared-core identity/count invariance,
optimizer uniqueness, gradient-through-time, determinism, and instrumentation
noninterference.

## Compute accounting

Active FLOPs retain the Task 05 matmul-only convention. The estimate adds:

- anchor projection: `2 * tokens * d_model^2` per recurrent iteration;
- scalar gate projection: `4 * tokens * d_model` per recurrent iteration;
- no matmul charge for fixed RMS stabilization.

Sigmoid, interpolation, normalization, activation, softmax, embeddings, and
diagnostic reductions remain excluded and are labeled as such. Training uses
the existing three-forward-equivalent convention. Actual unique parameter
counts, throughput, CPU peak-VRAM unavailability, gradient norms, and all state
and gate diagnostics are recorded beside loss.

## Controlled experiment matrix

All new training runs use the Task 05 tiny protocol unchanged: seed 17, CPU
fp32, production BPE and corpus identity, 40 steps, 5,120 tokens, batch/context,
optimizer, schedule, and data order. Task 05 S0 R=4/R=8 artifacts are referenced
as frozen controls rather than rerun.

The smallest matrix with two representative depths is:

| Variant | R=4 | R=8 |
|---|---|---|
| S0 frozen control | existing | existing |
| S1 anchor | train | train |
| S2 gate | train | train |
| S3 initial-RMS stabilization | train | train |
| S4 anchor+gate | train after S1/S2 | train after S1/S2 |

Each new run uses a unique immutable `EXP-06xx-*` directory. S4 is not launched
until S1/S2 results exist.

For the cross-depth diagnostic, select the R=4 checkpoint with the lowest
evaluation loss among S1–S4 variants that both (a) has finite diagnostics and
(b) improves final hidden RMS and cosine-to-`h_0` relative to frozen S0 R=4. If
none meets that filter, select the lowest finite R=4 evaluation loss and label
the stability filter as unmet. Evaluate the exact same weights at
`R_eval = 1, 2, 3, 4, 6, 8` without training, randomized depth, or parameter
changes.

## Interpretation and stop boundary

Results answer five questions separately: whether a mechanism reduces drift,
controls RMS growth, prevents successive-state convergence, improves deeper
validation behavior, and transfers beyond its training depth. Raw loss is
always presented beside parameter and active-compute differences. Negative and
mixed outcomes are preserved. Task 06 ends after the tests, controlled matrix,
cross-depth diagnostic, and research-record updates; Task 07 does not start.
