# Task 05 Fixed-Loop Recurrence Design

## Scope

Task 05 adds a decoder-only recurrent language model beside the unchanged
`DenseTransformer` B0 control. The model is an explicit composition:

```text
token embedding -> fixed prelude -> shared recurrent core x R -> fixed coda
                -> final RMSNorm -> tied language-model head
```

`R` is a configuration-selected fixed count in `{1, 2, 4, 8}`. The core is one
module object containing one set of Transformer-block parameters. Repetition is
implemented by calling that object in a Python loop, never by constructing a
module list of per-iteration copies.

This task does not add adaptive halting, input-anchor gates, neural memory,
scratch tokens, EAC, routing, STARS/JSRR regularization, randomized depth,
shortcut consistency, or any Task 06+ mechanism. Task 05 includes only the
minimal observational diagnostics explicitly required by its acceptance gate.

## Configuration and B0 isolation

Dense B0 configuration and model classes remain byte-for-byte unchanged.
Recurrent experiments use `RecurrentExperimentConfig` and
`RecurrentModelConfig` in a separate module. Recurrent model settings are:

- the existing vocabulary, context, width, attention, feed-forward, dropout,
  RoPE, QK-normalization, embedding-tying, and RMSNorm settings;
- positive counts for prelude, core, and coda blocks;
- `num_iterations` in `{1, 2, 4, 8}`;
- `max_iterations = 8` for all Task 05 comparison configurations.

The learned recurrent-step embedding has shape `[max_iterations, d_model]` and
is added before the shared core blocks. It supplies the Phase 2 step
conditioning named in `plan.md` without introducing the gated/time-modulated
updates reserved for Task 07. Because its size depends only on
`max_iterations`, changing `num_iterations` cannot change parameters or
state-dict structure.

## Model interfaces

`TransformerStage` owns a fixed sequence of unique `DecoderBlock` objects.
`RecurrentCore` owns one stage plus the fixed-size step embedding and accepts an
iteration index, defaulting to zero. `FixedLoopRecurrentTransformer.forward`
embeds tokens, runs the prelude once, invokes the exact same `core` object R
times, runs the coda once, and applies the final norm/head.

The prelude, core, coda, final norm, and head remain directly callable so the
one-loop path can be compared with the hand-expanded expression
`coda(core(prelude(embedding(input))))`. With dropout disabled in deterministic
CPU fp32, logits and aligned next-token loss must be bitwise identical.

The recurrent output can optionally retain `h0` and every post-core state for
gradient-through-time and determinism tests. Ordinary training does not request
those retained outputs.

## Weight sharing and gradients

Tests prove sharing through independent observations:

- every hook invocation receives `model.core` by Python identity;
- core parameter object IDs and storage pointers are unchanged before and after
  all loop calls;
- state-dict keys contain one `core.*` namespace and no iteration-indexed core
  copies;
- models at R=1/2/4/8 have identical trainable counts and state-dict keys;
- the optimizer contains each unique trainable parameter exactly once.

The one-loop loss must produce finite, nonzero parameter gradients in prelude,
core, and coda. A multi-loop test retains each recurrent state gradient and
requires finite, nonzero gradients at every state, catching an accidental
`detach()` or stop-gradient between iterations.

## Determinism and diagnostics

Repeated deterministic CPU fp32 forward passes with identical model state,
inputs, seed, and loop count must produce bitwise-identical recurrent states,
logits, and loss.

Optional forward diagnostics observe detached tensors and report per iteration:

- hidden-state RMS norm;
- recurrent-update RMS norm `RMS(h_(r+1) - h_r)`;
- mean per-example cosine similarity between successive flattened states.

After backward and before clipping, the recurrent trainer reports core-gradient
L2 norm, mean absolute value, maximum absolute value, nonzero fraction, and
finiteness. A test compares diagnostic-on and diagnostic-off forwards and
requires bitwise-identical model outputs.

## Training, compute, and experiments

`RecurrentTrainer` reuses Task 04 device, optimizer, scheduler, sampler,
checkpoint, evaluation, and resume behavior while overriding only the training
step required to record recurrent diagnostics and recurrent FLOPs. The FLOP
estimate uses the existing documented matmul-only convention:

```text
active blocks = prelude_blocks + coda_blocks + R * core_blocks
```

It reports inference active FLOPs/token and total estimated training FLOPs.
Normalization, activation, softmax, embeddings, and diagnostic reductions
remain excluded and the estimate is labeled accordingly.

Five CPU fp32 runs use the same generated corpus family, tokenizer policy,
seed, training steps, token count, width, and context:

- B0 unchanged dense control;
- recurrent R=1;
- recurrent R=2;
- recurrent R=4;
- recurrent R=8.

The records include parameter counts, initial/final/evaluation loss, active
FLOPs/token, total training FLOPs, throughput, peak VRAM availability, total and
core gradient norms, and recurrent-state diagnostics. Comparisons present loss
beside compute and throughput; they do not claim architectural improvement from
raw loss when active compute differs.

## Acceptance

Completion requires all seven user-specified gates, the inherited Tasks 01–04
suite, Ruff, mypy, a complete tiny recurrent integration test, actual controlled
runs for B0/R1/R2/R4/R8, and research-record updates containing only values read
from generated artifacts. Task 06 is not started.
