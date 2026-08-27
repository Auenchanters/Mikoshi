# Research and Engineering Decisions

## DEC-0001 — Dense-only control for Tasks 01–04

- **Status:** accepted as an engineering control; no AURORA research conclusion.
- **Hypothesis:** a small modern dense Transformer should learn a deliberately
  tiny repeated corpus and provide a reproducible control before recurrence or
  memory is introduced.
- **Implementation:** decoder-only pre-normalized Transformer with RMSNorm,
  RoPE, grouped-query causal attention, switchable QK normalization, SwiGLU,
  tied embeddings, and AdamW.
- **Control/baseline:** this model is the dense control for Task 05 and later
  recurrent comparisons. No recurrent, scratch, memory, or routing component is
  present.
- **Metric:** finite forward/backward values, loss decrease, causal masking,
  generation, complete checkpoint reload, and exact CPU fp32 replay.
- **Expected result:** loss decreases on a tiny repeated corpus and all
  reproducibility tests pass.
- **Actual result:** `EXP-0001-dense-sanity` decreased from 5.7461218834 to
  3.8214008808 loss; evaluation loss was 3.7949758768. The 41-test suite passed,
  including exact CPU fp32 resume replay.
- **Interpretation:** the dense control and infrastructure satisfy the initial
  engineering acceptance gate. The run is too small and artificial to support
  a scientific model-quality claim.
- **Hypothesis survives:** yes, only as a foundation/control decision.

## DEC-0002 — Production BPE and test-only byte fixture

- **Status:** accepted.
- **Hypothesis:** using a mature BPE implementation while owning the corpus,
  preprocessing specification, and tokenizer artifact gives reproducibility
  without conflating the main experiment with a tokenizer-free architecture.
- **Implementation:** Hugging Face `tokenizers` BPE with strict UTF-8, LF,
  NFKC, preserved case/whitespace, ByteLevel pre-tokenization and decoding, full
  byte alphabet, fixed special IDs, recorded library version, corpus-manifest
  hash, and artifact SHA-256.
- **Control/baseline:** a fixed byte tokenizer is injectable only from
  `tests/fixtures`; production configuration rejects it.
- **Metric:** train twice from the same ordered corpus and compare serialized
  tokenizer bytes and hashes; round-trip ASCII, Unicode, whitespace, and unseen
  valid UTF-8.
- **Expected result:** identical tokenizer artifacts for identical ordered input
  and lossless NFKC/ByteLevel round trips.
- **Actual result:** all tokenizer tests passed. The verified production
  tokenizer used `tokenizers` 0.22.2 and SHA-256
  `2a616f264e588f86c1bad49e2ff2728125b115991b5e1329bf5bb23fc3e390f5`.
- **Interpretation:** tokenizer identity is explicit and swappable without
  allowing the byte fixture to become the research default.
- **Hypothesis survives:** yes.

## DEC-0003 — Exactly one next-token shift

- **Status:** accepted.
- **Hypothesis:** target alignment must be owned by one interface to prevent a
  silent two-token shift.
- **Implementation:** `TokenBlockDataset` returns input windows and targets
  shifted by one token; `DenseTransformer` computes cross-entropy directly
  against those aligned targets.
- **Control/baseline:** a hand-derived cross-entropy test compares the model loss
  against explicitly shifted target IDs.
- **Metric:** exact equality to PyTorch cross-entropy on the aligned fixture.
- **Expected result:** the data→model path predicts the immediate next token.
- **Actual result:** the unit test passes and the end-to-end tiny run learns.
- **Interpretation:** the trainer does not apply a second hidden shift.
- **Hypothesis survives:** yes.

## DEC-0004 — Single-device trainer before distributed training

- **Status:** accepted for this phase.
- **Hypothesis:** exact single-device resume should be established before adding
  distributed sampler and optimizer state.
- **Implementation:** CPU/CUDA selection, CPU fp32, CUDA bf16/fp16 capability
  checks, optional fp16 scaler, deterministic sampler cursor, RNG capture, and
  atomic checkpoints.
- **Control/baseline:** uninterrupted CPU fp32 training versus a run split by a
  checkpoint and resumed into a freshly initialized model.
- **Metric:** exact per-step loss list, counters, and final parameters.
- **Expected result:** bitwise equality on CPU fp32.
- **Actual result:** exact equality verified by
  `tests/numerical/test_resume_replay.py` and by the fresh reload in
  `EXP-0001-dense-sanity`.
- **Interpretation:** single-device trajectory state is complete. CUDA/bf16 and
  distributed determinism remain unverified on this CPU-only host.
- **Hypothesis survives:** yes for CPU fp32.

## DEC-0005 — Fixed-loop prelude/shared-core/coda before adaptive recurrence

- **Status:** accepted as the Task 05 reference architecture; no performance
  claim.
- **Hypothesis:** a fixed prelude, one repeatedly invoked core object, and fixed
  coda can provide a numerically valid recurrent control whose parameter count
  is independent of loop count.
- **Implementation:** one prelude block, one shared core block, one coda block,
  and a learned `[8, d_model]` step-embedding table. Configured iteration counts
  are restricted to 1, 2, 4, or 8. No core copies, adaptive halting, anchors,
  gates, scratch state, memory, EAC, routing, randomized depth, or stability
  regularization are present.
- **Scale choice:** the configurable implementation supports positive counts per
  stage, but the CPU-safe Task 05 sweep uses 1/1/1 rather than the plan's later
  2/2/1–2 science-model suggestion. This keeps the run cheap and is not a claim
  that 1/1/1 is the preferred research architecture.
- **Control/baseline:** the Tasks 01–04 `DenseTransformer` and its B0 model/data/
  optimizer/scheduler/runtime settings remain unchanged.
- **Metric:** exact R=1 expansion, finite nonzero prelude/core/coda gradients,
  retained-state gradients through R=4, repeated-core hook identity, parameter
  IDs/storage pointers, optimizer uniqueness, state-dict/parameter invariance,
  CPU fp32 determinism, and diagnostic noninterference.
- **Expected result:** all structural and numerical gates pass; training loss
  decreases at each fixed depth.
- **Actual result:** all gates passed in the 66-test suite. R=1/2/4/8 each had
  91,152 trainable parameters and identical state-dict key sets. All four tiny
  recurrent runs decreased training loss and reproduced their saved model and
  evaluation loss exactly after fresh CPU fp32 checkpoint reload. Detached
  instrumentation also recorded cosine to the initial recurrent state and
  `||h_t-h_{t-1}|| / ||h_t||` without changing logits or loss.
- **Interpretation:** the fixed-loop implementation is a valid engineering
  control for recurrence. It does not establish a recurrence benefit.
- **Hypothesis survives:** yes as an implementation/control hypothesis only.

## DEC-0006 — Compare recurrence beside compute, not by raw loss alone

- **Status:** accepted.
- **Hypothesis:** fixed-loop results are interpretable only when parameter count,
  active FLOPs/token, throughput, and training compute are shown beside loss.
- **Implementation:** use the existing matmul-only FLOP convention with active
  blocks `prelude + coda + iterations × core`; report loss decrease per training
  GFLOP only as a descriptive quantity.
- **Control/baseline:** B0 has 65,328 parameters and 142,464 active FLOPs/token.
  Task 05 recurrent models have 91,152 parameters at every tested depth and
  199,296–597,120 active FLOPs/token.
- **Actual result:** R=1 produced the lowest raw evaluation loss (3.7306981683),
  but used 1.395× the B0 parameters and 1.399× the active FLOPs/token. R=2/4/8
  used progressively more compute and produced evaluation losses 3.8169981837,
  4.0504400730, and 4.1997586489. Loss decrease per training GFLOP declined
  from 0.8795720809 for B0 to 0.1584378633 for R=8.
- **Interpretation:** no recurrent configuration demonstrated a controlled
  advantage over B0 because neither parameter nor FLOP matching was performed.
  Deeper naïve recurrence degraded performance in this experiment despite its
  additional active compute. The single-seed repeated-corpus run is an
  engineering diagnostic, not a general model comparison, and the unmatched
  R=1 raw loss must not be described as beating B0.
- **Hypothesis survives:** yes; all future claims require matched controls and
  multiple seeds.

## DEC-0007 — Test minimal recurrent-state controls independently

- **Status:** accepted as the Task 06 controlled diagnostic design; results are
  mixed and do not support a model-quality claim.
- **Hypothesis:** three deliberately small controls may mitigate distinct parts
  of the frozen Task 05 R=4/R=8 failure pattern: directional drift, unconstrained
  update size, and RMS growth.
- **S1 projected additive anchor:** preserve the immutable prelude output `h0`
  and add one bias-free `d_model -> d_model` projection of it to the recurrent
  state before every call to the same shared core. This adds 2,304 parameters
  and 18,432/36,864 active FLOPs/token at R=4/R=8.
- **S2 scalar interpolation gate:** compute one sigmoid scalar per token from an
  affine projection of `[h_t; proposal_t]`, then form
  `(1-g_t)h_t + g_t proposal_t`. The gate controls interpolation only; it never
  halts execution. This adds 97 parameters and 768/1,536 active FLOPs/token at
  R=4/R=8 under the matmul-only convention.
- **S3 initial-RMS stabilizer:** after the core proposal, rescale each token's
  candidate so its hidden-dimension RMS equals that token's initial-state RMS,
  using a clamped denominator. It adds no parameters or charged matmul FLOPs.
- **S4 composition:** combine only the already-tested S1 anchor and S2 gate, in
  anchor -> shared core -> gate order, with no stabilizer. This adds 2,401
  parameters and 19,200/38,400 active FLOPs/token at R=4/R=8.
- **Rejected complexity for this task:** learned or gated anchors,
  vector/channel-wise or depth-specific gates, learned normalization,
  combinations involving S3, randomized training depth, adaptive halting,
  routing, scratch/state highways, Jacobian/STARS regularization, persistent
  memory, EAC, tokenizer-free modeling, and custom CUDA/Triton kernels. These
  alternatives would confound the independent mechanism comparison or cross
  into Task 07+.
- **Controls:** S0 is the exact delegated Task 05 path when every mechanism is
  disabled; frozen Task 05 R=4/R=8 artifacts were not rerun or edited. All new
  runs retained seed 17, CPU fp32, production tokenizer/corpus, 40 steps, 5,120
  tokens, data order, optimizer, scheduler, batch, and context.
- **Actual stability result:** relative to depth-matched S0, every Task 06 run
  had lower final hidden RMS. S1/S2/S4 increased final cosine to `h0` at both
  depths; S3 decreased it at R=4 and increased it at R=8. S3 held its RMS
  essentially constant across iterations by construction. No S2/S4 gate
  collapsed under the predeclared 0.05/0.95 thresholds.
- **Actual loss result:** R=4 evaluation losses for S1/S2/S3/S4 were
  3.8888302743 / 3.8293781579 / 4.2306061983 / 3.7531955242; R=8 losses were
  3.9617625475 / 4.0581546426 / 4.2106192112 / 3.9005704820. These are raw
  outcomes beside unequal parameter/compute costs, not improvement claims.
- **Interpretation:** projected anchoring and scalar interpolation are useful
  minimal controls because they measurably change the targeted state
  diagnostics without gate collapse. The parameter-free RMS constraint controls
  magnitude but did not improve directional alignment at R=4 or raw loss.
  Successive-state cosines still approached one, so no mechanism demonstrated
  prevention of state convergence.
- **Hypothesis survives:** partially as a diagnostic hypothesis. RMS control and
  some directional-drift reduction were observed; broader stability and
  model-quality benefits were not established.

## DEC-0008 — Predeclare checkpoint selection and keep cross-depth diagnostic-only

- **Status:** accepted and executed once.
- **Selection rule:** among S1–S4 R=4 runs with finite diagnostics and both a
  lower final hidden RMS and a higher final cosine to `h0` than frozen S0 R=4,
  choose the lowest evaluation loss. If no candidate qualifies, choose the
  lowest finite evaluation loss and mark the stability filter unmet.
- **Frozen inputs:** S0 R=4 final hidden RMS was 0.3238864541 and cosine to `h0`
  was 0.6662002206.
- **Selection result:** S1, S2, and S4 met both conditions; S3 failed the cosine
  condition. The filter was met, and S4 R=4 was selected with evaluation loss
  3.7531955242, final RMS 0.1380776316, and cosine to `h0` 0.8797061443.
- **Identity:** source run `EXP-0644-anchor-gate-r4`, config SHA-256
  `c50c114a331438b94c3b811690dcd4a9d12bdb7cc5e8e2b66394ef56f13efeaf`,
  checkpoint SHA-256
  `c94cb3d3386c251285432cfad36f0fd1b787c246f30b6f308f64eb1789760bee`.
  The exact checkpoint was evaluated at R=1/2/3/4/6/8 with no optimizer,
  weight update, configured-training-depth mutation, or randomized depth.
- **Cross-depth result:** evaluation losses were 3.8521460295 / 3.7836607397 /
  3.7593515813 / 3.7531955242 / 3.7674068511 / 3.7994184196. All state and gate
  diagnostics were finite, and no gate collapsed. The trained depth R=4 was the
  minimum; beyond it, loss and RMS rose while cosine to `h0` fell.
- **Research answer — drift:** S1/S2/S4 reduced final directional drift at both
  trained depths; S3 was mixed across depths.
- **Research answer — RMS growth:** every mechanism reduced final RMS versus
  depth-matched S0, and S3 enforced constant RMS, but lower RMS alone did not
  imply lower loss.
- **Research answer — successive-state convergence:** no mechanism prevented
  near-collinearity. At R=4 all final successive-state cosines were slightly
  higher than S0; at R=8 S3 reduced the value most but still reached
  0.9954776764.
- **Research answer — deeper validation:** S1/S2/S4 had lower raw R=8 losses
  than S0 R=8 and S3 had a higher one, but unequal compute/parameters prevent an
  improvement claim. Except S3, each separately trained R=8 variant was worse
  than its R=4 counterpart.
- **Research answer — transfer:** the one selected S4/4 checkpoint remained
  finite and noncollapsed at R=6/R=8, but its best loss occurred at training
  depth and diagnostics continued to drift. Transfer is partial in this one
  diagnostic and not established generally.
- **Limitations:** one seed; tiny 91K–94K models; 5,120 training tokens; repeated
  generated corpus and same-family evaluation; separately trained depth rows;
  unmatched parameters and active compute; matmul-only FLOPs that omit
  elementwise mechanism costs; CPU peak VRAM unavailable; host-dependent
  throughput; selection using the same R=4 evaluation set after a predeclared
  filter; cross-depth analysis of only one checkpoint; no significance test,
  independent corpus, large-scale, GPU, downstream-reasoning, or multi-seed
  evidence.
- **Conclusion boundary:** this is engineering evidence about measurable state
  controls, not evidence that recurrence is superior, that language modeling
  improved under matched conditions, or that AURORA is successful. Research
  Task 07 has not started.
