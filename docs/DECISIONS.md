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
