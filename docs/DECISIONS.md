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
