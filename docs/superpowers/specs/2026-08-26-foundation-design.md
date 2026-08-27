# AURORA Tasks 01–04 Foundation Design

## Scope

This phase builds the controlled dense-language-model foundation required by
`plan.md` Tasks 01–04:

1. repository packaging and configuration;
2. trainable BPE tokenization, deterministic data ordering, and resumable data
   position;
3. a modern dense causal Transformer control;
4. training, evaluation, generation, checkpointing, and experiment records.

It deliberately excludes recurrence, scratch/state highways, persistent neural
memory, Experience-Adaptive Compute, learned halting, custom kernels,
distributed training, and large training runs. Those exclusions prevent Tier B
or Tier C ideas from silently entering the experimental control.

## Approaches considered

### Selected: small first-principles PyTorch package

Use ordinary Python dataclasses, PyYAML, Hugging Face `tokenizers`, and readable
PyTorch modules. Keep orchestration explicit and dependency-light. This offers
the strongest auditability, makes mathematical tests straightforward, and
leaves clear interfaces for later backbone replacement.

### Rejected: framework-heavy training stack

Hydra, Lightning, Accelerate, and a dataset framework would reduce some
boilerplate but add configuration and lifecycle behavior that is harder to
audit during deterministic checkpoint/resume tests. They can be reconsidered
when multi-device scale becomes necessary.

### Rejected: custom tokenizer and training framework

A pure-Python BPE implementation would duplicate mature infrastructure and add
an unnecessary source of correctness and performance risk. The project should
own the tokenizer specification and artifacts, not reimplement the BPE
algorithm.

## Package and repository layout

The installable package remains under `src/aurora`:

```text
src/aurora/
  config.py                 validated dataclass configuration and hashing
  experiment.py             immutable run directories and metadata
  tokenization/
    base.py                 generic tokenizer protocol and artifact metadata
    bpe.py                  train/load/save Hugging Face BPE tokenizer
  data/
    dataset.py              causal token-block dataset
    sampler.py              deterministic resumable batch sampler
  model/
    norms.py                RMSNorm
    rope.py                 rotary position embedding
    attention.py            causal grouped-query self-attention
    mlp.py                  SwiGLU
    transformer.py          dense decoder and language-model loss
  training/
    reproducibility.py      seeding and RNG capture/restore
    metrics.py              parameter/FLOP/system measurements
    checkpoint.py           atomic complete-state checkpoints
    trainer.py              train/evaluate/generate orchestration
  cli/
    tiny.py                 cheap end-to-end research-path experiment
```

The deterministic byte tokenizer lives only in
`tests/fixtures/byte_tokenizer.py`. Production code cannot select it through a
configuration value or CLI flag.

## Configuration

Nested frozen dataclasses define tokenizer, data, model, optimizer, scheduler,
runtime, checkpoint, and experiment fields. YAML files are loaded with
`safe_load`, rejected on unknown keys, validated for cross-field constraints,
and serialized to a canonical primitive dictionary. A SHA-256 digest of
canonical JSON is the config identity.

Every run writes the resolved YAML and config hash before training. The
checkpoint also stores the resolved config and hash. Resume rejects a config
whose trajectory-affecting fields differ, while allowing explicitly documented
runtime-only changes such as the output path.

Initial configuration families are:

- `configs/debug/dense_tiny.yaml`: CPU-safe, short-context correctness run;
- `configs/debug/dense_overfit.yaml`: deliberately tiny overfit experiment.

No 30M or larger run configuration is added in this phase because the initial
brief forbids large training.

## Tokenization

### Research path

`BPETokenizer` trains a tokenizer from an explicitly ordered local corpus using
Hugging Face `tokenizers`. It records the library version, tokenizer JSON
SHA-256, corpus-manifest SHA-256, and all preprocessing/training settings.

The initial BPE policy is explicit:

- input decoding: UTF-8 with strict error handling;
- line endings: canonical `\n`;
- normalization: Unicode NFKC;
- case: preserved;
- whitespace: preserved, not collapsed;
- pre-tokenizer: ByteLevel, no synthetic prefix space;
- decoder: ByteLevel;
- special-token order and IDs: `<pad>` 0, `<bos>` 1, `<eos>` 2, `<unk>` 3;
- research vocabulary default: 16,384, configurable up to the plan's 32K
  range;
- debug vocabulary: at least 260 so four special tokens plus the full byte
  alphabet are representable;
- minimum frequency: 2 by default, configurable;
- byte fallback: guaranteed through the complete ByteLevel alphabet rather
  than silently substituting `<unk>` for valid UTF-8 bytes;
- corpus order: manifest order is authoritative and recorded.

Training twice from the same ordered corpus and settings must produce identical
serialized tokenizer bytes and hashes. Round-trip tests cover ASCII, Unicode,
whitespace, and previously unseen valid UTF-8 text.

### Test fixture

The byte tokenizer has a fixed mapping of byte values plus special tokens,
performs no downloads, and is injectable only through the generic tokenizer
protocol in tests. It is never the default trainer tokenizer and is not exposed
by production configuration.

## Data flow and deterministic ordering

Text is encoded once into a one-dimensional integer token stream. A
`TokenBlockDataset` exposes `(input_ids, target_ids)` windows of fixed sequence
length, with targets shifted by one token.

`DeterministicBatchSampler` owns a dedicated `torch.Generator`, produces an
epoch permutation from a declared seed, and records epoch, permutation, and
cursor. Checkpoints serialize that sampler state. Loading a checkpoint restores
the next batch rather than restarting an epoch. Deterministic mode uses a
single-process loader and deterministic PyTorch algorithms; unsupported
nondeterministic operations fail rather than silently continuing.

## Dense Transformer control

The decoder-only model uses only Tier A control components relevant before
recurrence:

- token embeddings and tied LM head by default;
- pre-normalized residual blocks;
- RMSNorm;
- rotary position embeddings;
- grouped-query causal self-attention;
- optional QK normalization as an explicit configuration switch;
- SwiGLU feed-forward layers;
- PyTorch scaled-dot-product attention with an explicit causal contract;
- AdamW, gradient clipping, and configurable dropout.

The implementation contains no shared recurrent layers, prelude/core/coda
partition, scratch tokens, memory, controller, or dynamic depth. Those belong
to Task 05 and later.

The model reports total and trainable parameters. For this dense control,
active parameters equal trainable parameters. FLOPs are an explicitly labeled
estimate based on projection, attention, MLP, and vocabulary-head matrix
multiplications; training FLOPs are estimated as three forward-equivalent
passes. The formula and assumptions are tested.

## Training, evaluation, and generation

The phase uses a single-process, single-device trainer. Device selection is
`cpu`, `cuda`, or `auto`. CPU debug runs use fp32. CUDA may use bf16 when
supported; fp16 uses a gradient scaler. Unsupported precision/device
combinations fail validation.

Each step performs causal loss, finite-value checks, backward propagation,
gradient clipping, optimizer update, scheduler update, and structured metric
logging. A NaN/Inf in loss, gradients, or key metrics aborts the run with a
clear error. Evaluation runs without gradients and reports token-weighted mean
loss. Generation supports deterministic greedy decoding and seeded sampling.

The tiny CLI creates a local corpus, trains a BPE tokenizer, constructs a tiny
model, deliberately overfits a repeated short pattern, evaluates it, saves and
reloads a checkpoint, generates text, and writes a result artifact. It performs
no network access.

## Checkpoint and reproducibility contract

Checkpoints are written atomically and contain:

- model state;
- optimizer state;
- scheduler state;
- gradient-scaler state when present;
- Python, NumPy, CPU Torch, and CUDA RNG states;
- deterministic sampler state and next-batch cursor;
- completed training step and token count;
- resolved config and config hash;
- tokenizer identity and hash;
- experiment ID and metadata.

A numerical replay test compares an uninterrupted run with a save/resume run
from identical initialization. CPU fp32 must match exactly for losses and model
parameters. CUDA/bf16 tolerances will be explicit when CUDA validation becomes
available; this CPU-only host cannot provide evidence for CUDA determinism.

## Experiment artifacts

Run IDs match `EXP-NNNN-slug`. Creating an existing run directory fails. A run
contains:

```text
runs/EXP-0001-dense-sanity/
  config.yaml
  metadata.json
  tokenizer/
  metrics.jsonl
  checkpoints/
  result.json
```

Metadata records commit hash, config hash, seed, model/active parameter counts,
dataset and tokenizer hashes, optimizer and scheduler configuration, precision,
hardware/software environment, training tokens, wall time, FLOPs estimate,
peak VRAM, validation loss, and generation output. Unknown values remain null
or explicitly unavailable; no metric is fabricated.

`runs/`, datasets, checkpoints, secrets, and heavyweight artifacts are ignored
by Git. Small machine-readable test fixtures remain tracked.

## Errors and invariants

- Configuration errors fail before allocating the model.
- Unknown tokenizer or config fields fail closed.
- A tokenizer/corpus/config hash mismatch blocks resume.
- Existing experiment directories are never overwritten.
- Non-finite values stop training immediately.
- Sequence length cannot exceed the configured positional limit.
- Invalid head/KV-head divisibility is rejected.
- Empty or undersized corpora produce actionable errors.

## Verification strategy

Unit tests cover config round trips and validation, both tokenizer round trips,
BPE determinism and artifact hashing, dataset shifting, sampler determinism and
resume, RMSNorm, RoPE, GQA shapes, causal masking, SwiGLU, loss, generation, and
experiment-directory immutability.

Numerical tests cover finite forward/backward values, expected gradients,
attention causality under future-token perturbation, checkpoint round-trip
equivalence, uninterrupted-versus-resumed CPU replay, and mixed-precision
tolerance when supported.

One integration test executes the complete tiny pipeline with the test-only
byte tokenizer. Separately, the user-facing tiny CLI exercises the production
BPE path. Completion requires formatter, linter, type checker, full pytest
suite, and an actual tiny overfit run whose initial and final losses are
recorded.

## Research records

`docs/DECISIONS.md` records the control-architecture and tokenizer decisions,
including hypotheses, controls, metrics, and result status. `docs/EXPERIMENTS.md`
records commands and actual outcomes. `docs/PROGRESS.md` lists implemented
scope and open work. `docs/REFERENCES.md` lists the primary sources needed for
the dense control. `docs/PAPER_NOTES.md` clearly labels this phase as foundation
work and makes no AURORA performance claim.
