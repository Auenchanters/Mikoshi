# AURORA Research Engineering

A correctness-first experimental foundation for testing the AURORA research
program described in [`plan.md`](plan.md).

This repository currently implements only Tasks 01–04:

- strict, hashable YAML configuration and immutable experiment directories;
- locally trained Hugging Face BPE tokenization with recorded preprocessing and
  artifact hashes;
- deterministic, checkpointable token-block ordering;
- a modern dense causal Transformer control using RMSNorm, RoPE, grouped-query
  attention, optional QK normalization, and SwiGLU;
- single-device training, evaluation, generation, instrumentation, and complete
  checkpoint resume.

It does **not** yet implement the recurrent AURORA architecture, scratch state,
neural memory, learned halting, or Experience-Adaptive Compute.

## Install

Python 3.10 or newer is required.

```bash
python -m pip install -e ".[dev]"
```

## Reproduce the tiny dense-control run

This command performs no downloads. It creates a local corpus, trains the
production BPE tokenizer, trains and evaluates a tiny dense model, saves and
reloads a complete checkpoint, generates text, and writes machine-readable
artifacts.

```bash
aurora-tiny --config configs/debug/dense_tiny.yaml --runs-dir runs
```

Run directories are immutable. Change the `experiment.run_id` in a copied
configuration before starting another run; the command will not overwrite an
existing experiment.

The verified `EXP-0001-dense-sanity` run from commit `cbeb5b2` produced:

| Measurement | Value |
|---|---:|
| Parameters / active parameters | 65,328 / 65,328 |
| Training tokens | 5,120 |
| Initial training loss | 5.7461218834 |
| Final training loss | 3.8214008808 |
| Evaluation loss | 3.7949758768 |
| Checkpoint replay | exact on CPU fp32 |
| Estimated training FLOPs | 2,188,247,040 |
| Wall time on the recorded host | 0.3696775 s |

This deliberately tiny repeated-corpus result proves that the control can
learn and that its infrastructure works. It is not evidence of language-model
quality or of the AURORA hypothesis.

## Tokenizer policy

The production path trains a BPE tokenizer using Hugging Face `tokenizers`.
The fixed policy for this phase is:

- strict UTF-8 input and canonical LF line endings;
- Unicode NFKC normalization;
- preserved case and whitespace;
- ByteLevel pre-tokenization with no artificial prefix space;
- ByteLevel decoding and a complete byte alphabet;
- `<pad>` 0, `<bos>` 1, `<eos>` 2, and `<unk>` 3;
- configurable vocabulary size and minimum frequency.

Each artifact records the tokenizer library version, settings, ordered corpus
manifest hash, and exact `tokenizer.json` SHA-256. The deterministic byte
tokenizer exists only in `tests/fixtures/byte_tokenizer.py`; production
configuration rejects `kind: byte`.

## Quality gates

```bash
python -m ruff format --check .
python -m ruff check .
python -m mypy --no-incremental src
python -m pytest -q
```

The suite covers configuration, tokenization, deterministic sampling,
mathematical modules, causal masking, finite forward/backward behavior, loss,
generation, experiment immutability, full checkpoint state, exact CPU resume,
NaN fail-fast behavior, and the end-to-end pipeline.

## Run artifacts

```text
runs/EXP-0001-dense-sanity/
├── config.yaml
├── metadata.json
├── metrics.jsonl
├── result.json
├── training_corpus.txt
├── checkpoints/final.pt
└── tokenizer/
    ├── metadata.json
    └── tokenizer.json
```

`runs/`, datasets, and heavyweight checkpoints are Git-ignored. The exact
scientific record for the verified run is in
[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

## Research discipline

- [`docs/DECISIONS.md`](docs/DECISIONS.md) records hypotheses and engineering
  decisions.
- [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) records measured runs.
- [`docs/PROGRESS.md`](docs/PROGRESS.md) distinguishes implemented and deferred
  work.
- [`docs/REFERENCES.md`](docs/REFERENCES.md) lists sources relevant to the dense
  control.
- [`docs/PAPER_NOTES.md`](docs/PAPER_NOTES.md) constrains claims at this phase.
