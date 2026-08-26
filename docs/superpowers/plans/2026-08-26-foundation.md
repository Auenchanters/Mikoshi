# AURORA Tasks 01–04 Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible dense causal-language-model control with trainable BPE tokenization, deterministic data ordering, complete checkpoint resume, and a cheap end-to-end experiment.

**Architecture:** Use a dependency-light `src`-layout Python package with frozen YAML-backed dataclass configuration, Hugging Face `tokenizers`, explicit PyTorch model modules, and a single-device trainer. Keep tokenizer, data, model, and training boundaries injectable so Task 05 can replace the dense backbone without changing experiment infrastructure.

**Tech Stack:** Python 3.10+, PyTorch 2.5+, Hugging Face `tokenizers`, PyYAML, NumPy, pytest, Ruff, and mypy.

**Spec:** `docs/superpowers/specs/2026-08-26-foundation-design.md`

## Global Constraints

- Implement only `plan.md` Tasks 01–04; recurrence, neural memory, EAC, learned halting, custom kernels, distributed training, and large runs are excluded.
- The production tokenizer is a locally trained BPE tokenizer; the deterministic byte tokenizer exists only under `tests/fixtures`.
- Production BPE preprocessing is UTF-8 strict, canonical LF, Unicode NFKC, case-preserving, whitespace-preserving, ByteLevel pre-tokenization with no prefix space, and ByteLevel decoding.
- Special-token IDs are `<pad>` 0, `<bos>` 1, `<eos>` 2, and `<unk>` 3.
- Experiment IDs match `EXP-NNNN-slug`; existing run directories are never overwritten.
- CPU debug runs use fp32; CUDA may use bf16 or fp16, and fp16 scaler state is checkpointed.
- Every trajectory-affecting source of randomness and data position is explicit and resumable.
- Non-finite losses or gradients abort immediately.
- The user-owned uncommitted `AGENTS.md` change must not be overwritten or silently staged.

---

## File map

```text
pyproject.toml                         packaging and tool configuration
.gitignore                             generated/data/checkpoint exclusions
configs/debug/dense_tiny.yaml          cheap BPE end-to-end run
configs/debug/dense_overfit.yaml       tiny loss-decrease run
src/aurora/config.py                   frozen schemas, YAML IO, validation, hash
src/aurora/experiment.py               immutable run creation and artifacts
src/aurora/tokenization/base.py        tokenizer protocol and identity
src/aurora/tokenization/bpe.py         BPE train/load/save implementation
src/aurora/data/dataset.py             shifted fixed-length token blocks
src/aurora/data/sampler.py             deterministic resumable batch order
src/aurora/model/norms.py              RMSNorm
src/aurora/model/rope.py               rotary embedding cache and application
src/aurora/model/attention.py          causal GQA self-attention
src/aurora/model/mlp.py                SwiGLU
src/aurora/model/transformer.py        dense decoder, loss, generation
src/aurora/training/reproducibility.py seed and RNG state helpers
src/aurora/training/metrics.py         parameter, FLOP, throughput, VRAM metrics
src/aurora/training/checkpoint.py      atomic complete-state save/load
src/aurora/training/trainer.py         train/evaluate/resume pipeline
src/aurora/cli/tiny.py                 production-BPE tiny experiment command
tests/fixtures/byte_tokenizer.py       deterministic byte tokenizer test fixture
tests/unit/                            component tests
tests/numerical/                       causality, finite-gradient, replay tests
tests/integration/                     complete cheap pipeline test
```

### Task 1: Package, configuration, and immutable experiment records

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `configs/debug/dense_tiny.yaml`
- Create: `configs/debug/dense_overfit.yaml`
- Create: `src/aurora/__init__.py`
- Create: `src/aurora/config.py`
- Create: `src/aurora/experiment.py`
- Test: `tests/unit/test_config.py`
- Test: `tests/unit/test_experiment.py`

**Interfaces:**
- Produces: `ExperimentConfig.from_yaml(path: Path) -> ExperimentConfig`
- Produces: `ExperimentConfig.to_dict() -> dict[str, object]`
- Produces: `ExperimentConfig.sha256() -> str`
- Produces: `create_run(config: ExperimentConfig, root: Path) -> RunArtifacts`
- Produces: `append_jsonl(path: Path, payload: Mapping[str, object]) -> None`

- [ ] **Step 1: Write failing configuration tests**

```python
def test_config_yaml_round_trip_and_hash(tmp_path: Path) -> None:
    config = tiny_config()
    path = tmp_path / "config.yaml"
    config.to_yaml(path)
    loaded = ExperimentConfig.from_yaml(path)
    assert loaded == config
    assert loaded.sha256() == config.sha256()

def test_unknown_config_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("experiment:\n  run_id: EXP-0001-x\nunknown: true\n")
    with pytest.raises(ConfigError, match="unknown"):
        ExperimentConfig.from_yaml(path)

def test_invalid_gqa_shape_is_rejected() -> None:
    with pytest.raises(ConfigError, match="num_heads"):
        tiny_config(model={"num_heads": 3, "num_kv_heads": 2}).validate()
```

- [ ] **Step 2: Run the focused tests and confirm import/schema failures**

Run: `python -m pytest tests/unit/test_config.py -q`

Expected: collection fails because `aurora.config` does not exist.

- [ ] **Step 3: Implement frozen config schemas and canonical hashing**

```python
@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    max_seq_len: int
    d_model: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    d_ff: int
    dropout: float = 0.0
    rope_base: float = 10_000.0
    qk_norm: bool = True
    tie_embeddings: bool = True

@dataclass(frozen=True)
class ExperimentConfig:
    experiment: ExperimentIdentityConfig
    tokenizer: TokenizerConfig
    data: DataConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    runtime: RuntimeConfig
    checkpoint: CheckpointConfig

    def sha256(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
```

Use strict recursive dataclass construction: each mapping key must match a
declared field, missing required fields list their full dotted path, and
`validate()` checks run-ID syntax, positive dimensions, `d_model % num_heads ==
0`, `num_heads % num_kv_heads == 0`, precision/device combinations, scheduler
bounds, and BPE vocabulary minimum 260.

- [ ] **Step 4: Write failing immutable-run tests**

```python
def test_create_run_writes_resolved_config_and_refuses_overwrite(tmp_path: Path) -> None:
    config = tiny_config()
    run = create_run(config, tmp_path)
    assert run.config_path.exists()
    assert yaml.safe_load(run.config_path.read_text())["config_hash"] == config.sha256()
    with pytest.raises(FileExistsError):
        create_run(config, tmp_path)
```

- [ ] **Step 5: Implement run artifacts and append-only JSONL metrics**

```python
@dataclass(frozen=True)
class RunArtifacts:
    root: Path
    config_path: Path
    metadata_path: Path
    metrics_path: Path
    checkpoints_dir: Path
    tokenizer_dir: Path
    result_path: Path

def append_jsonl(path: Path, payload: Mapping[str, object]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True) + "\n")
```

- [ ] **Step 6: Run Task 1 tests, Ruff, and mypy**

Run: `python -m pytest tests/unit/test_config.py tests/unit/test_experiment.py -q`

Run: `python -m ruff check src/aurora/config.py src/aurora/experiment.py tests/unit/test_config.py tests/unit/test_experiment.py`

Run: `python -m mypy src/aurora/config.py src/aurora/experiment.py`

Expected: all commands exit zero.

- [ ] **Step 7: Commit Task 1**

```bash
git add pyproject.toml .gitignore configs src/aurora/__init__.py src/aurora/config.py src/aurora/experiment.py tests/unit/test_config.py tests/unit/test_experiment.py
git commit -m "feat: add reproducible project configuration"
```

### Task 2: Trainable BPE tokenizer and test-only byte fixture

**Files:**
- Create: `src/aurora/tokenization/__init__.py`
- Create: `src/aurora/tokenization/base.py`
- Create: `src/aurora/tokenization/bpe.py`
- Create: `tests/fixtures/__init__.py`
- Create: `tests/fixtures/byte_tokenizer.py`
- Test: `tests/unit/test_tokenization.py`

**Interfaces:**
- Produces: `TokenizerProtocol.encode(text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]`
- Produces: `TokenizerProtocol.decode(ids: Sequence[int], *, skip_special_tokens: bool = True) -> str`
- Produces: `TokenizerIdentity(kind: str, version: str, sha256: str, settings: Mapping[str, object])`
- Produces: `train_bpe(corpus_paths: Sequence[Path], output_dir: Path, config: TokenizerConfig) -> BPETokenizer`
- Produces: `BPETokenizer.load(path: Path) -> BPETokenizer`

- [ ] **Step 1: Write failing tokenizer tests**

```python
@pytest.mark.parametrize("text", ["hello", " café\n", "नमस्ते", "a\t b"])
def test_byte_fixture_round_trip(text: str) -> None:
    tokenizer = ByteTokenizer()
    assert tokenizer.decode(tokenizer.encode(text)) == text

def test_bpe_training_is_byte_identical(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("alpha beta\nalpha gamma\nβeta alpha\n", encoding="utf-8")
    first = train_bpe([corpus], tmp_path / "first", bpe_config(vocab_size=300))
    second = train_bpe([corpus], tmp_path / "second", bpe_config(vocab_size=300))
    assert first.identity.sha256 == second.identity.sha256
    assert (tmp_path / "first/tokenizer.json").read_bytes() == (
        tmp_path / "second/tokenizer.json"
    ).read_bytes()

def test_bpe_round_trip_unseen_utf8(bpe_tokenizer: BPETokenizer) -> None:
    text = "Unseen 🚀 text\nwith  spaces"
    assert bpe_tokenizer.decode(bpe_tokenizer.encode(text)) == text
```

- [ ] **Step 2: Run tests and confirm missing tokenizer imports**

Run: `python -m pytest tests/unit/test_tokenization.py -q`

Expected: collection fails because tokenizer modules do not exist.

- [ ] **Step 3: Implement protocol, identity, and byte fixture**

```python
class TokenizerProtocol(Protocol):
    @property
    def vocab_size(self) -> int: ...
    @property
    def pad_id(self) -> int: ...
    @property
    def bos_id(self) -> int: ...
    @property
    def eos_id(self) -> int: ...
    @property
    def identity(self) -> TokenizerIdentity: ...
    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]: ...
    def decode(self, ids: Sequence[int], *, skip_special_tokens: bool = True) -> str: ...
```

The test fixture maps special IDs 0–3 and byte value `b` to ID `b + 4`; its
identity hash is derived from that fixed specification.

- [ ] **Step 4: Implement deterministic BPE artifact creation**

```python
tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
tokenizer.normalizer = normalizers.NFKC()
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()
trainer = trainers.BpeTrainer(
    vocab_size=config.vocab_size,
    min_frequency=config.min_frequency,
    special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    show_progress=False,
)
```

Normalize source line endings into deterministic temporary training files in
manifest order, save `tokenizer.json` and `metadata.json`, hash the exact
tokenizer bytes, and verify special-token IDs before returning.

- [ ] **Step 5: Run tokenizer tests and static checks**

Run: `python -m pytest tests/unit/test_tokenization.py -q`

Run: `python -m ruff check src/aurora/tokenization tests/fixtures tests/unit/test_tokenization.py`

Run: `python -m mypy src/aurora/tokenization`

Expected: all commands exit zero.

- [ ] **Step 6: Commit Task 2**

```bash
git add src/aurora/tokenization tests/fixtures tests/unit/test_tokenization.py
git commit -m "feat: add reproducible BPE tokenization"
```

### Task 3: Deterministic token blocks and resumable batch order

**Files:**
- Create: `src/aurora/data/__init__.py`
- Create: `src/aurora/data/dataset.py`
- Create: `src/aurora/data/sampler.py`
- Test: `tests/unit/test_data.py`

**Interfaces:**
- Produces: `TokenBlockDataset(tokens: Sequence[int] | Tensor, sequence_length: int, stride: int | None = None)`
- Produces: `DeterministicBatchSampler(dataset_size: int, batch_size: int, seed: int, drop_last: bool = True)`
- Produces: `DeterministicBatchSampler.state_dict() -> dict[str, object]`
- Consumes: sampler state through `load_state_dict(state: Mapping[str, object]) -> None`

- [ ] **Step 1: Write failing dataset and sampler tests**

```python
def test_token_blocks_shift_targets() -> None:
    dataset = TokenBlockDataset(list(range(10)), sequence_length=4, stride=4)
    inputs, targets = dataset[1]
    assert inputs.tolist() == [4, 5, 6, 7]
    assert targets.tolist() == [5, 6, 7, 8]

def test_sampler_resume_returns_exact_next_batches() -> None:
    sampler = DeterministicBatchSampler(12, batch_size=3, seed=7)
    iterator = iter(sampler)
    consumed = [next(iterator), next(iterator)]
    state = sampler.state_dict()
    expected_tail = list(iterator)
    resumed = DeterministicBatchSampler(12, batch_size=3, seed=7)
    resumed.load_state_dict(state)
    assert consumed
    assert list(resumed) == expected_tail
```

- [ ] **Step 2: Run tests and confirm missing data modules**

Run: `python -m pytest tests/unit/test_data.py -q`

Expected: collection fails because `aurora.data` does not exist.

- [ ] **Step 3: Implement shifted blocks and dedicated-generator sampler**

```python
def _permutation(self, epoch: int) -> list[int]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(self.seed + epoch)
    return torch.randperm(self.dataset_size, generator=generator).tolist()

def state_dict(self) -> dict[str, object]:
    return {"epoch": self.epoch, "cursor": self.cursor, "order": list(self.order)}
```

Advance `cursor` before yielding each batch so a checkpoint taken after a
completed optimizer step resumes at the following batch. Validate all restored
indices and dimensions.

- [ ] **Step 4: Run data tests and static checks**

Run: `python -m pytest tests/unit/test_data.py -q`

Run: `python -m ruff check src/aurora/data tests/unit/test_data.py`

Run: `python -m mypy src/aurora/data`

Expected: all commands exit zero.

- [ ] **Step 5: Commit Task 3**

```bash
git add src/aurora/data tests/unit/test_data.py
git commit -m "feat: add deterministic resumable data order"
```

### Task 4: Dense Transformer mathematical modules

**Files:**
- Create: `src/aurora/model/__init__.py`
- Create: `src/aurora/model/norms.py`
- Create: `src/aurora/model/rope.py`
- Create: `src/aurora/model/attention.py`
- Create: `src/aurora/model/mlp.py`
- Test: `tests/unit/test_model_components.py`
- Test: `tests/numerical/test_attention_causality.py`

**Interfaces:**
- Produces: `RMSNorm(d_model: int, eps: float = 1e-6)`
- Produces: `RotaryEmbedding(head_dim: int, max_seq_len: int, base: float)`
- Produces: `GroupedQueryAttention(config: ModelConfig)`
- Produces: `SwiGLU(d_model: int, d_ff: int, dropout: float)`

- [ ] **Step 1: Write failing component shape and normalization tests**

```python
def test_rms_norm_matches_reference() -> None:
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    norm = RMSNorm(4, eps=1e-6)
    expected = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(norm(x), expected)

def test_gqa_shape_and_gradients() -> None:
    module = GroupedQueryAttention(model_config())
    x = torch.randn(2, 8, 32, requires_grad=True)
    output = module(x)
    assert output.shape == x.shape
    output.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
```

- [ ] **Step 2: Run component tests and confirm missing model modules**

Run: `python -m pytest tests/unit/test_model_components.py -q`

Expected: collection fails because mathematical modules do not exist.

- [ ] **Step 3: Implement RMSNorm, RoPE, GQA, and SwiGLU**

```python
q = self.q_proj(x).view(batch, time, self.num_heads, self.head_dim).transpose(1, 2)
k = self.k_proj(x).view(batch, time, self.num_kv_heads, self.head_dim).transpose(1, 2)
v = self.v_proj(x).view(batch, time, self.num_kv_heads, self.head_dim).transpose(1, 2)
q, k = self.rope(q, k, start_pos=0)
if self.qk_norm:
    q = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + self.norm_eps)
    k = k * torch.rsqrt(k.pow(2).mean(-1, keepdim=True) + self.norm_eps)
k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
attended = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout, is_causal=True)
```

RoPE rotates adjacent half-dimensions in fp32 and casts back to the input dtype.
Reject odd head dimensions and sequences beyond `max_seq_len`.

- [ ] **Step 4: Write and run a future-token perturbation test**

```python
def test_attention_is_causal_under_future_perturbation() -> None:
    torch.manual_seed(0)
    module = GroupedQueryAttention(model_config(dropout=0.0)).eval()
    original = torch.randn(1, 6, 32)
    changed = original.clone()
    changed[:, 4:] = torch.randn_like(changed[:, 4:]) * 100
    torch.testing.assert_close(module(original)[:, :4], module(changed)[:, :4])
```

Run: `python -m pytest tests/numerical/test_attention_causality.py -q`

Expected: pass with exact causal independence within PyTorch tolerance.

- [ ] **Step 5: Run Task 4 tests and static checks**

Run: `python -m pytest tests/unit/test_model_components.py tests/numerical/test_attention_causality.py -q`

Run: `python -m ruff check src/aurora/model tests/unit/test_model_components.py tests/numerical/test_attention_causality.py`

Run: `python -m mypy src/aurora/model`

Expected: all commands exit zero.

- [ ] **Step 6: Commit Task 4**

```bash
git add src/aurora/model tests/unit/test_model_components.py tests/numerical/test_attention_causality.py
git commit -m "feat: add dense Transformer components"
```

### Task 5: Dense causal LM, loss, and generation

**Files:**
- Create: `src/aurora/model/transformer.py`
- Test: `tests/unit/test_transformer.py`
- Test: `tests/numerical/test_forward_backward.py`

**Interfaces:**
- Produces: `DenseTransformer(config: ModelConfig)`
- Produces: `LMOutput(logits: Tensor, loss: Tensor | None)`
- Produces: `DenseTransformer.generate(input_ids: Tensor, max_new_tokens: int, temperature: float = 0.0, top_k: int | None = None, generator: torch.Generator | None = None) -> Tensor`

- [ ] **Step 1: Write failing language-model tests**

```python
def test_lm_loss_matches_shifted_cross_entropy() -> None:
    model = DenseTransformer(model_config())
    input_ids = torch.randint(0, 64, (2, 8))
    output = model(input_ids, targets=input_ids)
    expected = F.cross_entropy(output.logits[:, :-1].reshape(-1, 64), input_ids[:, 1:].reshape(-1))
    torch.testing.assert_close(output.loss, expected)

def test_greedy_generation_is_deterministic() -> None:
    model = DenseTransformer(model_config()).eval()
    prompt = torch.tensor([[1, 5, 6]])
    assert torch.equal(model.generate(prompt, 4), model.generate(prompt, 4))
```

- [ ] **Step 2: Run tests and confirm missing transformer module**

Run: `python -m pytest tests/unit/test_transformer.py -q`

Expected: collection fails because `aurora.model.transformer` does not exist.

- [ ] **Step 3: Implement decoder blocks, initialization, loss, and generation**

```python
class DecoderBlock(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.attention_norm(x))
        return x + self.mlp(self.mlp_norm(x))

def forward(self, input_ids: Tensor, targets: Tensor | None = None) -> LMOutput:
    if input_ids.shape[1] > self.config.max_seq_len:
        raise ValueError("sequence length exceeds model.max_seq_len")
    hidden = self.token_embedding(input_ids)
    for block in self.blocks:
        hidden = block(hidden)
    logits = self.lm_head(self.final_norm(hidden))
    loss = None if targets is None else shifted_cross_entropy(logits, targets)
    return LMOutput(logits=logits, loss=loss)
```

Greedy decoding uses argmax when `temperature == 0`. Sampling divides logits by
temperature, optionally masks below top-k, and calls `torch.multinomial` with
the provided generator.

- [ ] **Step 4: Add finite forward/backward numerical test**

```python
def test_forward_backward_is_finite_and_all_trainable_parameters_receive_gradients() -> None:
    model = DenseTransformer(model_config(dropout=0.0))
    ids = torch.randint(0, 64, (2, 8))
    loss = model(ids, targets=ids).loss
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
```

- [ ] **Step 5: Run Task 5 tests and static checks**

Run: `python -m pytest tests/unit/test_transformer.py tests/numerical/test_forward_backward.py -q`

Run: `python -m ruff check src/aurora/model/transformer.py tests/unit/test_transformer.py tests/numerical/test_forward_backward.py`

Run: `python -m mypy src/aurora/model/transformer.py`

Expected: all commands exit zero.

- [ ] **Step 6: Commit Task 5**

```bash
git add src/aurora/model/transformer.py tests/unit/test_transformer.py tests/numerical/test_forward_backward.py
git commit -m "feat: add dense causal language model"
```

### Task 6: RNG state, instrumentation, and full checkpoints

**Files:**
- Create: `src/aurora/training/__init__.py`
- Create: `src/aurora/training/reproducibility.py`
- Create: `src/aurora/training/metrics.py`
- Create: `src/aurora/training/checkpoint.py`
- Test: `tests/unit/test_metrics.py`
- Test: `tests/numerical/test_checkpoint.py`

**Interfaces:**
- Produces: `seed_everything(seed: int, deterministic: bool) -> None`
- Produces: `capture_rng_state() -> dict[str, object]`
- Produces: `restore_rng_state(state: Mapping[str, object]) -> None`
- Produces: `count_parameters(model: nn.Module) -> ParameterCounts`
- Produces: `estimate_transformer_flops(config: ModelConfig, batch_size: int, sequence_length: int, training: bool) -> int`
- Produces: `save_checkpoint(path: Path, state: TrainingState) -> None`
- Produces: `load_checkpoint(path: Path, *, map_location: str | torch.device = "cpu") -> dict[str, object]`

- [ ] **Step 1: Write failing RNG, metric, and checkpoint tests**

```python
def test_rng_capture_restore_replays_python_numpy_and_torch() -> None:
    seed_everything(9, deterministic=True)
    state = capture_rng_state()
    expected = (random.random(), np.random.rand(), torch.rand(2))
    restore_rng_state(state)
    actual = (random.random(), np.random.rand(), torch.rand(2))
    assert expected[:2] == actual[:2]
    torch.testing.assert_close(expected[2], actual[2], rtol=0, atol=0)

def test_checkpoint_round_trip_restores_all_declared_keys(tmp_path: Path) -> None:
    save_checkpoint(tmp_path / "step.pt", training_state())
    payload = load_checkpoint(tmp_path / "step.pt")
    assert set(payload) == {
        "format_version", "model", "optimizer", "scheduler", "scaler", "rng",
        "sampler", "step", "tokens_seen", "config", "config_hash",
        "tokenizer", "experiment",
    }
```

- [ ] **Step 2: Run tests and confirm missing training modules**

Run: `python -m pytest tests/unit/test_metrics.py tests/numerical/test_checkpoint.py -q`

Expected: collection fails because training modules do not exist.

- [ ] **Step 3: Implement deterministic RNG capture and explicit FLOP estimates**

```python
def capture_rng_state() -> dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
```

FLOP estimation includes Q/K/V/output projections, attention score/value
products, three SwiGLU projections, and the vocabulary projection for every
layer and token. `training=True` multiplies the forward estimate by three and
the returned metric is labeled estimated.

- [ ] **Step 4: Implement atomic complete-state checkpoint IO**

```python
def save_checkpoint(path: Path, state: TrainingState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state.to_payload(), temporary)
    os.replace(temporary, path)
```

`load_checkpoint` validates `format_version == 1`, required keys, nonnegative
step/token counts, and presence of config/tokenizer hashes before returning.

- [ ] **Step 5: Run Task 6 tests and static checks**

Run: `python -m pytest tests/unit/test_metrics.py tests/numerical/test_checkpoint.py -q`

Run: `python -m ruff check src/aurora/training tests/unit/test_metrics.py tests/numerical/test_checkpoint.py`

Run: `python -m mypy src/aurora/training`

Expected: all commands exit zero.

- [ ] **Step 6: Commit Task 6**

```bash
git add src/aurora/training tests/unit/test_metrics.py tests/numerical/test_checkpoint.py
git commit -m "feat: add complete reproducible checkpoints"
```

### Task 7: Single-device trainer, evaluation, and deterministic replay

**Files:**
- Create: `src/aurora/training/trainer.py`
- Test: `tests/unit/test_trainer.py`
- Test: `tests/numerical/test_resume_replay.py`

**Interfaces:**
- Produces: `Trainer(config, model, tokenizer_identity, train_dataset, eval_dataset, run)`
- Produces: `Trainer.train(max_steps: int | None = None) -> TrainingResult`
- Produces: `Trainer.evaluate(max_batches: int | None = None) -> EvaluationResult`
- Produces: `Trainer.save(path: Path) -> None`
- Produces: `Trainer.resume(path: Path) -> None`

- [ ] **Step 1: Write failing loss-decrease and non-finite tests**

```python
def test_trainer_reduces_loss_on_repeated_pattern(tmp_path: Path) -> None:
    trainer = make_tiny_trainer(tmp_path, max_steps=30)
    result = trainer.train()
    assert result.final_loss < result.initial_loss

def test_trainer_stops_on_non_finite_loss(tmp_path: Path) -> None:
    trainer = make_tiny_trainer(tmp_path)
    trainer.model.forward = non_finite_forward
    with pytest.raises(FloatingPointError, match="non-finite loss"):
        trainer.train(max_steps=1)
```

- [ ] **Step 2: Run trainer tests and confirm missing trainer**

Run: `python -m pytest tests/unit/test_trainer.py -q`

Expected: collection fails because `aurora.training.trainer` does not exist.

- [ ] **Step 3: Implement device/precision setup and finite train loop**

```python
for input_ids, targets in self._batches():
    self.optimizer.zero_grad(set_to_none=True)
    with self._autocast():
        output = self.model(input_ids, targets=targets)
    loss = require_finite_loss(output.loss)
    self.scaler.scale(loss).backward()
    self.scaler.unscale_(self.optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
    require_finite("gradient norm", grad_norm)
    self.scaler.step(self.optimizer)
    self.scaler.update()
    self.scheduler.step()
```

On CPU/fp32, use a disabled scaler with the same call surface. Log step, loss,
learning rate, gradient norm, tokens, estimated FLOPs, tokens/sec, and peak VRAM
to `metrics.jsonl`.

- [ ] **Step 4: Implement eval, save, and resume validation**

Evaluation accumulates summed cross-entropy and target-token count, returning a
token-weighted mean. Resume verifies config and tokenizer hashes before loading
model, optimizer, scheduler, scaler, sampler, counters, and RNG in that order.

- [ ] **Step 5: Write deterministic uninterrupted-versus-resume test**

```python
def test_cpu_fp32_resume_matches_uninterrupted_training(tmp_path: Path) -> None:
    full = make_tiny_trainer(tmp_path / "full", seed=17, max_steps=8)
    full_losses = full.train().losses

    split = make_tiny_trainer(tmp_path / "split", seed=17, max_steps=8)
    first_losses = split.train(max_steps=3).losses
    split.save(tmp_path / "resume.pt")
    resumed = make_tiny_trainer(tmp_path / "resumed", seed=999, max_steps=8)
    resumed.resume(tmp_path / "resume.pt")
    second_losses = resumed.train(max_steps=8).losses

    assert first_losses + second_losses == full_losses
    for left, right in zip(full.model.parameters(), resumed.model.parameters(), strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
```

- [ ] **Step 6: Run Task 7 tests and static checks**

Run: `python -m pytest tests/unit/test_trainer.py tests/numerical/test_resume_replay.py -q`

Run: `python -m ruff check src/aurora/training/trainer.py tests/unit/test_trainer.py tests/numerical/test_resume_replay.py`

Run: `python -m mypy src/aurora/training/trainer.py`

Expected: all commands exit zero.

- [ ] **Step 7: Commit Task 7**

```bash
git add src/aurora/training/trainer.py tests/unit/test_trainer.py tests/numerical/test_resume_replay.py
git commit -m "feat: add deterministic dense model trainer"
```

### Task 8: Tiny end-to-end commands, documentation, and research records

**Files:**
- Create: `src/aurora/cli/__init__.py`
- Create: `src/aurora/cli/tiny.py`
- Create: `tests/integration/test_tiny_pipeline.py`
- Modify: `README.md`
- Modify: `docs/REFERENCES.md`
- Modify: `docs/DECISIONS.md`
- Modify: `docs/EXPERIMENTS.md`
- Modify: `docs/PROGRESS.md`
- Modify: `docs/PAPER_NOTES.md`

**Interfaces:**
- Produces: console command `aurora-tiny --config PATH --runs-dir PATH`
- Produces: `run_tiny_experiment(config_path: Path, runs_dir: Path) -> dict[str, object]`
- Consumes: all Tasks 1–7 interfaces.

- [ ] **Step 1: Write failing integration test using the byte fixture**

```python
def test_tiny_pipeline_trains_evaluates_checkpoints_reloads_and_generates(tmp_path: Path) -> None:
    result = run_integration_pipeline(
        tokenizer=ByteTokenizer(),
        runs_dir=tmp_path,
        run_id="EXP-9000-integration",
        steps=8,
    )
    assert result["initial_loss"] > result["final_loss"]
    assert result["checkpoint_reloaded"] is True
    assert result["generated_text"]
    assert (tmp_path / "EXP-9000-integration/result.json").exists()
```

- [ ] **Step 2: Run integration test and confirm missing CLI/pipeline**

Run: `python -m pytest tests/integration/test_tiny_pipeline.py -q`

Expected: collection fails because the tiny pipeline does not exist.

- [ ] **Step 3: Implement production-BPE tiny experiment**

The command writes a repeated local training corpus, trains BPE into the run's
tokenizer directory, hashes the corpus and tokenizer, encodes train/eval token
streams, trains, evaluates, checkpoints, constructs a fresh model/trainer,
resumes, generates, and writes only measured values to `result.json`.

```python
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/debug/dense_tiny.yaml"))
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    args = parser.parse_args()
    result = run_tiny_experiment(args.config, args.runs_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
```

- [ ] **Step 4: Document commands and record decisions without invented results**

README commands:

```bash
python -m pip install -e ".[dev]"
aurora-tiny --config configs/debug/dense_tiny.yaml --runs-dir runs
python -m pytest
python -m ruff format --check .
python -m ruff check .
python -m mypy src
```

`docs/DECISIONS.md` records the BPE/test-fixture split and dense-only control.
`docs/EXPERIMENTS.md` initially records command/config/metrics only after the
actual command completes. `docs/PAPER_NOTES.md` states that Tasks 01–04 provide
a control and do not demonstrate the AURORA hypothesis.

- [ ] **Step 5: Run the full quality gate**

Run: `python -m ruff format --check .`

Run: `python -m ruff check .`

Run: `python -m mypy src`

Run: `python -m pytest -q`

Expected: formatter, linter, type checker, and all tests exit zero.

- [ ] **Step 6: Run the real production-BPE tiny experiment**

Run: `aurora-tiny --config configs/debug/dense_tiny.yaml --runs-dir runs`

Expected: exits zero, reports finite initial/final/evaluation loss, final loss
below initial loss, a reload confirmation, generation text, parameter counts,
estimated FLOPs, throughput, wall time, and peak VRAM availability.

- [ ] **Step 7: Copy only measured tiny-run values into experiment records**

Read `runs/<run-id>/result.json`, the final metrics line, environment metadata,
and checkpoint manifest. Record exact values and command in
`docs/EXPERIMENTS.md`; update `docs/PROGRESS.md` with verified scope.

- [ ] **Step 8: Inspect generated/tracked artifacts and complete the final commit**

Run: `git status --short`

Run: `git diff --check`

Run: `git ls-files | rg "(^runs/|\.pt$|\.bin$|tokenizer\.json$|secret|token)"`

Expected: no run directory, checkpoint, trained tokenizer, dataset, or secret is
tracked; only source, tests, configs, and documentation are staged.

```bash
git add README.md docs pyproject.toml configs src tests .gitignore
git commit -m "feat: complete dense research foundation"
```

## Final verification

- [ ] Confirm all `plan.md` Tasks 01–04 completion criteria against fresh output.
- [ ] Confirm the original user-owned `AGENTS.md` working-tree change remains intact.
- [ ] Report implemented files, every test command and count, actual loss values,
      deterministic replay evidence, deviations, open issues, and Task 05 scope.
- [ ] State explicitly that the dense control does not demonstrate that AURORA works.
