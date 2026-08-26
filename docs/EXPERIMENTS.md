# Experiment Log

Experiment directories are immutable. Values below are copied from the actual
machine-readable artifacts; unavailable measurements are marked unavailable.

## EXP-0001-dense-sanity

- **Purpose:** cheap production-BPE dense-control sanity/overfit run.
- **Command:**
  `aurora-tiny --config configs/debug/dense_tiny.yaml --runs-dir runs`
- **Git commit:** `cbeb5b2ad078605b806b0d0b9979634b7ddf53f7`
- **Config:** `configs/debug/dense_tiny.yaml`
- **Config SHA-256:**
  `11ecd60372065e305d1c6bc32b88c302bb6b3cbbea016d7b8cd88769373be42b`
- **Seed:** 17
- **Model parameters:** 65,328 total / 65,328 trainable / 65,328 active
- **Dataset version:** `generated-tiny-corpus-v1`
- **Dataset SHA-256:**
  `fc22f720158bfc1d253d69bccbb16e0aeaacd03bd233d78fde92e69c0854fd44`
- **Tokenizer:** Hugging Face BPE, `tokenizers` 0.22.2, vocab 300
- **Tokenizer SHA-256:**
  `2a616f264e588f86c1bad49e2ff2728125b115991b5e1329bf5bb23fc3e390f5`
- **Tokenizer corpus-manifest SHA-256:**
  `5b228df44f50376eb7fab14b3b47182848b0a0b83da7bcac6341a96e33161cc8`
- **Optimizer:** AdamW, learning rate 0.003, betas (0.9, 0.95), weight decay 0
- **Schedule:** 3-step warmup followed by cosine decay to 0.1× base LR
- **Precision/device:** fp32 / CPU
- **Hardware:** CPU, 16 logical cores, PyTorch capability AVX2
- **Software:** Python 3.10.11, PyTorch 2.5.1+cpu, NumPy 2.2.6
- **Training steps/tokens:** 40 / 5,120
- **Wall-clock training time:** 0.3696775000 seconds
- **Estimated training FLOPs:** 2,188,247,040; matmul-only formula documented
  in `src/aurora/training/metrics.py`
- **Peak VRAM:** unavailable/not applicable on CPU
- **Initial training loss:** 5.7461218834
- **Final training loss:** 3.8214008808
- **Evaluation loss:** 3.7949758768
- **Aggregate throughput:** 13,849.9097 tokens/second on the recorded host
- **Checkpoint:** `runs/EXP-0001-dense-sanity/checkpoints/final.pt`
- **Checkpoint reload:** succeeded into a freshly initialized model
- **Deterministic replay:** exact model state and evaluation loss verified
- **Generation:** `"AURORA learns            "`

### Interpretation

The loss decrease is evidence that the tiny dense control can learn the
deliberately repeated corpus. The evaluation text comes from the same generated
pattern family and is not an independent language-quality benchmark. The
whitespace-heavy generation is retained as an honest measured output, not
presented as a qualitative success.

### Hypothesis status

The Tasks 01–04 foundation acceptance hypothesis survives: the control trains,
evaluates, checkpoints, reloads, generates, and records reproducibility
metadata. No claim about recurrent reasoning, persistent memory, adaptive
compute, general language quality, or AURORA performance is supported.
