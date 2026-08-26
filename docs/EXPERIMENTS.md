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

## Task 05 fixed-loop controlled sweep

### Shared protocol

- **Git commit:** `48b157805ff423922b6ae55e1f1525d34b04dbbc`
- **Commands:** `aurora-tiny --config configs/task05/b0_dense.yaml --runs-dir runs`
  and `aurora-recurrent --config configs/task05/recurrent_{1,2,4,8}.yaml
  --runs-dir runs`
- **Seed / precision / device:** 17 / fp32 / CPU
- **Software:** Python 3.10.11, PyTorch 2.5.1+cpu, NumPy 2.2.6,
  `tokenizers` 0.22.2
- **Dataset:** `generated-tiny-corpus-v1`, SHA-256
  `fc22f720158bfc1d253d69bccbb16e0aeaacd03bd233d78fde92e69c0854fd44`
- **Tokenizer:** production BPE, vocabulary 300, SHA-256
  `2a616f264e588f86c1bad49e2ff2728125b115991b5e1329bf5bb23fc3e390f5`
- **Training:** 40 steps, 5,120 tokens, AdamW at 0.003, 3-step warmup then
  cosine decay, identical batch/context/data order
- **Peak memory:** peak VRAM unavailable/not applicable for every CPU run
- **Checkpoint:** every run reloaded into a freshly initialized model and
  reproduced model state and evaluation loss exactly
- **Generation:** every run produced `"AURORA learns            "`; this
  whitespace-heavy output is not a qualitative success

### Loss and parameter outcomes

| Run | Config SHA-256 | Iterations | Parameters | Initial loss | Final loss | Evaluation loss |
|---|---|---:|---:|---:|---:|---:|
| `EXP-0500-b0-control` | `3de9739f0f761f0da255403132d546434411c5038316ec09bdbb378bc0b2737b` | B0 | 65,328 | 5.7461218834 | 3.8214008808 | 3.7949758768 |
| `EXP-0501-recurrent-1` | `60622bea7ef7b2e0b4a752780f3d7ed96c1fc0760e59921d21bdd63883176525` | 1 | 91,152 | 5.7524333000 | 3.7462275028 | 3.7306981683 |
| `EXP-0502-recurrent-2` | `c9c6da105fef438d7b878dd9e79504205b170851f91fff6797a9e7443931899a` | 2 | 91,152 | 5.7602577209 | 3.8636636734 | 3.8169981837 |
| `EXP-0504-recurrent-4` | `efea67089223cbc0a2fa3417fff8618756a99d9a5f83893c557b19be58033b97` | 4 | 91,152 | 5.7275633812 | 4.0923538208 | 4.0504400730 |
| `EXP-0508-recurrent-8` | `80ded0998e6b133fd940ecceda595dbb02bd0fa45557b5179d3d911ea5a78fcc` | 8 | 91,152 | 5.6968469620 | 4.2436923981 | 4.1997586489 |

### Compute and system outcomes

Active FLOPs/token are forward-pass matmul estimates. B0's value is derived
from its recorded training FLOPs using the same three-forward-equivalent
training convention. Loss decrease per GFLOP is descriptive and is not an
accuracy or efficiency claim.

| Run | Active FLOPs/token | Training GFLOPs | Loss drop/GFLOP | Tokens/s | Wall time (s) | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 142,464 | 2.18824704 | 0.8795720809 | 9,623.0165 | 0.5320577 | unavailable |
| R=1 | 199,296 | 3.06118656 | 0.6553686807 | 7,158.6385 | 0.7152198 | unavailable |
| R=2 | 256,128 | 3.93412608 | 0.4820877646 | 5,848.0755 | 0.8755017 | unavailable |
| R=4 | 369,792 | 5.68000512 | 0.2878887476 | 4,230.7675 | 1.2101823 | unavailable |
| R=8 | 597,120 | 9.17176320 | 0.1584378633 | 2,971.3817 | 1.7231041 | unavailable |

### Gradient outcomes

Whole-model gradient norms are pre-clipping norms. Core statistics are from the
final step before clipping.

| Run | Final model grad norm | Mean model grad norm | Final core L2 | Core mean abs | Core max abs | Core nonzero fraction |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 0.9220585227 | 1.4551847756 | not applicable | not applicable | not applicable | not applicable |
| R=1 | 1.0874402523 | 1.3304632187 | 0.3246398753 | 0.0008624807 | 0.0735349655 | 0.9869888476 |
| R=2 | 1.1062213182 | 1.4480993435 | 0.4703405840 | 0.0013576653 | 0.0657748431 | 0.9888475836 |
| R=4 | 0.9175071716 | 1.5674355939 | 0.4577092582 | 0.0013645177 | 0.0423749387 | 0.9925650558 |
| R=8 | 0.7632266879 | 2.7799535826 | 0.3368274126 | 0.0009412797 | 0.0302075669 | 1.0000000000 |

Every recurrent core had finite gradients for all 25,824 trainable core
elements at the final step.

### Final-step recurrent-state diagnostics

Norms are RMS values. Cosine is the mean per-example cosine similarity between
successive flattened states.

| Run | Iteration | Hidden-state RMS | Update RMS | Successive-state cosine |
|---|---:|---:|---:|---:|
| R=1 | 1 | 0.1055833250 | 0.0423967540 | 0.9288075566 |
| R=2 | 1 | 0.1063502580 | 0.0392616801 | 0.9389828444 |
| R=2 | 2 | 0.1361574531 | 0.0451650880 | 0.9599708319 |
| R=4 | 1 | 0.1614000350 | 0.0647321716 | 0.9283083677 |
| R=4 | 2 | 0.2128396183 | 0.0734026954 | 0.9595167637 |
| R=4 | 3 | 0.2714299262 | 0.0721892118 | 0.9841122627 |
| R=4 | 4 | 0.3238864541 | 0.0624082386 | 0.9936016798 |
| R=8 | 1 | 0.1714783907 | 0.0795602128 | 0.9037932754 |
| R=8 | 2 | 0.2587973773 | 0.1152451262 | 0.9334675074 |
| R=8 | 3 | 0.3744523227 | 0.1326087862 | 0.9768995047 |
| R=8 | 4 | 0.4958645105 | 0.1318707019 | 0.9926722040 |
| R=8 | 5 | 0.6233661771 | 0.1360260695 | 0.9963927865 |
| R=8 | 6 | 0.7528877258 | 0.1365506798 | 0.9980368614 |
| R=8 | 7 | 0.8819421530 | 0.1353005916 | 0.9987708926 |
| R=8 | 8 | 1.0111789703 | 0.1345700920 | 0.9992300272 |

### Interpretation and limitations

R=1 has the lowest raw evaluation loss, but it also has 39.53% more trainable
parameters and 39.89% more active FLOPs/token than B0. It is therefore not an
improvement claim. R=2/4/8 used progressively more active compute while their
evaluation losses worsened in this run. The increasing hidden-state RMS and
cosine values at R=8 show growing state magnitude with increasingly aligned
successive updates; this is a diagnostic observation, not proof of instability,
convergence, or overthinking.

The sweep has one seed, a 65K–91K parameter scale, only 5,120 training tokens,
a repeated generated corpus, evaluation from the same pattern family, and no
parameter- or FLOP-matched recurrent control. It supports Task 05 engineering
validity only. It does not show that recurrence improves language modeling or
that AURORA works.
