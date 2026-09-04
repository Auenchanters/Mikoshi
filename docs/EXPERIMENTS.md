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

- **B0 Git commit:** `48b157805ff423922b6ae55e1f1525d34b04dbbc`
- **Recurrent-run Git commit:** `98220bf039b5d04761c9eb01033c44640c6daf3e`
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
| R=1 | 199,296 | 3.06118656 | 0.6553686807 | 7,842.2086 | 0.6528773 | unavailable |
| R=2 | 256,128 | 3.93412608 | 0.4820877646 | 6,824.4737 | 0.7502410 | unavailable |
| R=4 | 369,792 | 5.68000512 | 0.2878887476 | 4,789.3092 | 1.0690477 | unavailable |
| R=8 | 597,120 | 9.17176320 | 0.1584378633 | 3,166.0777 | 1.6171429 | unavailable |

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

Norms are RMS values. Cosines are mean per-example similarities between
flattened states. Relative update is the exact global L2 ratio
`||h_t-h_{t-1}|| / ||h_t||`. Diagnostic collection is detached,
output-neutral, and deterministic under the recorded CPU fp32 conditions.

| Run | Iteration | Hidden-state RMS | Update RMS | Cosine to previous | Cosine to h0 | Relative update |
|---|---:|---:|---:|---:|---:|---:|
| R=1 | 1 | 0.1055833250 | 0.0423967540 | 0.9288075566 | 0.9288075566 | 0.4015478194 |
| R=2 | 1 | 0.1063502580 | 0.0392616801 | 0.9389828444 | 0.9389828444 | 0.3691733181 |
| R=2 | 2 | 0.1361574531 | 0.0451650880 | 0.9599708319 | 0.8452925682 | 0.3317122459 |
| R=4 | 1 | 0.1614000350 | 0.0647321716 | 0.9283083677 | 0.9283083677 | 0.4010666609 |
| R=4 | 2 | 0.2128396183 | 0.0734026954 | 0.9595167637 | 0.8102678061 | 0.3448732495 |
| R=4 | 3 | 0.2714299262 | 0.0721892118 | 0.9841122627 | 0.7185750008 | 0.2659589350 |
| R=4 | 4 | 0.3238864541 | 0.0624082386 | 0.9936016798 | 0.6662002206 | 0.1926855445 |
| R=8 | 1 | 0.1714783907 | 0.0795602128 | 0.9037932754 | 0.9037932754 | 0.4639664590 |
| R=8 | 2 | 0.2587973773 | 0.1152451262 | 0.9334675074 | 0.7228146791 | 0.4453102946 |
| R=8 | 3 | 0.3744523227 | 0.1326087862 | 0.9768995047 | 0.5939795971 | 0.3541406691 |
| R=8 | 4 | 0.4958645105 | 0.1318707019 | 0.9926722050 | 0.5215638876 | 0.2659409940 |
| R=8 | 5 | 0.6233661771 | 0.1360260695 | 0.9963927865 | 0.4780544043 | 0.2182121277 |
| R=8 | 6 | 0.7528877258 | 0.1365506798 | 0.9980368614 | 0.4459280074 | 0.1813692600 |
| R=8 | 7 | 0.8819421530 | 0.1353005916 | 0.9987708926 | 0.4202508628 | 0.1534121186 |
| R=8 | 8 | 1.0111789703 | 0.1345700920 | 0.9992300272 | 0.4045824409 | 0.1330823600 |

### Interpretation and limitations

R=1 has the lowest raw evaluation loss, but it also has 39.53% more trainable
parameters and 39.89% more active FLOPs/token than B0. It is therefore not an
improvement claim. Deeper naïve recurrence degraded performance in this
experiment: R=2/4/8 used progressively more active compute while evaluation
loss worsened monotonically relative to R=1. At R=8, cosine to h0 fell from
0.9038 to 0.4046 while hidden-state RMS rose from 0.1715 to 1.0112; meanwhile,
successive-state cosine approached 0.9992 and relative update fell to 0.1331.
These are diagnostic observations, not proof of instability, convergence, or
overthinking.

The sweep has one seed, a 65K–91K parameter scale, only 5,120 training tokens,
a repeated generated corpus, evaluation from the same pattern family, and no
parameter- or FLOP-matched recurrent control. It supports Task 05 engineering
validity only. It does not show that recurrence improves language modeling or
that AURORA works.

## Task 06 recurrent-stability controls

### Predeclared hypotheses

- **S0 — frozen fixed-loop control:** without a controlled update, greater
  recurrent depth was expected to reproduce the Task 05 pattern of rising
  hidden-state RMS, falling cosine to `h0`, and near-collinear successive
  states. The frozen Task 05 R=4/R=8 records above were not rerun or edited.
- **S1 — projected input anchoring:** adding one bias-free projection of the
  immutable prelude state to every core input was expected to reduce drift from
  `h0`, with a parameter and matmul cost.
- **S2 — scalar gated update:** interpolating between the prior state and core
  proposal with one learned scalar sigmoid gate per token was expected to
  reduce update magnitude and drift without collapsing the gate to zero or one.
- **S3 — initial-RMS stabilization:** parameter-free rescaling to each token's
  initial-state RMS was expected to control RMS growth directly; no raw-loss or
  directional-drift benefit was assumed.
- **S4 — anchor plus gate:** composing only S1 and S2 was expected to combine
  the anchor's directional signal with the gate's controlled interpolation.

These hypotheses were tested without post-result tuning. Anchor gating,
vector-valued or depth-specific gates, learned normalization, randomized depth,
adaptive halting, Jacobian/STARS regularization, scratch state, and persistent
memory were outside Task 06.

### Shared protocol and artifact identity

- **Commands:** `aurora-stability --config <configs/task06/*.yaml> --runs-dir
  runs`, once for each of the eight configurations. S1/S2/S3 artifacts existed
  before the S4 configurations were committed or run.
- **Git commits:** S1/S2/S3 configurations and runs at
  `ac4329b72d29cd83787d16b1c43c31392c2634c5`; S4 configurations and runs at
  `ac627b8f859b6d4373183012b8d8524512f61477`.
- **Seed / precision / device:** 17 / fp32 / CPU; deterministic execution.
- **Software:** Python 3.10.11, PyTorch 2.5.1+cpu, NumPy 2.2.6,
  `tokenizers` 0.22.2.
- **Data and training:** the frozen Task 05 production-BPE corpus, batch/context,
  order, optimizer, schedule, 40 steps, and 5,120 tokens. Dataset SHA-256 is
  `fc22f720158bfc1d253d69bccbb16e0aeaacd03bd233d78fde92e69c0854fd44`;
  tokenizer SHA-256 is
  `2a616f264e588f86c1bad49e2ff2728125b115991b5e1329bf5bb23fc3e390f5`.
- **Artifact gates:** all eight runs completed with finite diagnostics, exact
  fresh-checkpoint reload, deterministic replay, 40 metric rows, nonempty
  generation, and matching result/metadata/config/tokenizer/checkpoint hashes.
- **Peak memory:** `peak_vram_bytes` is `null` for every CPU run, so peak VRAM
  is unavailable rather than estimated.

### Complete R=4/R=8 outcome matrix

Training GFLOPs use the frozen three-forward-equivalent, matmul-only convention.
Elementwise interpolation, sigmoid, RMS rescaling, activation, normalization,
softmax, embeddings, and diagnostic reductions are excluded. Throughput is a
host measurement and is not used as a numerical claim.

| Run | Variant/depth | Parameters | Initial train loss | Final train loss | Eval loss | Active FLOPs/token | Training GFLOPs | Tokens/s | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `EXP-0504-recurrent-4` | S0/4 | 91,152 | 5.7275633812 | 4.0923538208 | 4.0504400730 | 369,792 | 5.68000512 | 4,789.3092 | unavailable |
| `EXP-0508-recurrent-8` | S0/8 | 91,152 | 5.6968469620 | 4.2436923981 | 4.1997586489 | 597,120 | 9.17176320 | 3,166.0777 | unavailable |
| `EXP-0614-anchor-r4` | S1/4 | 93,456 | 5.7259039879 | 3.9334132671 | 3.8888302743 | 388,224 | 5.96312064 | 2,681.2295 | unavailable |
| `EXP-0618-anchor-r8` | S1/8 | 93,456 | 5.6956567764 | 4.0114674568 | 3.9617625475 | 633,984 | 9.73799424 | 1,711.6084 | unavailable |
| `EXP-0624-gate-r4` | S2/4 | 91,249 | 5.7279758453 | 3.8603482246 | 3.8293781579 | 370,560 | 5.69180160 | 2,728.9802 | unavailable |
| `EXP-0628-gate-r8` | S2/8 | 91,249 | 5.7031197548 | 4.1045541763 | 4.0581546426 | 598,656 | 9.19535616 | 695.5938 | unavailable |
| `EXP-0634-initial-rms-r4` | S3/4 | 91,152 | 5.7009363174 | 4.3252892494 | 4.2306061983 | 369,792 | 5.68000512 | 1,254.1470 | unavailable |
| `EXP-0638-initial-rms-r8` | S3/8 | 91,152 | 5.6817059517 | 4.3064498901 | 4.2106192112 | 597,120 | 9.17176320 | 815.0961 | unavailable |
| `EXP-0644-anchor-gate-r4` | S4/4 | 93,553 | 5.7260551453 | 3.7805457115 | 3.7531955242 | 388,992 | 5.97491712 | 1,487.8856 | unavailable |
| `EXP-0648-anchor-gate-r8` | S4/8 | 93,553 | 5.7028470039 | 3.9312365055 | 3.9005704820 | 635,520 | 9.76158720 | 979.9423 | unavailable |

### Mechanism compute differences from depth-matched S0

| Variant/depth | Parameter delta | Active-FLOPs/token delta | Training-FLOPs delta |
|---|---:|---:|---:|
| S1/4 | +2,304 (+2.5276%) | +18,432 (+4.9844%) | +283,115,520 (+4.9844%) |
| S1/8 | +2,304 (+2.5276%) | +36,864 (+6.1736%) | +566,231,040 (+6.1736%) |
| S2/4 | +97 (+0.1064%) | +768 (+0.2077%) | +11,796,480 (+0.2077%) |
| S2/8 | +97 (+0.1064%) | +1,536 (+0.2572%) | +23,592,960 (+0.2572%) |
| S3/4 | +0 (+0.0000%) | +0 (+0.0000%) | +0 (+0.0000%) |
| S3/8 | +0 (+0.0000%) | +0 (+0.0000%) | +0 (+0.0000%) |
| S4/4 | +2,401 (+2.6341%) | +19,200 (+5.1921%) | +294,912,000 (+5.1921%) |
| S4/8 | +2,401 (+2.6341%) | +38,400 (+6.4309%) | +589,824,000 (+6.4309%) |

### Gradient outcomes

Whole-model norms are pre-clipping. Core statistics are from the final step
before clipping; all were finite. “Elements” is recurrent-core elements with a
gradient over trainable recurrent-core elements.

| Variant/depth | Final model L2 | Mean model L2 | Final core L2 | Core mean abs | Core max abs | Core nonzero fraction | Elements |
|---|---:|---:|---:|---:|---:|---:|---:|
| S0/4 | 0.9175071716 | 1.5674355939 | 0.4577092582 | 0.0013645177 | 0.0423749387 | 0.9925650558 | 25,824/25,824 |
| S0/8 | 0.7632266879 | 2.7799535826 | 0.3368274126 | 0.0009412797 | 0.0302075669 | 1.0000000000 | 25,824/25,824 |
| S1/4 | 1.0167909861 | 1.5778273076 | 0.4043694932 | 0.0011601002 | 0.0422900990 | 0.9925650558 | 25,824/25,824 |
| S1/8 | 1.1081949472 | 1.5093514994 | 0.5979103575 | 0.0014646038 | 0.0438893512 | 1.0000000000 | 25,824/25,824 |
| S2/4 | 0.9379813075 | 1.2930969849 | 0.2926835782 | 0.0009604106 | 0.0240154900 | 0.9925650558 | 25,824/25,824 |
| S2/8 | 1.0275723934 | 1.8114571095 | 0.4595949608 | 0.0013811513 | 0.0464674644 | 1.0000000000 | 25,824/25,824 |
| S3/4 | 0.5267558098 | 1.4288665622 | 0.0416927138 | 0.0000865812 | 0.0032053208 | 0.9925650558 | 25,824/25,824 |
| S3/8 | 0.5314630866 | 0.9795832112 | 0.0543626412 | 0.0001192004 | 0.0047570327 | 1.0000000000 | 25,824/25,824 |
| S4/4 | 0.9998998046 | 1.3933508068 | 0.2982896104 | 0.0009689501 | 0.0270400792 | 0.9925650558 | 25,824/25,824 |
| S4/8 | 1.0130316019 | 1.3955651492 | 0.4104842140 | 0.0011778300 | 0.0426837206 | 1.0000000000 | 25,824/25,824 |

### Task 06 final-step per-iteration state diagnostics

These are the full per-iteration diagnostics from each run's final training
step. The frozen S0 per-iteration records remain verbatim in the Task 05 table
above.

| Variant/depth | Iteration | Hidden RMS | Update RMS | Cosine to previous | Cosine to h0 | Relative update |
|---|---:|---:|---:|---:|---:|---:|
| S1/4 | 1 | 0.1192474961 | 0.0575596057 | 0.9106874466 | 0.9106874466 | 0.4826902449 |
| S1/4 | 2 | 0.1716781408 | 0.0633739680 | 0.9685462713 | 0.8172830343 | 0.3691440821 |
| S1/4 | 3 | 0.2278197110 | 0.0638545603 | 0.9879253507 | 0.7522556782 | 0.2802855074 |
| S1/4 | 4 | 0.2799493968 | 0.0584718473 | 0.9944366217 | 0.7183855772 | 0.2088657618 |
| S1/8 | 1 | 0.1210904792 | 0.0594624318 | 0.8962442875 | 0.8962442875 | 0.4910578430 |
| S1/8 | 2 | 0.1769568622 | 0.0685867295 | 0.9636814594 | 0.7827765346 | 0.3875900805 |
| S1/8 | 3 | 0.2407860309 | 0.0723265409 | 0.9862929583 | 0.7015300989 | 0.3003767431 |
| S1/8 | 4 | 0.3008445501 | 0.0660507381 | 0.9947358370 | 0.6575459242 | 0.2195510119 |
| S1/8 | 5 | 0.3603489697 | 0.0655812323 | 0.9964038134 | 0.6289246082 | 0.1819937080 |
| S1/8 | 6 | 0.4189636707 | 0.0637331083 | 0.9978713393 | 0.6076858640 | 0.1521208733 |
| S1/8 | 7 | 0.4744727910 | 0.0605260357 | 0.9984790087 | 0.5928670764 | 0.1275648028 |
| S1/8 | 8 | 0.5294163227 | 0.0604901388 | 0.9986625910 | 0.5808753967 | 0.1142581552 |
| S2/4 | 1 | 0.1008674130 | 0.0184222013 | 0.9850482345 | 0.9850482345 | 0.1826377809 |
| S2/4 | 2 | 0.1107416674 | 0.0204052664 | 0.9856318235 | 0.9518977404 | 0.1842600852 |
| S2/4 | 3 | 0.1231086180 | 0.0204841178 | 0.9901201725 | 0.9103979468 | 0.1663906276 |
| S2/4 | 4 | 0.1343801618 | 0.0171103068 | 0.9949808121 | 0.8788264990 | 0.1273276359 |
| S2/8 | 1 | 0.1507106572 | 0.0265385769 | 0.9859283566 | 0.9859283566 | 0.1760895848 |
| S2/8 | 2 | 0.1666649878 | 0.0327405073 | 0.9836663008 | 0.9449477196 | 0.1964449883 |
| S2/8 | 3 | 0.1872619987 | 0.0333793797 | 0.9889309406 | 0.8919954300 | 0.1782495826 |
| S2/8 | 4 | 0.2078693658 | 0.0291277375 | 0.9945680499 | 0.8461048603 | 0.1401252002 |
| S2/8 | 5 | 0.2295523584 | 0.0282371473 | 0.9965535998 | 0.8064408302 | 0.1230096146 |
| S2/8 | 6 | 0.2508398592 | 0.0269193538 | 0.9976416230 | 0.7713018656 | 0.1073168814 |
| S2/8 | 7 | 0.2704058290 | 0.0246692356 | 0.9983361959 | 0.7410275340 | 0.0912304223 |
| S2/8 | 8 | 0.2894707322 | 0.0230864827 | 0.9989181757 | 0.7178062797 | 0.0797540992 |
| S3/4 | 1 | 0.0964846089 | 0.0556652062 | 0.8229961395 | 0.8229961395 | 0.5769335032 |
| S3/4 | 2 | 0.0964846089 | 0.0226834882 | 0.9696481228 | 0.6910266876 | 0.2350995541 |
| S3/4 | 3 | 0.0964846089 | 0.0112474468 | 0.9920668602 | 0.6445190310 | 0.1165724471 |
| S3/4 | 4 | 0.0964846015 | 0.0089926226 | 0.9953469038 | 0.6281622052 | 0.0932026729 |
| S3/8 | 1 | 0.1567830592 | 0.0623058416 | 0.8703228235 | 0.8703228235 | 0.3974016011 |
| S3/8 | 2 | 0.1567830592 | 0.0368681587 | 0.9660374522 | 0.7491737008 | 0.2351539731 |
| S3/8 | 3 | 0.1567830741 | 0.0208479147 | 0.9906459451 | 0.6807472706 | 0.1329730153 |
| S3/8 | 4 | 0.1567830592 | 0.0132733006 | 0.9961761832 | 0.6432643533 | 0.0846602991 |
| S3/8 | 5 | 0.1567830592 | 0.0107145458 | 0.9971652627 | 0.6260128021 | 0.0683399364 |
| S3/8 | 6 | 0.1567830741 | 0.0095673976 | 0.9975993633 | 0.6153239012 | 0.0610231571 |
| S3/8 | 7 | 0.1567830592 | 0.0106418571 | 0.9969105721 | 0.6123993993 | 0.0678763315 |
| S3/8 | 8 | 0.1567830592 | 0.0125516849 | 0.9954776764 | 0.5952686667 | 0.0800576657 |
| S4/4 | 1 | 0.0897507370 | 0.0204784572 | 0.9810348749 | 0.9810348749 | 0.2281703949 |
| S4/4 | 2 | 0.1045933738 | 0.0216099881 | 0.9867403507 | 0.9480113983 | 0.2066095471 |
| S4/4 | 3 | 0.1217660978 | 0.0227203798 | 0.9912361503 | 0.9100549221 | 0.1865903437 |
| S4/4 | 4 | 0.1380776316 | 0.0205879547 | 0.9953181744 | 0.8797061443 | 0.1491041780 |
| S4/8 | 1 | 0.0956922174 | 0.0238591265 | 0.9765490890 | 0.9765490890 | 0.2493319511 |
| S4/8 | 2 | 0.1145160422 | 0.0265837666 | 0.9837862253 | 0.9334448576 | 0.2321401238 |
| S4/8 | 3 | 0.1365356892 | 0.0281145163 | 0.9900616407 | 0.8870131969 | 0.2059132904 |
| S4/8 | 4 | 0.1582148373 | 0.0259375870 | 0.9952487350 | 0.8512851596 | 0.1639390290 |
| S4/8 | 5 | 0.1798282266 | 0.0251250286 | 0.9970709682 | 0.8242350817 | 0.1397168487 |
| S4/8 | 6 | 0.2010744512 | 0.0246023200 | 0.9978277087 | 0.7999002337 | 0.1223542914 |
| S4/8 | 7 | 0.2209208757 | 0.0228691157 | 0.9985176325 | 0.7820432186 | 0.1035172120 |
| S4/8 | 8 | 0.2400724739 | 0.0220179074 | 0.9988639951 | 0.7686322927 | 0.0917135924 |

### Gate diagnostics at the final training step

Collapse thresholds were fixed before running: mean at most 0.05 is collapse
toward zero and mean at least 0.95 is collapse toward one. No recorded S2 or S4
iteration collapsed.

| Variant/depth | Iteration | Mean | Population std | Minimum | Maximum | Collapse |
|---|---:|---:|---:|---:|---:|---|
| S2/4 | 1 | 0.4738693535 | 0.0184693988 | 0.4437989593 | 0.5132510066 | none |
| S2/4 | 2 | 0.4638476670 | 0.0191579461 | 0.4330238104 | 0.5075527430 | none |
| S2/4 | 3 | 0.4517212808 | 0.0197871868 | 0.4178956449 | 0.4973346889 | none |
| S2/4 | 4 | 0.4418598115 | 0.0205412321 | 0.4052161574 | 0.4888092577 | none |
| S2/8 | 1 | 0.4523423612 | 0.0233236514 | 0.3972531855 | 0.4978437722 | none |
| S2/8 | 2 | 0.4305236936 | 0.0219121054 | 0.3786760271 | 0.4776660204 | none |
| S2/8 | 3 | 0.4052237868 | 0.0219705291 | 0.3543199003 | 0.4547794163 | none |
| S2/8 | 4 | 0.3849553466 | 0.0230092537 | 0.3353578746 | 0.4374617934 | none |
| S2/8 | 5 | 0.3658730388 | 0.0243770052 | 0.3171966374 | 0.4209060669 | none |
| S2/8 | 6 | 0.3469260037 | 0.0257276092 | 0.2991452217 | 0.4038817883 | none |
| S2/8 | 7 | 0.3335104287 | 0.0270623323 | 0.2820628285 | 0.3917352259 | none |
| S2/8 | 8 | 0.3190125525 | 0.0282360818 | 0.2630709410 | 0.3783378303 | none |
| S4/4 | 1 | 0.4542708695 | 0.0120617896 | 0.4138292670 | 0.4789430499 | none |
| S4/4 | 2 | 0.4474214613 | 0.0130564598 | 0.4057061672 | 0.4731702507 | none |
| S4/4 | 3 | 0.4405429363 | 0.0141697675 | 0.3974108100 | 0.4666051567 | none |
| S4/4 | 4 | 0.4344317317 | 0.0152832065 | 0.3901495636 | 0.4608785808 | none |
| S4/8 | 1 | 0.4564691782 | 0.0109604076 | 0.4353131056 | 0.4802732468 | none |
| S4/8 | 2 | 0.4474558234 | 0.0120503930 | 0.4219431579 | 0.4720166326 | none |
| S4/8 | 3 | 0.4377064407 | 0.0131284548 | 0.4048896432 | 0.4624910057 | none |
| S4/8 | 4 | 0.4288047552 | 0.0143165095 | 0.3899268806 | 0.4545309246 | none |
| S4/8 | 5 | 0.4214301705 | 0.0156026157 | 0.3768897355 | 0.4490883648 | none |
| S4/8 | 6 | 0.4129438400 | 0.0169265512 | 0.3635368347 | 0.4430277050 | none |
| S4/8 | 7 | 0.4072155654 | 0.0182348583 | 0.3532133698 | 0.4399102032 | none |
| S4/8 | 8 | 0.4018296003 | 0.0195945408 | 0.3432334065 | 0.4368115664 | none |

### Predeclared R=4 checkpoint selection

The frozen comparison inputs were S0 R=4 final hidden RMS 0.3238864541 and
cosine to `h0` 0.6662002206. A candidate qualified only if all diagnostics were
finite, RMS was lower, and cosine to `h0` was higher. Among qualifiers, the
lowest evaluation loss was selected.

| Candidate | Eval loss | Final hidden RMS | Final cosine to h0 | Finite | Lower RMS | Higher cosine | Qualifies |
|---|---:|---:|---:|---|---|---|---|
| S1/4 | 3.8888302743 | 0.2799493968 | 0.7183855772 | yes | yes | yes | yes |
| S2/4 | 3.8293781579 | 0.1343801618 | 0.8788264990 | yes | yes | yes | yes |
| S3/4 | 4.2306061983 | 0.0964846015 | 0.6281622052 | yes | yes | no | no |
| S4/4 | 3.7531955242 | 0.1380776316 | 0.8797061443 | yes | yes | yes | yes |

The stability filter was met, and the rule selected
`EXP-0644-anchor-gate-r4` (S4/4). Its config SHA-256 is
`c50c114a331438b94c3b811690dcd4a9d12bdb7cc5e8e2b66394ef56f13efeaf`;
its unchanged checkpoint SHA-256 is
`c94cb3d3386c251285432cfad36f0fd1b787c246f30b6f308f64eb1789760bee`.

### S4/4 cross-depth diagnostic

The selected checkpoint was evaluated without training at inference depths
1, 2, 3, 4, 6, and 8 on the same eight ordered production-tokenizer evaluation
batches (1,024 target tokens at each depth). The source checkpoint and config
file hashes were identical before and after evaluation; the output records
configured training depth 4, 93,553 unchanged parameters, and the source
run/config/checkpoint identity. The ignored result is
`runs/EXP-0644-anchor-gate-r4/analysis/cross-depth-r1-r2-r3-r4-r6-r8.json`,
SHA-256
`462619f9e84d6a98cc4b0a50d7389ac8bfae6b1979e745347d21870c608ac6bc`.
State and gate columns below are the final inference iteration's token-weighted
diagnostic means.

| Inference depth | Active FLOPs/token | Eval loss | Hidden RMS | Update RMS | Cosine to previous | Cosine to h0 | Relative update | Gate mean | Gate std | Gate min | Gate max | Collapse |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 204,096 | 3.8521460295 | 0.0871087424 | 0.0202255065 | 0.9804488048 | 0.9804488048 | 0.2323375810 | 0.4544668496 | 0.0120316666 | 0.4138848037 | 0.4768613838 | none |
| 2 | 265,728 | 3.7836607397 | 0.1022591852 | 0.0216064807 | 0.9866622984 | 0.9472238198 | 0.2115692217 | 0.4476110600 | 0.0128545581 | 0.4057125449 | 0.4695882872 | none |
| 3 | 327,360 | 3.7593515813 | 0.1193386838 | 0.0222604657 | 0.9916436076 | 0.9102945998 | 0.1866576783 | 0.4409213029 | 0.0137910169 | 0.3975648582 | 0.4634466916 | none |
| 4 | 388,992 | 3.7531955242 | 0.1356606781 | 0.0203775370 | 0.9953976050 | 0.8803609014 | 0.1502591372 | 0.4349177219 | 0.0147281478 | 0.3903673030 | 0.4597052485 | none |
| 6 | 512,256 | 3.7674068511 | 0.1605426744 | 0.0170360506 | 0.9972394109 | 0.8415475115 | 0.1061853301 | 0.4322200418 | 0.0169674603 | 0.3845385201 | 0.4635464326 | none |
| 8 | 635,520 | 3.7994184196 | 0.1834889650 | 0.0162580785 | 0.9978623614 | 0.8112493157 | 0.0886994302 | 0.4317342080 | 0.0195436515 | 0.3805301338 | 0.4710663855 | none |

### Five research answers

1. **Does a mechanism reduce drift?** At R=4, S1, S2, and S4 increased final
   cosine to `h0` over S0; S3 decreased it. At R=8, all four exceeded S0, with
   S2 and S4 retaining the most alignment. Drift reduction is therefore
   mechanism- and depth-dependent, not universal.
2. **Does a mechanism control RMS growth?** Yes on this diagnostic: every
   Task 06 variant had lower final RMS than depth-matched S0. S3 held RMS
   essentially constant across iterations by construction, but this did not
   improve its raw evaluation loss.
3. **Does a mechanism prevent successive-state convergence?** No. At R=4 all
   Task 06 final successive-state cosines were slightly higher than S0's
   0.9936016798. At R=8 S3 reduced the value most, to 0.9954776764 from
   0.9992300272, while the other controls remained near 0.999. These are
   diagnostic values, not proof of convergence.
4. **Does a mechanism improve deeper validation behavior?** Raw R=8 losses for
   S1, S2, and S4 were below frozen S0 R=8, while S3 was higher. Every mechanism
   except S3 also had a worse R=8 loss than its separately trained R=4 run.
   Because parameter counts and/or active compute are unmatched, none of these
   raw differences is a controlled model-quality improvement claim.
5. **Does the selected mechanism transfer beyond training depth?** Partially in
   this one checkpoint: S4/4 remained finite with no gate collapse at R=6/R=8,
   but evaluation loss reached its minimum at the trained depth 4 and then rose
   to 3.7674068511 and 3.7994184196 while RMS continued upward and cosine to
   `h0` continued downward. This does not establish general depth transfer.

### Limitations and conclusion boundary

The matrix has one seed, tiny 91K–94K models, only 5,120 training tokens, a
repeated generated corpus, and evaluation from the same pattern family. The
R=4 and R=8 rows are separately trained checkpoints, not one trajectory.
Parameter and active-compute differences are reported but not matched; S1 and
S4 add both parameters and matmul work, S2 adds a smaller amount, and the
matmul-only estimate omits S2/S3 elementwise costs. CPU peak VRAM is unavailable
and host throughput is noisy. The selection rule reuses the R=4 evaluation
loss after a predeclared stability filter, and only the selected S4/4 checkpoint
received cross-depth analysis. No multi-seed, independent-corpus, large-scale,
GPU, statistical-significance, or downstream-reasoning evidence exists.

Task 06 supports an engineering conclusion that the controls are measurable
and that several reduce selected state-drift diagnostics in this tiny setting.
It does not establish recurrence superiority, language-model improvement, or
that AURORA is successful. Research Task 07 has not started.
