# AURA Research & Engineering Plan

> **Working codename:** AURA — Adaptive Updating Recurrent Architecture  
> **Status:** research blueprint, not a locked architecture  
> **Literature snapshot:** 2026-08-26  
> **Primary goal:** build a small, fully trainable-from-random-initialization language/reasoning model that is unusually strong **per parameter and per unit of inference compute** at continual memory, long-context reasoning, and adaptive latent computation.

---

## 0. Executive decision

Do **not** try to create a miniature general-purpose ChatGPT and call it state of the art. SmolLM2-135M, for example, was trained on roughly **2 trillion tokens**, so beating mature small LMs at generic world knowledge with a student-scale run is mainly a data/compute contest rather than a clean research contribution.

The research target should instead be a **capability frontier**:

1. **persistent neural memory** that can incorporate new information after pretraining without modifying the frozen backbone on every query;
2. **recurrent latent reasoning** whose effective depth can grow at test time without growing parameter count;
3. **adaptive computation** so easy/familiar inputs use fewer recurrent steps while difficult, novel, or conflicting inputs use more;
4. **stable recurrence** that does not degrade when extra loops are provided;
5. **explicit scratch state** so transient reasoning does not compete with long-lived memory in one hidden representation;
6. rigorous comparisons at **iso-parameter, iso-FLOP, iso-token, and iso-latency** budgets.

The strongest initial paper hypothesis is:

> **A recurrent language model can learn to jointly allocate memory plasticity and reasoning depth from memory-derived novelty, conflict, retrieval uncertainty, and convergence signals, reducing compute on familiar problems while spending additional latent computation on novel or contradictory ones.**

This is deliberately narrower than “memory + recurrence.” Pieces of that broader idea already exist in SpeedupLLM, MeSH, AURA-Mem, Titans/ATLAS, LoopFormer, Mixture-of-Recursions, and other work. Novelty must be re-audited immediately before the paper is written.

### Naming warning

A June 2026 paper already uses **AURA** for *AURA: Action-Gated Memory for Robot Policies at Constant VRAM* (arXiv:2606.02775). Keep AURA only as an internal codename. Rename the public model/paper before release. A candidate working alternative is **AURORA-LM — Adaptive Updating Recurrent Online Reasoning Architecture**, but that name must also be searched for collisions before publication.

---

# 1. What is actually new enough to investigate?

## 1.1 The proposed scientific contribution

Define a controller, provisionally called the **Experience-Adaptive Controller (EAC)**.

For each segment / reasoning state, the controller receives a compact signal vector:

- prediction surprisal;
- retrieval confidence;
- disagreement between retrieved memory and current evidence;
- change in next-token distribution between recurrent iterations;
- state-update magnitude;
- entropy of the output distribution;
- recency and frequency of retrieved memories;
- memory-capacity pressure;
- optional learned “utility of remembering” estimate.

It produces four decisions:

1. **WRITE** — how strongly to encode the current information in fast memory;
2. **FORGET/REPLACE** — how much stale or conflicting memory to decay/overwrite;
3. **THINK** — how much additional recurrent depth this example/token needs;
4. **CONSOLIDATE** — whether knowledge should be promoted from fast episodic memory into a slower semantic tier.

A minimal formulation:

```text
signals_t = [surprise, retrieval_uncertainty, conflict,
             delta_logits, state_velocity, output_entropy,
             recency, frequency, capacity_pressure]

[w_t, f_t, p_t, c_t] = Controller(signals_t)

M_fast  <- MemoryUpdate(M_fast, x_t, strength=w_t, forget=f_t)
h_(r+1) <- RecurrentCore(h_r, input_anchor, scratch, Read(M, h_r))
stop     <- HaltingRule(p_t, convergence, max_depth)
M_slow   <- Consolidate(M_fast, M_slow) when c_t is high or on scheduled sleep
```

### Why this is promising

- **Titans / ATLAS / MIRAS** show that memory update rules, retention, attentional bias, and online optimization are independent design axes rather than one fixed recipe.
- **Nested Learning / HOPE** argues for multiple update frequencies and self-modifying learning modules.
- **Language Models Need Sleep** argues that online memory alone is insufficient and explores periodic offline consolidation.
- **MoNe** shows local fast-weight updates can summarize very long context with constant query-time dependence on original context length.
- **RecurrentGPT / Gated Recurrent Transformers, LoopFormer, MeSH, MoR, ACT work** show that shared-depth models require iteration specialization, explicit state handling, and careful routing.
- **STARS** demonstrates that recurrent latent reasoning can become unstable and motivates fixed-point / dynamical-systems regularization.
- **SpeedupLLM** already connects prior experience to reduced reasoning compute at a system level, so AURA must go beyond simply appending memory and early-stopping.

The novelty should therefore live in the **joint, end-to-end learned coupling between memory quality and compute allocation**, not in the mere existence of memory or recurrence.

---

# 2. Evidence tiers: what goes into the main model versus the research backlog

A common failure mode in student research is to combine every interesting paper into one giant architecture. That destroys causal attribution. Use three evidence tiers.

## Tier A — put in the main baseline path

These are mature enough, simple enough, or sufficiently supported to form the default implementation.

- decoder-only autoregressive LM objective;
- pre-normalized residual blocks using RMSNorm;
- QK normalization or QK-LayerNorm as a stability option;
- SwiGLU feed-forward layers;
- Grouped Query Attention (GQA) for KV-cache efficiency;
- RoPE or another well-understood relative positional method;
- one or two **fixed prelude blocks** before recurrence;
- a small shared recurrent core;
- one or two **fixed coda blocks** after recurrence;
- explicit scratch/memory tokens;
- recurrence-step conditioning;
- randomized recurrence depth during training;
- maximum-depth guardrail;
- output/trajectory consistency loss for short versus long recurrence;
- AdamW as the control optimizer;
- bf16 training initially;
- gradient clipping;
- high-quality data filtering, deduplication, contamination checks;
- FlashAttention / PyTorch SDPA/FlexAttention where supported;
- exact reproducible evaluation harness.

## Tier B — strong experimental branches

These deserve serious ablations but should not be mandatory for v0.

- Gated Recurrent Transformer-style input anchoring and per-loop modulation;
- MeSH-style state highways / separated long-lived and transient state;
- Mixture-of-Recursions or continuous token-level compute routing;
- ACT with **deep-start** router initialization and ponder-cost warmup;
- LoopFormer-style normalized-time + step-size conditioning;
- LoopFormer shortcut consistency loss;
- STARS-style Jacobian spectral-radius regularization;
- ATLAS-style context-level memory optimization rather than strictly one-token online updates;
- MIRAS retention gates and alternative associative-memory objectives;
- MoNe-style layer-local fast-weight memory;
- multi-timescale memory inspired by HOPE/Mela;
- Muon versus AdamW;
- u-µP for hyperparameter transfer across widths;
- multi-token prediction as an auxiliary loss;
- byte-level/BLT tokenization branch after the central memory/recurrence result is established.

## Tier C — high-risk / high-upside research

These are interesting enough for follow-up papers but too risky to make the first result depend on them.

- deep-equilibrium / attractor fixed-point inference with implicit differentiation;
- “sleep” consolidation that activates new capacity and distills fast memory into slow modules;
- REFINE-style next-sequence-prediction objectives for fast weights plus RL;
- self-guided test-time training;
- continual expansion of slow memory experts;
- learnable-novelty objectives;
- consolidation-driven attention suppression (“familiar information should require less attention”);
- byte-level block-diffusion decoding;
- differential attention;
- FP8-first training;
- custom learned optimizers / Nested Learning optimizer-as-memory designs;
- autonomous synthetic-data evolution/curation loops;
- external verifier / RLVR reasoning post-training at very small model sizes;
- token-specific variable recurrence plus memory-specific variable recurrence in the same first model.

---

# 3. Architecture: AURA v1

## 3.1 High-level data flow

```text
Tokens
  │
  ▼
Embedding + Position
  │
  ▼
Fixed Prelude Blocks  ──────────────┐
  │                                  │ input anchor
  ▼                                  │
Initial Working State h0             │
  │                                  │
  ├────► Fast/Slow Memory Read ◄─────┤
  │                                  │
  ▼                                  │
┌─────────────────────────────────────────────┐
│       Shared Recurrent Reasoning Core       │
│                                             │
│  h_r ─► Norm ─► Attention/Memory ─► FFN    │
│   │                               │         │
│   └──── gated residual update ◄───┘         │
│                                             │
│  scratch tokens / state highway persist     │
│  step/time embedding modulates each loop    │
└─────────────────────────────────────────────┘
  │             ▲
  │             └── loop until budget/halting/convergence
  ▼
Fixed Coda Blocks
  │
  ▼
LM Head
  │
  ├──► output distribution
  ├──► convergence / uncertainty signals
  └──► memory-write signals
```

## 3.2 Why fixed prelude and coda blocks

Shared layers tend to perform the same transformation repeatedly unless they receive iteration-specific context. Recent gated recurrent Transformer work uses fixed prelude/coda blocks around a shared core, plus an input anchor and loop-conditioned gates, to preserve input grounding while letting the shared core specialize across recurrence.

AURA should begin with:

- **Prelude:** 2 unique Transformer blocks;
- **Recurrent core:** 2 unique blocks shared across loops;
- **Coda:** 1 or 2 unique blocks;
- **default training recurrence:** sample 2–8 loops;
- **test-time recurrence:** allow 1–16 initially, then test extrapolation to 32 if stable.

Do not start with 64+ loops. Multiple 2026 studies show overthinking, representation drift, or outright collapse outside the training depth range.

## 3.3 Recurrent gated update

A useful starting equation is:

```text
u_r = Core(Norm(h_r), scratch_r, Read(M, h_r), loop_embed_r)

g_r = sigmoid(Wg [h_r ; h_anchor ; loop_embed_r ; memory_signal_r])

h_(r+1) = h_r + g_r ⊙ Project(u_r)
```

Optional additions:

- resampled loop noise during training, inspired by Gated Recurrent Transformers;
- residual scale dependent on normalized loop time;
- separate attention and FFN gates;
- a step-size variable Δt so a 2-step trajectory can learn to approximate an 8-step trajectory.

## 3.4 Scratch state

Do **not** force permanent facts, current token state, and temporary reasoning variables into the same hidden vector.

Start with 16 learned scratch tokens per sequence. Ablate:

- 0;
- 8;
- 16;
- 32;
- 64.

“Universal Transformers Need Memory” reports a sharp success threshold and later dilution on its Sudoku setting, so more scratch state is not automatically better. Treat scratch capacity and recurrent depth as substitutable resources.

Potential MeSH-inspired split:

```text
working token state   = short-lived per-token representation
scratch state         = temporary recurrent reasoning variables
persistent fast state = knowledge learned within current stream/session
slow state            = consolidated semantic knowledge
```

## 3.5 Recurrent step specialization

Every loop receives an embedding of:

- loop index r;
- normalized time t=r/R;
- optional Δt;
- current memory confidence;
- current convergence statistic.

Modulate RMSNorm scale and residual gates with this embedding. This prevents the shared block from behaving identically on every pass.

## 3.6 Halting strategy: build in stages

### Stage 1 — externally chosen compute budget

During early architecture work, **do not learn halting yet**. Train elastic depth and evaluate at fixed loop counts {1,2,4,8,12,16}. This makes failures interpretable.

### Stage 2 — global convergence halting

Stop when all required conditions are satisfied, e.g.:

```text
KL(p_r || p_(r-1)) < τ_KL
AND output_entropy < τ_H
AND state_delta < τ_state
```

Always enforce min_loops and max_loops.

### Stage 3 — learned sequence-level halting

Train a small controller to predict the marginal value of another loop. Penalize compute:

```text
L_total = L_LM + λ_compute * E[loops] + other_aux_losses
```

Initialize the halting bias toward **deeper computation**, not immediate exit. ACT experiments in 2026 found shallow-halting initialization traps; a “deep start” solved the failure in that setting.

### Stage 4 — optional token-level routing

Only after sequence-level dynamic depth is stable, test:

- top-k token routing (MoD style);
- per-token recursion depth (MoR style);
- continuous residual/token gates (TSA style).

Token-level dynamic depth creates padding/divergence and kernel-efficiency issues. It is scientifically interesting but should not block the first paper.

---

# 4. Stability: recurrence is a dynamical system

This should be a first-class part of the project, not an afterthought.

## 4.1 Known problem

Looped LMs often improve until some recurrence depth and then degrade. STARS reports that normalization placement changes a stability/effectiveness tradeoff: internal normalization can preserve information while allowing state growth; external normalization can bound trajectories but become too inert.

## 4.2 Instrument these quantities from day one

For every recurrent step log:

- ||h_r|| RMS;
- ||h_r - h_(r-1)||;
- cosine similarity between successive states;
- next-token KL divergence between iterations;
- output entropy;
- recurrent gate mean/std and saturation rate;
- attention entropy;
- scratch-token norm;
- memory-read norm;
- approximate top Jacobian singular/eigen value on probe batches;
- gradient norm by module and loop;
- CKA or representation similarity periodically.

## 4.3 Stable-depth training recipe

1. sample recurrence depth rather than train at exactly one depth;
2. include shallow and deep routes;
3. keep an explicit input anchor from the prelude;
4. provide loop/time conditioning;
5. use shortcut/trajectory consistency;
6. optionally add a small fixed-point regularizer;
7. evaluate beyond the training depth at every checkpoint;
8. reject checkpoints whose accuracy collapses with additional recurrence even if their fixed-depth score is strong.

## 4.4 STARS branch

Experiment with Jacobian Spectral Radius Regularization (JSRR): approximate the leading Jacobian eigenvalue using Jacobian-vector products and one-step power iteration. Penalize trajectories whose recurrent map is locally unstable.

Do this only after the plain recurrent model works, because it adds training cost and complexity.

## 4.5 Attractor/equilibrium branch

High-risk experiment:

- reason until `||h_(r+1)-h_r|| < ε`;
- use a learned attractor/refinement module;
- test implicit differentiation to avoid storing all recurrent activations.

This branch could become its own paper if it gives reliable compute-adaptive reasoning. Do not make AURA v1 depend on it.

---

# 5. Persistent neural memory

## 5.1 Separate memory from the backbone

Backbone slow weights should not be directly gradient-updated for every test-time fact. Use a dedicated memory module with fast weights/state. Reasons:

- easier to reset for experiments;
- easier to measure forgetting;
- easier to bound memory footprint;
- prevents test-time adaptation from corrupting general capabilities;
- makes ablations clean;
- allows multiple user/session memories with one backbone.

## 5.2 Memory tiers

### M0 — working/scratch memory

Lifetime: recurrent loops within one forward decision.

### M1 — fast episodic memory

Lifetime: stream/session. Updated frequently from current context.

Candidate implementation:

- small associative linear map or SwiGLU MLP;
- DeltaNet-style delta updates or learned gradient descent;
- layer-local fast weights;
- learned write rate;
- learned momentum/decay;
- normalization of memory weights after updates.

### M2 — slow semantic memory

Lifetime: many streams / long horizon.

Updated much less often. Initially implement as a slower copy/expert or compressed memory bank rather than permanently modifying the core Transformer.

### M3 — backbone knowledge

Lifetime: pretraining / major retraining cycle. Do not update during ordinary inference in the first paper.

## 5.3 MIRAS design matrix

Treat memory as four independent choices:

1. **memory architecture** — linear map, MLP, low-rank memory, key/value slots;
2. **attentional-bias objective** — dot product, L2-style regression, alternatives;
3. **retention gate** — decay/forget mechanism;
4. **learning rule** — delta rule, SGD-like update, momentum, learned optimizer.

Create a sweep table rather than guessing one configuration.

## 5.4 Write gate

A memory write should depend on more than raw next-token surprise. Raw surprise also fires on noise.

Candidate inputs:

- model surprisal;
- **learnable novelty**: surprise that the model can reduce after learning;
- contradiction with retrieved memory;
- predicted future utility;
- recurrence depth required to resolve the current item;
- frequency/recency;
- whether the current item changed the output distribution;
- memory capacity pressure.

The write score:

```text
w_t = sigmoid(MLP(write_features_t))
```

Train it with both downstream prediction loss and synthetic memory supervision in procedural environments where the ground truth of “what matters later” is known.

## 5.5 Forget/overwrite gate

Memory must support:

- addition;
- correction;
- deletion;
- supersession;
- temporal qualification;
- source conflict;
- aging.

Generate explicit benchmark events such as:

```text
T1: CEO(Zeta) = Maya
T2: CEO(Zeta) = Arjun
T3: Arjun leaves Zeta
T4: CEO(Zeta) = Priya
```

Then ask both current-state and historical questions.

The model must learn that “forget Maya as the current CEO” does **not** mean erase the historical fact that Maya once held the role.

## 5.6 Context-level update versus purely online update

ATLAS argues that memory optimized only on the last token is limited. Test chunk/segment-level update rules that incorporate recent tokens jointly.

Candidate chunk sizes: 64, 128, 256, 512.

Use truncated/local gradients for memory updates where possible so very long streams remain tractable.

## 5.7 MoNe-style local fast-weight branch

A strong practical branch:

1. process context in fixed-size segments;
2. update a small memory attached to selected layers only;
3. discard raw context;
4. answer future questions from the final memory state;
5. reuse the final memory across multiple queries;
6. append new segments incrementally.

This gives a clean long-context efficiency experiment because query cost can be made independent of the original raw context length.

## 5.8 Consolidation / sleep branch

After fast episodic memory works, add a periodic consolidation process:

```text
Wake:
  read stream
  update M1 frequently
  collect replay candidates

Sleep:
  rank/replay M1 memories
  compress / distill into M2
  prune redundant M1 entries
  generate counterfactual/rehearsal examples
  test old + new knowledge before committing
```

Never allow “sleep” to silently destroy previous capabilities. Gate consolidation on a regression suite.

This is a high-risk branch because it introduces a second training process and possible capacity growth.

---

# 6. The key proposed mechanism: Experience-Adaptive Compute

## 6.1 Core idea

Familiar, confidently retrieved information should require less reasoning. Novel, conflicting, or uncertain information should receive more recurrent computation.

Define:

```text
novelty_t       = f_surprise(x_t, model, memory)
conflict_t      = distance(current_evidence, retrieved_memory)
retrieval_unc_t = uncertainty(Read(M, query_t))
convergence_r   = distance(p_r, p_(r-1))

compute_need = EAC([novelty_t, conflict_t, retrieval_unc_t,
                    convergence_r, output_entropy, scratch_state])
```

Then:

```text
R_example = clamp(round(R_min + compute_need * (R_max - R_min)), R_min, R_max)
```

Later replace this discrete formula with learned halting.

## 6.2 Scientific hypotheses

**H1 — Memory helps accuracy:** persistent fast memory improves update/recall tasks versus equal-compute recurrent baseline.

**H2 — Recurrence helps reasoning:** recurrent depth improves compositional/algorithmic reasoning versus same-parameter fixed-depth baseline.

**H3 — Joint controller helps efficiency:** memory-aware depth allocation yields a better accuracy/FLOP frontier than fixed depth or depth routing that ignores memory.

**H4 — Experience causes speedup:** repeated exposure to task families decreases average recurrent steps without reducing accuracy.

**H5 — Conflict reverses speedup:** when known information is contradicted, compute temporarily increases, memory is revised, then later queries become cheaper again.

**H6 — Consolidation reduces future cost:** slow-memory consolidation lowers retrieval/attention/recurrent cost on repeatedly encountered patterns.

## 6.3 Critical controls

To prove H3/H4, compare:

- random-depth controller;
- entropy-only controller;
- convergence-only controller;
- memory-confidence-only controller;
- novelty-only controller;
- full EAC;
- fixed depth at the same mean FLOPs;
- oracle difficulty controller on synthetic tasks.

If the full controller does not beat these, do not claim joint memory-compute allocation is the contribution.

---

# 7. Tokenization strategy

## 7.1 Main paper: use a normal tokenizer first

Although tokenizer-free models are attractive, adding raw-byte modeling simultaneously with novel memory and recurrence makes attribution much harder.

For AURA v1:

- train a tokenizer from the project corpus;
- 16K–32K vocabulary;
- byte fallback;
- publish tokenizer training code and exact corpus manifest.

## 7.2 AURA-Byte branch

After the central model works, test a Byte Latent Transformer-inspired front end:

- consume raw UTF-8 bytes;
- estimate local entropy/surprise;
- dynamically patch easy byte spans into larger units;
- retain fine granularity on difficult/unusual spans;
- compare robustness to misspellings, code, multilingual text, identifiers, and transliteration.

This could become a second paper or a major appendix. Do not let it delay the primary result.

---

# 8. Backbone choices to benchmark

AURA should not be compared only with a vanilla Transformer.

## 8.1 Dense Transformer baseline

Modern small decoder:

- RMSNorm;
- RoPE;
- SwiGLU;
- GQA;
- no memory;
- fixed unique layers.

## 8.2 Weight-shared Transformer baseline

Same core parameter count but simple repeated layers, no gating/scratch/memory.

## 8.3 Gated recurrent baseline

Prelude + recurrent core + coda + input anchor + loop-conditioned gate, but no persistent memory.

## 8.4 MoR baseline

Mixture-of-Recursions-like variable-depth recurrent baseline if implementation is available/reproducible.

## 8.5 Recurrent state-space / linear-attention baseline

At minimum one of:

- Mamba-2;
- DeltaNet;
- GLA-style model;
- xLSTM.

## 8.6 Memory baseline

At minimum one of:

- Titans-inspired neural memory;
- TTT-Linear/MLP-style fast weights;
- MoNe-inspired fast-weight plug-in.

## 8.7 External memory baseline

Simple retrieval baseline:

- exact key/value memory for synthetic tasks;
- vector retrieval/RAG for text tasks.

This prevents the paper from overclaiming that a learned neural memory is better when a database would solve the task more simply.

---

# 9. Parameter scales

Do research at multiple scales.

## 9.1 Debug model: 5M–15M

Purpose:

- verify losses;
- overfit tiny datasets;
- test memory update correctness;
- test recurrence convergence;
- run hundreds of ablations cheaply.

## 9.2 Architecture-search model: ~30M–60M

Purpose:

- compare memory rules;
- scratch-token counts;
- loop conditioning;
- stability methods;
- controller features;
- optimizer choices.

Run multiple seeds.

## 9.3 Main science model: ~125M–160M

This is the first serious publication target because strong comparison models exist near 125M/135M/160M.

Suggested initial shape, to be tuned with u-µP or sweeps:

```text
vocab:          24K–32K
width:          768–1024
heads:          12–16 query heads
KV heads:       4
prelude:        2 blocks
recurrent core: 2 blocks
coda:           1–2 blocks
scratch:        16 tokens
train loops:    sampled 2–8
max eval loops: 16 initially
context:        2K → 4K staged; memory streams much longer
```

Do not freeze this exact layout until FLOP- and parameter-matched baselines are generated.

## 9.4 Stretch model: 300M–400M

Only scale after the 30M/135M results show a stable advantage.

The stretch model tests whether the mechanism scales, not whether adding parameters fixes a weak idea.

---

# 10. Training objectives

## 10.1 Base objective

Autoregressive next-token prediction:

```text
L_NTP = -Σ log p(x_t | x_<t)
```

## 10.2 Multi-depth LM loss

Compute next-token loss at several recurrent exits:

```text
L_depth = Σ_r α_r * CE(p_r, target)
```

Weight later exits more strongly but keep early exits useful.

## 10.3 Shortcut consistency

Align shorter trajectories with a stop-gradient deep trajectory:

```text
L_cons = D(stopgrad(p_deep), p_short)
```

KL or logit MSE can be tested.

## 10.4 Convergence loss

Optional:

```text
L_conv = ||h_R - h_(R-1)||²
```

Use cautiously: forcing states to be identical too quickly may suppress useful deep reasoning.

## 10.5 Stability loss

Optional JSRR/STARS-style penalty on estimated recurrent Jacobian spectral radius.

## 10.6 Compute/ponder loss

After fixed-budget training is stable:

```text
L_compute = λ * expected_recurrent_steps
```

Warm λ from zero rather than applying strong early pressure.

## 10.7 Memory-write supervision

On procedural data where future utility is known, supervise whether an item should be remembered.

Possible labels:

- needed later;
- stale;
- superseded;
- historical-only;
- irrelevant distractor.

## 10.8 Memory retrieval loss

Train the memory to reproduce or transform relevant targets from keys/queries. Compare token-local loss versus chunk/context-level memory loss.

## 10.9 Fast-weight update objective

Start with NTP-style memory adaptation. Later test next-sequence prediction / REFINE-style objectives because next-token updates may not optimize the information that future queries actually require.

## 10.10 Multi-token prediction branch

Try predicting 2–4 future tokens from shared representations only after baseline stability. Literature shows advantages can be scale-dependent; very small models may be hurt by the extra burden.

---

# 11. Data strategy: create an auditable corpus, not a mystery pile

The dataset should be a research artifact in its own right.

Call the provisional training mixture **AURA-Corpus** and the procedural benchmark **AURA-MemoryBench** until renaming.

## 11.1 Data principles

- provenance for every source;
- license/terms manifest;
- exact hashes and dataset versioning;
- language detection;
- PII/secrets filtering;
- document-level and near-duplicate deduplication;
- benchmark-contamination scanning;
- quality scoring;
- syntax/code quality filters for code;
- educational/reasoning quality filters;
- domain mixture recorded in machine-readable config;
- train/validation/test split before synthetic transformations where possible.

## 11.2 Natural-data mixture

Build from legally usable/open sources with strong provenance. Categories:

- high-quality web text;
- educational/explanatory text;
- books/public-domain or properly licensed text;
- code with compatible licenses;
- math/science text;
- structured reference text;
- multilingual data if included in the research scope.

DataComp-LM and FineWeb work strongly motivate model-based quality filtering rather than assuming more scraped text is better.

## 11.3 Synthetic data policy

Do not train primarily on model-generated textbooks just because synthetic data is cheap.

A large August 24, 2026 study across >1000 models found synthetic-data benefits are highly mixture-dependent; rephrased synthetic text mixed with natural text performed much better than synthetic-only setups in its experiments. Use this as a **starting hypothesis**, not a universal constant.

Initial sweep:

- 0% transformed synthetic;
- 10%;
- 20%;
- 30%;
- 40%.

Keep procedural memory/reasoning data as a separate category from rephrased natural text.

## 11.4 Procedural mutable-world generator

This is central to the paper.

Generate universes with:

- entities;
- properties;
- relations;
- events;
- temporal order;
- causal chains;
- aliases;
- contradictions;
- corrections;
- deletions;
- repeated facts;
- irrelevant distractors;
- latent rules.

Example:

```text
World seed: 938402

T0: Company Aster founded by Lian.
T1: Aster's database is QuillDB.
T2: Mira becomes CTO of Aster.
T3: Aster migrates from QuillDB to PineDB.
T4: Mira leaves. Niko becomes CTO.
T5: Aster creates product Helio.

Questions:
- What database does Aster use now?
- Which database did Aster use before PineDB?
- Who is the current CTO?
- Was Mira CTO before or after the migration?
- What changed between T2 and T5?
```

Every world has perfect ground truth, enabling exact measurement of memory write/overwrite behavior.

## 11.5 Procedural reasoning families

Add generators for:

- graph reachability;
- shortest paths;
- symbolic arithmetic;
- variable binding;
- set operations;
- relational composition;
- temporal logic;
- scheduling;
- constraint satisfaction;
- simple program execution;
- boolean circuits;
- finite-state machines;
- copy/reverse/sort/deduplicate;
- nested parentheses and stack operations;
- maze/Sudoku-like small combinatorial tasks if architecture supports them.

Difficulty is controlled by graph depth, number of distractors, required hops, or state size.

This lets you test whether recurrent steps increase as objective difficulty increases.

## 11.6 Familiarity curriculum

For EAC research, each stream contains clusters of related tasks:

```text
new family → repeated related problems → mastered family → distribution shift → contradiction
```

Measure whether compute falls with experience and rises again when the environment changes.

## 11.7 Long-stream curriculum

Train ordinary attention contexts at 512/1K/2K/4K, but train persistent memory on streams far longer than the raw attention window by processing segments sequentially.

Evaluation lengths:

- 4K;
- 8K;
- 16K;
- 32K;
- 64K;
- 128K;
- longer synthetic streams if memory state remains bounded.

Length extrapolation must be tested on lengths not seen during training.

---

# 12. Optimizer and numerical-stability research

## 12.1 Control: AdamW

Use AdamW first because it is a stable, understood baseline.

Typical initial region to sweep, not fixed prescriptions:

- β1 = 0.9;
- β2 in {0.95, 0.98};
- weight decay around 0.1;
- gradient clip 1.0;
- warmup followed by cosine or WSD.

## 12.2 Muon branch

Muon is worth a real controlled trial. Published work reports strong compute efficiency in LLM training after careful scaling/weight-decay treatment.

Rules:

- do not switch to Muon and change architecture simultaneously;
- compare equal tokens and equal wall-clock;
- record optimizer-state memory;
- run at least three seeds for small-scale tests;
- keep embeddings/norm/scalars in appropriate Adam-like groups if required by implementation.

## 12.3 SOAP / COSMOS branch

Only evaluate if optimizer becomes a bottleneck or Muon is inconclusive. These add memory/implementation complexity.

## 12.4 u-µP

Strongly consider u-µP for architecture scaling because AURA will be tested at 30M, ~135M, and possibly ~350M. The point is to transfer hyperparameters across width rather than re-sweep everything.

Before depending on it, reproduce its scaling behavior on your plain Transformer baseline.

## 12.5 Normalization

Default:

- Pre-RMSNorm;
- QK-Norm/QK-LayerNorm enabled as a switch;
- norm statistics logged by recurrence depth.

Experimental:

- PostNorm recurrence;
- mixed internal/external norm;
- FlashNorm exact kernel reformulation;
- TaperNorm/HybridNorm.

STARS suggests normalization location materially changes recurrent dynamics, so this is a scientific ablation, not merely a code detail.

---

# 13. Training schedule

## Phase 0 — mathematical/unit-test implementation

### Exit criteria

- every module has shape tests;
- causal masking tests pass;
- memory reset/isolation works;
- deterministic tiny-seed run matches expected loss;
- 5M model overfits a tiny corpus;
- gradient checks for custom memory update pass;
- repeated recurrence does not silently detach gradients;
- inference and training recurrence agree when dropout off.

## Phase 1 — dense baseline reproduction

Build a 30M plain Transformer.

Prove:

- expected loss curve;
- no data leakage;
- stable throughput;
- lm-eval harness works;
- checkpoint resume gives the same trajectory within numerical tolerance.

## Phase 2 — recurrent baseline

Add:

- prelude/core/coda;
- recurrent step embedding;
- fixed loop counts.

Compare to parameter- and FLOP-matched dense models.

## Phase 3 — recurrence stability

Add one at a time:

1. input anchoring/gated update;
2. scratch tokens;
3. randomized depths;
4. shortcut consistency;
5. convergence metrics;
6. STARS/JSRR branch.

Freeze the best recurrence recipe before adding persistent memory.

## Phase 4 — fast memory

Implement simplest memory first:

- linear/MLP associative memory;
- delta update;
- write gate;
- forget gate;
- resettable state.

Train on procedural memory tasks before mixing into full LM pretraining.

## Phase 5 — integrated AURA

Combine the frozen best recurrence recipe with the best memory recipe.

Initially use **fixed recurrence budgets** so memory benefits are measurable independently.

## Phase 6 — Experience-Adaptive Controller

Add EAC.

Start sequence-level. Train compute penalty gradually. Compare with all controller baselines.

## Phase 7 — multi-timescale memory

Add M2 slow memory and optional consolidation.

## Phase 8 — 125M–160M main run

Only when all Phase 0–7 criteria are met at 30M–60M.

Run:

- dense baseline;
- recurrent-no-memory;
- memory-no-adaptive-compute;
- full AURA;
- at least one external modern baseline.

## Phase 9 — stretch experiments

Candidates:

- 300M–400M scale;
- byte-level front end;
- token-level dynamic depth;
- attractor inference;
- sleep consolidation;
- RL/post-training;
- custom kernels.

---

# 14. Benchmark suite

## 14.1 Generic LM quality

Use a standard harness and report at least:

- validation perplexity/loss on held-out natural corpora;
- HellaSwag;
- PIQA;
- ARC-Easy/Challenge;
- WinoGrande;
- OpenBookQA;
- BoolQ;
- MMLU/MMLU-Pro only with proper caveats at very small scales.

Do not tune AURA specifically against public test sets.

## 14.2 Long-context / memory

Prioritize:

- RULER;
- BABILong;
- LongBench v2;
- LongBench Pro;
- InfiniteBench;
- OneRuler where multilingual evaluation is relevant;
- LongGenBench where long generation matters;
- AURA-MemoryBench.

## 14.3 Reasoning

- GSM8K;
- small subset / level-appropriate MATH or MATH-500;
- algorithmic generators with exact answers;
- relational/graph reasoning;
- ProofWriter / RuleTaker-like logical tasks;
- CLUTRR-like relational composition;
- optional Sudoku/Maze at controlled sizes.

For a 135M base model, algorithmic/procedural benchmarks are more scientifically informative than expecting frontier-level MATH scores.

## 14.4 Continual learning

Include:

- sequence of factual updates;
- conflicting facts;
- old/new task mixture;
- class/domain incremental streams;
- long-horizon retention;
- recovery after stale-memory correction.

Look at emerging continual-learning benchmarks such as CL-Bench and the 2025/2026 memory/continual-learning evaluation literature.

## 14.5 Adaptive compute

For every reasoning benchmark, plot:

- accuracy versus mean loops;
- accuracy versus FLOPs;
- accuracy versus latency;
- accuracy versus energy if measurable;
- loops versus objective problem difficulty;
- loops versus familiarity/exposure count;
- loop histogram by token/example;
- marginal gain from each additional recurrent step;
- overthinking curve beyond the chosen budget.

## 14.6 Memory-specific metrics

Create explicit measures:

- write precision / recall;
- retrieval recall@k if slot-based;
- factual update success;
- stale-fact rejection;
- historical-fact preservation;
- contradiction resolution;
- forgetting curve / retention half-life;
- interference after N unrelated writes;
- number of reliable memories per parameter/byte;
- memory write FLOPs;
- memory read FLOPs;
- memory state bytes;
- context preprocessing latency;
- per-query latency after memory creation;
- multi-query amortization.

---

# 15. The benchmark that could make the paper memorable: AURA-MemoryBench

Public long-context benchmarks often test retrieval from static context. AURA's core claim is stronger: **state changes over time and the model should become computationally more efficient as it learns.**

Create a benchmark with five axes.

## Axis A — temporal mutability

- stable fact;
- updated fact;
- reverted fact;
- deleted fact;
- conflicting-source fact.

## Axis B — dependency depth

- direct recall;
- 2-hop;
- 4-hop;
- 8-hop;
- compositional rule use.

## Axis C — interference

Insert 0 → 100K+ unrelated events between learning and query.

## Axis D — familiarity

Present similar task families multiple times and measure compute reduction.

## Axis E — distribution shift

After proficiency develops, change a governing rule and measure:

1. accuracy drop;
2. temporary compute increase;
3. memory correction speed;
4. eventual compute reduction after relearning.

### Primary benchmark score

Do not collapse everything into accuracy only. Report a Pareto surface and optionally define a transparent summary score such as:

```text
Utility = accuracy / normalized_inference_FLOPs
```

but always publish raw accuracy, FLOPs, latency, and memory separately so the summary score cannot hide tradeoffs.

---

# 16. Ablation matrix

At minimum the final paper should contain these.

## Architecture

- dense unique layers;
- naive shared recurrence;
- + prelude/coda;
- + input anchor/gating;
- + loop/time conditioning;
- + scratch tokens;
- + shortcut consistency;
- + stability regularization.

## Memory

- no memory;
- exact external memory;
- vector retrieval;
- linear fast weights;
- MLP fast weights;
- token-local update;
- chunk/context update;
- no forget gate;
- learned forget gate;
- one timescale;
- multiple timescales.

## Controller

- fixed loops;
- random loops;
- entropy;
- convergence;
- memory confidence;
- novelty;
- full EAC;
- oracle difficulty.

## Scratch size

0, 8, 16, 32, 64.

## Loop range

train 2–4 / 2–8 / 1–12 and test past the training range.

## Optimizer

AdamW vs Muon on a controlled model.

## Data

- natural only;
- + procedural;
- + 10/20/30/40% transformed synthetic;
- different quality thresholds.

## Scale

~30M, ~135M, optional ~350M.

## Seeds

Use multiple seeds for every central claim. Three is a minimum; 5+ at small scale is better.

---

# 17. Fair SOTA claims

“SOTA for parameter count” is easy to make misleading.

Every headline result should specify at least one of:

- **isoPARAMS** — same trainable/total parameter count;
- **isoFLOPs** — same training/inference computation;
- **isoTOKENS** — same amount of training data;
- **isoLATENCY** — same measured hardware latency;
- **isoMEMORY** — same peak VRAM / recurrent state budget.

Report:

- total parameters;
- active parameters per token;
- memory-state parameters/bytes;
- fast weights separately from frozen/slow weights;
- FLOPs per generated token at each recurrent depth;
- training tokens;
- training FLOPs;
- wall-clock;
- hardware;
- peak VRAM;
- tokens/sec;
- compile state / kernel versions.

The safe first paper claim should look like:

> “At matched trainable parameter count and training tokens, AURORA improves mutable-memory accuracy by X points and achieves a better accuracy-versus-inference-FLOP Pareto frontier than dense, naive-recurrent, and memory-only controls.”

That is stronger scientifically than an ambiguous “best 135M model.”

---

# 18. Systems implementation

The repository should have a correctness-first implementation and an optimized path.

## 18.1 Level 0 — reference PyTorch

Pure readable PyTorch modules:

```text
src/
  model/
    config.py
    embeddings.py
    attention.py
    ffn.py
    prelude.py
    recurrent_core.py
    scratch.py
    memory_fast.py
    memory_slow.py
    controller.py
    coda.py
    aura.py
  data/
  train/
  eval/
  kernels/
  tests/
```

No custom kernel until the reference model has numerical tests.

## 18.2 Level 1 — PyTorch optimized

Use:

- `torch.compile` where it actually helps;
- PyTorch SDPA;
- FlexAttention for custom masks/score modifications;
- FlashAttention-4 backend on supported Hopper/Blackwell setups;
- fused optimizer where safe;
- activation checkpointing;
- bf16;
- FSDP2 / distributed sharding when scale requires it.

## 18.3 Level 2 — Triton kernels

Good custom-kernel candidates:

1. fused RMSNorm + recurrent gate;
2. fast-memory read/update;
3. delta-rule update;
4. memory normalization;
5. controller feature reduction;
6. token-routing compaction if token-level recurrence is used.

For every kernel publish:

- reference implementation;
- correctness tests;
- forward/backward max error;
- microbenchmark;
- shape sweep;
- dtype sweep;
- speedup and VRAM.

## 18.4 Level 3 — CUDA/CuTe DSL

Only after profiling shows a kernel matters. A hand-written kernel that saves 2% end-to-end time is less impressive than a correct model result.

Potential systems paper/demo value comes from a fused recurrent-memory kernel if the memory update becomes a genuine bottleneck.

## 18.5 Profiling

Track:

- MFU;
- tokens/s;
- recurrent loops/s;
- memory read/write bandwidth;
- kernel occupancy;
- peak activation memory;
- optimizer state memory;
- compile time;
- CUDA graph compatibility;
- end-to-end generation latency.

---

# 19. Experiment management and reproducibility

Every run gets:

```text
run_id
Git commit
config hash
dataset version/hash
tokenizer hash
seed
hardware
CUDA version
PyTorch version
kernel version
start/end timestamps
training tokens
training FLOPs estimate
peak VRAM
checkpoint lineage
```

Use immutable YAML/TOML configs.

Save:

- periodic checkpoints;
- optimizer state for reproducibility;
- training/validation curves;
- recurrent diagnostics;
- memory diagnostics;
- system metrics.

Never overwrite an experiment directory.

## 19.1 Statistical hygiene

- predeclare the primary metrics before the final run;
- run central ablations with multiple seeds;
- report confidence intervals / bootstrap intervals;
- show negative results;
- do not select one lucky seed;
- keep public benchmark test sets out of architecture search;
- use a hidden procedural test generator seed bank that is frozen before final experiments.

---

# 20. Interactive demo

The demo should make the research contribution visible, not just provide a chat box.

## Panel 1 — conversation / task

User interacts with the model.

## Panel 2 — memory inspector

Show:

- memory items / latent slots in an interpretable projection where possible;
- new writes;
- overwrite/forget events;
- memory confidence;
- fast vs slow tier;
- source timestamp.

## Panel 3 — “thinking depth” meter

Live display:

```text
recurrent loops used: 3 / 16
estimated compute need: 0.18
stop reason: converged
```

Then show a difficult/contradictory query needing more loops.

## Panel 4 — familiarity experiment

Ask related questions repeatedly and chart compute decreasing while accuracy stays stable.

## Panel 5 — contradiction experiment

Change a fact. Visualize:

- conflict spike;
- additional recurrent loops;
- memory rewrite;
- later compute returning downward.

## Panel 6 — benchmark dashboard

Interactive plots:

- accuracy/FLOPs Pareto;
- accuracy/parameter Pareto;
- context length curves;
- forgetting curves;
- overthinking curves;
- loop histograms;
- ablations.

This demo is far more compelling to a researcher/recruiter than a generic chatbot UI.

---

# 21. Publication package

The goal is to make the project auditable enough that someone can reproduce the central figure.

## 21.1 Paper

Recommended structure:

1. Abstract
2. Introduction
3. Motivation: memory–compute coupling
4. Related work
5. AURORA architecture
6. Experience-Adaptive Controller
7. AURA-MemoryBench
8. Experimental setup
9. Main results
10. Ablations
11. Dynamics / interpretability analysis
12. Systems efficiency
13. Limitations
14. Safety / continual-learning risks
15. Reproducibility statement
16. Conclusion

## 21.2 Model release

Publish:

- base pretrained checkpoint;
- integrated memory checkpoint;
- final adaptive-compute checkpoint;
- small debug checkpoint;
- intermediate checkpoints if storage allows;
- exact configs.

## 21.3 Dataset

Publish:

- corpus manifests and scripts;
- procedural generator;
- benchmark seeds/specification;
- dataset card;
- source/license provenance;
- contamination report;
- quality-filter model/settings where legal.

If raw source redistribution is not allowed, publish deterministic download/process scripts rather than illegally re-hosting data.

## 21.4 Training code

Release:

- single-GPU reference trainer;
- distributed trainer;
- resume logic;
- eval hooks;
- profiler;
- exact CLI used for paper figures.

## 21.5 PyTorch / CUDA implementation

Release both readable and optimized paths. Make optimized kernels optional so reproducibility does not require a specific GPU.

## 21.6 Training curves

Release raw machine-readable logs, not screenshots only.

Must include:

- train loss;
- validation loss;
- gradient norm;
- learning rate;
- parameter norm;
- gate statistics;
- recurrent-state norm;
- recurrent-depth distribution;
- memory-write rate;
- memory-conflict rate;
- memory retrieval accuracy;
- stability metrics;
- tokens/sec;
- VRAM.

## 21.7 Ablations

Release the full table including failed variants.

## 21.8 Interactive demo

Host the research views described above and link directly to the paper/repo/model/dataset.

## 21.9 Research log

Maintain a public experiment diary after paper release:

```text
hypothesis → experiment → result → interpretation → next decision
```

A rigorous record of failed ideas is a strong signal that the work was actually understood rather than generated by an agent.

---

# 22. What a strong repository should look like

```text
AURORA/
├── README.md
├── LICENSE
├── CITATION.cff
├── pyproject.toml
├── configs/
│   ├── debug/
│   ├── 30m/
│   ├── 135m/
│   └── 350m/
├── src/aurora/
│   ├── model/
│   ├── memory/
│   ├── routing/
│   ├── data/
│   ├── training/
│   ├── eval/
│   └── kernels/
├── tests/
│   ├── unit/
│   ├── numerical/
│   └── integration/
├── benchmarks/
│   ├── aura_memorybench/
│   ├── long_context/
│   └── reasoning/
├── scripts/
│   ├── build_data/
│   ├── train/
│   ├── eval/
│   └── reproduce_paper/
├── notebooks/
│   └── analysis_only/
├── kernels/
│   ├── triton/
│   └── cuda_or_cute/
├── paper/
│   ├── manuscript/
│   ├── figures/
│   └── tables/
├── demo/
├── model_card.md
├── dataset_card.md
├── contamination_report.md
├── reproducibility.md
└── experiment_log.md
```

The README should have **one command that reproduces a tiny end-to-end run** and one command that reproduces each major table/figure from downloaded checkpoints.

---

# 23. Career-signal checklist

Current frontier-lab roles emphasize architecture work, model training, evaluations, synthetic data/environments, debugging, computational performance, training/inference infrastructure, and turning failures into systematic experiments. Build evidence for all of those.

The final profile should be able to state, truthfully:

- designed a novel recurrent/memory architecture;
- trained models from random initialization;
- built and versioned a training corpus;
- built a procedural continual-memory benchmark;
- implemented distributed training;
- implemented and benchmarked custom GPU kernels;
- designed controlled ablations;
- analyzed recurrent dynamics and failure modes;
- reproduced modern baselines;
- released model weights and data tooling;
- published a reproducible paper;
- built an interactive scientific demo;
- documented negative results and limitations.

What matters most is being able to explain every important choice yourself.

---

# 24. Failure conditions: when to kill or pivot an idea

Do not protect the hypothesis from the data.

## Kill naive recurrence if

At iso-FLOPs it consistently loses to unique-layer dense baselines and gating/scratch/time-conditioning do not close the gap.

## Kill learned halting if

A fixed-depth model at the same average FLOPs is more accurate or more stable.

## Kill persistent neural memory if

A simple external memory/RAG baseline is both more accurate and cheaper on the target tasks, unless the neural memory has a different demonstrated advantage such as multi-hop integration or constant-query latency.

## Kill multi-timescale consolidation if

It improves new-memory retention only by damaging old capabilities.

## Kill byte modeling from paper 1 if

It consumes a large fraction of engineering time without improving the memory/recurrence thesis.

## Kill custom kernels if

They do not materially move end-to-end performance.

A negative result with a clear mechanism can itself become publishable.

---

# 25. Concrete first 20 experiments

1. 10M dense Transformer sanity run.
2. 10M naive recurrent model at matched FLOPs.
3. Add prelude/coda.
4. Add loop-index conditioning.
5. Add Gated Recurrent Transformer-style anchor/gate.
6. Sweep scratch tokens {0,8,16,32,64} on algorithmic tasks.
7. Sweep training loop distributions.
8. Measure accuracy beyond trained loop count.
9. Add shortcut consistency.
10. Add convergence logging and early-stop oracle.
11. Test JSRR/STARS on one small reasoning task.
12. Implement linear fast memory on key/value stream.
13. Compare delta rule versus gradient-update memory.
14. Add learned write gate.
15. Add learned forget gate and contradiction dataset.
16. Compare online token update versus 128/256/512-token chunk update.
17. Integrate recurrent core + memory with fixed depth.
18. Train simple memory-confidence compute controller.
19. Train full EAC and compare to fixed-mean-FLOP baseline.
20. Run “familiar → cheap; contradiction → expensive; relearn → cheap again” end-to-end demonstration.

Do not begin the 135M main run before experiments 1–20 produce a coherent story.

---

# 26. Decision gates before spending serious compute

## Gate A — recurrence

Must show at least one of:

- better accuracy at matched parameters and competitive FLOPs;
- same accuracy with fewer parameters/VRAM;
- clear test-time scaling with additional loops.

## Gate B — memory

Must show:

- reliable updates;
- bounded interference;
- better long-stream accuracy than recurrence-only;
- measurable advantage over raw context at some length/latency regime.

## Gate C — adaptive compute

Must show:

- positive correlation between objective difficulty and allocated depth;
- reduced mean FLOPs at matched accuracy versus fixed depth;
- no shallow-halting collapse.

## Gate D — joint hypothesis

Must show:

- memory familiarity decreases compute;
- contradictions increase compute;
- after memory correction, compute decreases again.

Only after Gate D should the project claim an integrated memory-compute learning mechanism.

---

# 27. Risks and countermeasures

| Risk | Why it matters | Countermeasure |
|---|---|---|
| Recurrence overthinks | accuracy can collapse beyond trained depth | random depths, time conditioning, consistency, convergence controls, JSRR branch |
| Shared core loses specialization | same transform repeated | prelude/coda, anchor, loop modulation, routers |
| Hidden-state overload | transient + persistent info interfere | scratch state + explicit memory/state highways |
| Halting collapses shallow | controller finds cheap local optimum | deep-start bias, warm compute penalty, min loops |
| Memory writes noise | surprise != useful novelty | future-utility supervision, learnable novelty, write sparsity |
| Catastrophic forgetting | fast updates overwrite prior facts | delta/retention gates, tiers, replay/consolidation |
| Synthetic-data collapse/narrowness | generated text loses diversity | natural-majority mixtures, diversity metrics, provenance |
| Generic benchmarks remain weak | small model lacks world knowledge | make claim about capability/efficiency frontier, not general IQ |
| New paper scoops mechanism | field is moving rapidly | monthly literature audit and preprint early once results solid |
| Too many innovations | impossible attribution | staged versions and hard ablation gates |
| Kernel work distracts science | systems optimization consumes time | profile first; optimize only top bottlenecks |
| Benchmark contamination | invalid claims | exact contamination scanner + private procedural seeds |
| Name collision | AURA already exists | rename before public release |

---

# 28. Literature map and implementation takeaways

This is a curated high-relevance map, **not a claim that every paper ever published has been enumerated**. The field changes weekly. Re-run literature search before architecture freeze and again before paper submission.

## Neural memory / test-time learning

1. **Titans: Learning to Memorize at Test Time** — Behrouz, Zhong, Mirrokni. arXiv:2501.00663.  
   https://arxiv.org/abs/2501.00663  
   Takeaway: neural long-term memory + attention; surprise-driven memory; multiple integration variants.

2. **ATLAS: Learning to Optimally Memorize the Context at Test Time**. arXiv:2505.23735.  
   https://arxiv.org/abs/2505.23735  
   Takeaway: memory capacity, context-level update, expressive memory management.

3. **It's All Connected: A Journey Through Test-Time Memorization, Attentional Bias, Retention, and Online Optimization (MIRAS)**. arXiv:2504.13173.  
   https://arxiv.org/abs/2504.13173  
   Takeaway: architecture/objective/retention/learning-rule design matrix.

4. **Nested Learning: The Illusion of Deep Learning Architectures**. arXiv:2512.24695.  
   https://arxiv.org/abs/2512.24695  
   Takeaway: optimizers as memory, self-modifying modules, Continuum Memory System, HOPE.

5. **Language Models Need Sleep: Learning to Self-Modify and Consolidate Memories**. arXiv:2606.03979.  
   https://arxiv.org/abs/2606.03979  
   Takeaway: online + offline consolidation; fast/slow tiers; self-distillation/dreaming.

6. **Mela: Test-Time Memory Consolidation based on Transformation Hypothesis**. arXiv:2605.10537.  
   https://arxiv.org/abs/2605.10537  
   Takeaway: hierarchical episodic/gist memory and distributed decoder memory.

7. **MoNe: Modular Neural Memory for Efficient Long Context Inference**. arXiv:2608.17616.  
   https://arxiv.org/abs/2608.17616  
   Takeaway: layer-localized fast-weight updates, constant query dependence on stored context length, reusable memory state.

8. **Learning to (Learn at Test Time): RNNs with Expressive Hidden States**. arXiv:2407.04620.  
   https://arxiv.org/abs/2407.04620  
   Takeaway: hidden state as learnable model; TTT-Linear/TTT-MLP.

9. **End-to-End Test-Time Training for Long Context**. arXiv:2512.23675.  
   https://arxiv.org/abs/2512.23675  
   Takeaway: meta-learned TTT in standard language-model setting; constant recurrent query behavior.

10. **Test-Time Training Done Right**. arXiv:2505.23884.  
    https://arxiv.org/abs/2505.23884

11. **In-Place Test-Time Training**. arXiv:2604.06169.  
    https://arxiv.org/abs/2604.06169

12. **Self-Guided Test-Time Training**. arXiv:2607.09415.  
    https://arxiv.org/abs/2607.09415

13. **Reinforced Fast Weights with Next-Sequence Prediction (REFINE)**. arXiv:2602.16704.  
    https://arxiv.org/abs/2602.16704  
    Takeaway: fast weights may need a future-oriented objective rather than plain NTP.

14. **Fast-weight Product Key Memory**. arXiv:2601.00671.  
    https://arxiv.org/abs/2601.00671

15. **Memory Layers at Scale**. arXiv:2412.09764.  
    https://arxiv.org/abs/2412.09764  
    Takeaway: lookup-like memory can add factual capacity with low active compute.

16. **Memory³: Language Modeling with Explicit Memory**. arXiv:2407.01178.  
    https://arxiv.org/abs/2407.01178

17. **Associative Recurrent Memory Transformer (ARMT)**. arXiv:2607.11614.  
    https://arxiv.org/abs/2607.11614

18. **Panini: Continual Learning in Token Space via Structured Memory**. arXiv:2602.15156.  
    https://arxiv.org/abs/2602.15156

19. **Learning Fast and Slow**. arXiv:2605.12484.  
    https://arxiv.org/abs/2605.12484

20. **Generalizing Neural Memory to be Controllable in Natural Language**. arXiv:2602.23201.  
    https://arxiv.org/abs/2602.23201

21. **Mixture of Chapters: Scaling Learnt Memory in Transformers**. arXiv:2603.21096.  
    https://arxiv.org/abs/2603.21096

22. **Learning to Forget Attention: Memory Consolidation for Adaptive Compute Reduction**. arXiv:2602.12204.  
    https://arxiv.org/abs/2602.12204  
    Takeaway: speculative/high-risk evidence that consolidation may reduce future attention demand.

23. **Can Past Experience Accelerate LLM Reasoning?** arXiv:2505.20643.  
    https://arxiv.org/abs/2505.20643  
    Takeaway: explicit connection between memory/familiarity and adaptive reasoning cost; essential prior art for EAC novelty analysis.

24. **AURA: Action-Gated Memory for Robot Policies at Constant VRAM**. arXiv:2606.02775.  
    https://arxiv.org/abs/2606.02775  
    Takeaway: utility-gated write mechanism; also the reason to rename this project publicly.

## Recurrent depth / latent reasoning / dynamic compute

25. **Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach (Huginn)**. arXiv:2502.05171.  
    https://arxiv.org/abs/2502.05171

26. **Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation**. arXiv:2507.10524.  
    https://arxiv.org/abs/2507.10524

27. **Mixture-of-Depths: Dynamically Allocating Compute in Transformer-Based Language Models**. arXiv:2404.02258.  
    https://arxiv.org/abs/2404.02258

28. **PonderNet: Learning to Ponder**. arXiv:2107.05407.  
    https://arxiv.org/abs/2107.05407

29. **Universal Transformers**. arXiv:1807.03819.  
    https://arxiv.org/abs/1807.03819

30. **Gated Recurrent Transformers / RecurrentGPT: Expressive Depth through Recurrent Modulation in Transformers**. arXiv:2608.15062.  
    https://arxiv.org/abs/2608.15062  
    Takeaway: prelude/core/coda, input anchor, recurrent gating, loop noise; very recent and highly relevant.

31. **LoopFormer: Elastic-Depth Looped Transformers for Latent Reasoning via Shortcut Modulation**. arXiv:2602.11451.  
    https://arxiv.org/abs/2602.11451  
    Takeaway: normalized time + Δt, self-distilled trajectory consistency, budget-controlled depth.

32. **MeSH: Memory-as-State-Highways for Recursive Transformers**. arXiv:2510.07739.  
    https://arxiv.org/abs/2510.07739  
    Takeaway: recursive models suffer undifferentiated computation and information overload; externalized state highways + routers.

33. **Universal Transformers Need Memory: Depth-State Trade-offs in Adaptive Recursive Reasoning**. arXiv:2604.21999.  
    https://arxiv.org/abs/2604.21999  
    Takeaway: scratch memory can be necessary; ACT router initialization traps; deep-start and lambda warmup.

34. **Stabilizing Recurrent Dynamics for Test-Time Scalable Latent Reasoning in Looped Language Models (STARS)**. arXiv:2605.26733.  
    https://arxiv.org/abs/2605.26733  
    Takeaway: fixed-point stability, Jacobian spectral-radius regularization.

35. **Attractor Models: Solve the Loop**. arXiv:2605.12466.  
    https://arxiv.org/abs/2605.12466  
    Takeaway: iterative attractor refinement, convergence-based depth, implicit differentiation.

36. **Loop, Think, & Generalize**. arXiv:2604.07822.  
    https://arxiv.org/abs/2604.07822  
    Takeaway: recurrence can improve systematic/depth generalization but can overthink.

37. **Tiny Autoregressive Recursive Models**. arXiv:2603.08082.  
    https://arxiv.org/abs/2603.08082  
    Takeaway: valuable negative evidence; recurrence is not automatically beneficial.

38. **Looping Back to Move Forward: Relaxed Recursive Transformers**. arXiv:2602.09080.  
    https://arxiv.org/abs/2602.09080

39. **Inner Thinking Transformer**. arXiv:2502.13842.  
    https://arxiv.org/abs/2502.13842

40. **Linear-Time Looped Transformers (LT2)**. arXiv:2605.20670.  
    https://arxiv.org/abs/2605.20670

41. **ReLIT**. arXiv:2608.08113.  
    https://arxiv.org/abs/2608.08113  
    Takeaway: repeated computation cannot recover information that the initial representation failed to preserve; supports input anchors/state highways.

42. **LOTUS: Bridging Latent and Explicit Reasoning**. arXiv:2606.31779.  
    https://arxiv.org/abs/2606.31779

43. **Depth-Recurrent Attention Mixtures**. arXiv:2601.21582.  
    https://arxiv.org/abs/2601.21582

44. **Adaptive token routing / token-wise skip mechanisms**. See arXiv:2605.05222.  
    https://arxiv.org/abs/2605.05222

## Efficient sequence modeling / tokenization

45. **Mamba-2 / Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality**. arXiv:2405.21060.  
    https://arxiv.org/abs/2405.21060

46. **DeltaNet / Parallelizing Linear Transformers with the Delta Rule**. arXiv:2406.06484.  
    https://arxiv.org/abs/2406.06484

47. **xLSTM: Extended Long Short-Term Memory**. arXiv:2405.04517.  
    https://arxiv.org/abs/2405.04517

48. **Differential Transformer**. arXiv:2410.05258.  
    https://arxiv.org/abs/2410.05258

49. **Byte Latent Transformer: Patches Scale Better Than Tokens**. arXiv:2412.09871.  
    https://arxiv.org/abs/2412.09871

50. **Fast/Block-Diffusion BLT work**. arXiv:2605.08044.  
    https://arxiv.org/abs/2605.08044

51. **Better & Faster Large Language Models via Multi-token Prediction**. arXiv:2404.19737.  
    https://arxiv.org/abs/2404.19737

52. **Pre-training Curriculum for Multi-token Prediction**. arXiv:2505.22757.  
    https://arxiv.org/abs/2505.22757

53. **MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases**. arXiv:2402.14905.  
    https://arxiv.org/abs/2402.14905

54. **OpenELM: An Efficient Language Model Family with Open-source Training and Inference Framework**. arXiv:2404.14619.  
    https://arxiv.org/abs/2404.14619

55. **SmolLM2: When Smol Goes Big — Data-Centric Training of a Small Language Model**. arXiv:2502.02737.  
    https://arxiv.org/abs/2502.02737

## Optimization / training stability

56. **Muon is Scalable for LLM Training**. arXiv:2502.16982.  
    https://arxiv.org/abs/2502.16982

57. **Variance-Adaptive Muon**. arXiv:2601.14603.  
    https://arxiv.org/abs/2601.14603

58. **Muon+**. arXiv:2602.21545.  
    https://arxiv.org/abs/2602.21545

59. **SOAP: Improving and Stabilizing Shampoo using Adam**. arXiv:2409.11321.  
    https://arxiv.org/abs/2409.11321

60. **COSMOS**. arXiv:2502.17410.  
    https://arxiv.org/abs/2502.17410

61. **u-µP: The Unit-Scaled Maximal Update Parametrization**. arXiv:2407.17465.  
    https://arxiv.org/abs/2407.17465

62. **Sparse µP**. arXiv:2405.15743.  
    https://arxiv.org/abs/2405.15743

63. **Root Mean Square Layer Normalization**. arXiv:1910.07467.  
    https://arxiv.org/abs/1910.07467

64. **GLU Variants Improve Transformer**. arXiv:2002.05202.  
    https://arxiv.org/abs/2002.05202

## Data

65. **DataComp-LM: In Search of the Next Generation of Training Sets for Language Models**. arXiv:2406.11794.  
    https://arxiv.org/abs/2406.11794

66. **The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale**. arXiv:2406.17557.  
    https://arxiv.org/abs/2406.17557

67. **Textbooks Are All You Need**. arXiv:2306.11644.  
    https://arxiv.org/abs/2306.11644

68. **TinyStories: How Small Can Language Models Be and Still Speak Coherent English?**. arXiv:2305.07759.  
    https://arxiv.org/abs/2305.07759

69. **Demystifying Synthetic Data in LLM Pre-training: A Systematic Study of Scaling Laws, Benefits, and Pitfalls**. arXiv:2510.01631, version dated 2026-08-24.  
    https://arxiv.org/abs/2510.01631

70. **Scaling Data Mixing via Model Merging / DeMix**. arXiv:2602.00747.  
    https://arxiv.org/abs/2602.00747

71. **Scaling Laws for Mixture Pretraining Under Data Constraints**. arXiv:2605.12715.  
    https://arxiv.org/abs/2605.12715

72. **SynPro**. arXiv:2605.17849.  
    https://arxiv.org/abs/2605.17849

73. **Darwin-CC / autonomous data-curation evolution**. arXiv:2603.14420.  
    https://arxiv.org/abs/2603.14420

74. **L20-Edu-135M: An Auditable Single-GPU Study of Data-Efficient Small Language Modeling**. arXiv:2606.22189.  
    https://arxiv.org/abs/2606.22189  
    Takeaway: useful reality check for what a reproducible 135M student-scale run can and cannot achieve.

## Long-context / memory evaluation

75. **RULER: What's the Real Context Size of Your Long-Context Language Models?**. arXiv:2404.06654.  
    https://arxiv.org/abs/2404.06654

76. **BABILong**. arXiv:2406.10149.  
    https://arxiv.org/abs/2406.10149

77. **LongBench v2: Towards Deeper Understanding and Reasoning on Realistic Long-context Multitasks**. arXiv:2412.15204.  
    https://arxiv.org/abs/2412.15204

78. **LongBench Pro**. arXiv:2601.02872.  
    https://arxiv.org/abs/2601.02872

79. **InfiniteBench**. arXiv:2402.13718.  
    https://arxiv.org/abs/2402.13718

80. **LOCA-bench**. arXiv:2602.07962.  
    https://arxiv.org/abs/2602.07962

81. **A Benchmark for Memory and Continual Learning in LLM Systems**. arXiv:2510.17281.  
    https://arxiv.org/abs/2510.17281

82. **CL-Bench / Continual Learning in Real-World Stateful Environments**. arXiv:2606.05661.  
    https://arxiv.org/abs/2606.05661

## Systems

83. **FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling**. arXiv:2603.05451.  
    https://arxiv.org/abs/2603.05451

84. **PyTorch FlexAttention + FlashAttention-4** — PyTorch engineering material, 2026.  
    https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/

---

# 29. Monthly literature-watch queries

Because this field is changing too fast for a one-time bibliography, schedule repeated searches for:

```text
"recurrent depth" language model
"looped transformer" latent reasoning
"adaptive computation" language model
"test-time training" long context
"neural memory" language model
"fast weights" transformer
"memory consolidation" language model
"continual learning" language model benchmark
"dynamic depth" transformer
"fixed point" recurrent language model
"attractor" language model reasoning
"memory routing" recurrent transformer
"state highway" recursive transformer
"online optimization" sequence model
"learned optimizer" sequence model memory
"test-time memorization"
"byte latent transformer"
"small language model" 135M pretraining
"Muon" language model optimizer
"FlashAttention" recurrent memory kernel
```

For each new paper record:

- publication/update date;
- exact claim;
- model scale;
- training tokens;
- baselines;
- whether code exists;
- whether reproduced independently;
- what AURA component it affects;
- whether it threatens novelty.

---

# 30. Definition of success

A successful first paper does **not** need to beat every 135M model on every benchmark.

It should prove a clean scientific result such as:

1. a small recurrent model learns useful latent test-time scaling;
2. a bounded neural memory acquires and revises facts over long streams;
3. memory familiarity reduces the model's recurrent compute;
4. contradictions trigger extra compute and memory correction;
5. the resulting model lies on a better accuracy/FLOP/parameter frontier than matched controls;
6. the result survives multiple seeds, scale-up, and independent benchmarks;
7. every key claim is reproducible from released code/data/configs.

If those seven conditions are satisfied, this is a serious research project even if a much larger Transformer still has better general knowledge.

---

# 31. Immediate implementation order

The build agent should receive tasks in this order:

```text
01. repository skeleton + config system
02. tokenizer/data-loader + deterministic shuffling
03. modern dense Transformer baseline
04. training/eval/checkpoint infrastructure
05. recurrent prelude/core/coda model
06. recurrence instrumentation
07. gated anchor + loop/time modulation
08. scratch/state-highway module
09. shortcut-consistency training
10. procedural mutable-world generator
11. fast neural-memory reference implementation
12. write/forget gates
13. AURA-MemoryBench evaluator
14. integrated fixed-depth recurrent+memory model
15. EAC signal extraction
16. fixed/heuristic EAC controllers
17. learned EAC + compute regularization
18. long-stream memory training
19. 30M ablation sweep
20. architecture freeze
21. 135M main runs
22. baseline reproductions
23. custom kernel profiling/optimization
24. final benchmark suite
25. paper figures/tables
26. model/dataset cards
27. interactive scientific demo
28. public reproducibility release
```

**Rule:** the coding agent may implement, test, profile, and suggest hypotheses. It must never silently choose a research conclusion. Every architectural decision that survives into the paper should have a written hypothesis, an experiment, and a result attached to it.

---

# 32. Final recommendation

Build **the smallest model that can falsify the idea first**.

The order of scientific importance is:

```text
correct controlled experiment
    > reproducible baseline
    > evidence of mechanism
    > scale-up
    > benchmark headline
    > custom CUDA
    > flashy demo
```

The project becomes impressive when the paper can say not merely **“we built a new model”**, but:

> **“We identified a measurable failure mode in recurrent language models, proposed a memory-aware adaptive-compute mechanism, showed why it works through controlled ablations and recurrent-dynamics analysis, and released the model, data generator, benchmark, training stack, kernels, checkpoints, and full experiment record.”**

That is the standard this plan is designed to reach.
