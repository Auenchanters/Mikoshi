# Paper Notes — Foundation Phase

## What exists

The repository has a tested, reproducible dense causal Transformer control and
the experiment infrastructure needed to falsify later recurrent/memory
hypotheses. The initial production-BPE sanity run learns a tiny repeated corpus,
and its CPU fp32 checkpoint reload is exact.

## Claims supported now

- The dense control executes finite forward/backward passes.
- Causal masking prevents future-token influence in the tested reference path.
- The tiny repeated-corpus loss decreases.
- BPE artifact training is deterministic for a fixed ordered corpus and fixed
  preprocessing/settings in the tested environment.
- CPU fp32 checkpoint/resume restores the exact tested training trajectory.
- Experiment artifacts record config, seed, commit, dataset/tokenizer identity,
  optimizer/schedule, precision, hardware, tokens, wall time, estimated FLOPs,
  losses, throughput, and checkpoint lineage.

## Claims not supported

- AURORA works.
- Recurrence improves reasoning or test-time scaling.
- Persistent neural memory learns, updates, or forgets facts.
- Familiarity reduces compute or contradiction increases compute.
- The model is competitive with any external language model.
- The tiny evaluation loss estimates generalization.
- CUDA, bf16/fp16, or distributed replay is deterministic.

## Important limitations of EXP-0001

- 65,328 parameters and only 5,120 training tokens;
- generated repeated corpus with evaluation from the same pattern family;
- CPU fp32 only;
- no external benchmark;
- no multiple-seed scientific comparison;
- generation is whitespace-heavy and not a qualitative success;
- FLOPs are an explicit matmul estimate, not a profiler measurement;
- VRAM is not applicable on the CPU host.

## Next paper-relevant work

Task 05 should introduce the recurrent prelude/core/coda model with fixed loop
budgets and compare it against this unchanged dense control. Parameter and FLOP
matching, loop-depth extrapolation, recurrence diagnostics, and multiple seeds
must precede any claim. Neural memory and EAC remain later phases.
