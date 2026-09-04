# MiniBug-RL experiment record

## Methodology and how to use these reports

MiniBug-RL tests whether executable hidden-test rewards can improve a tiny code model through a hardware-bounded GRPO/LoRA run. The experiment flow is: audit a frozen 60-task curriculum, establish an untouched validation baseline, prove the reward/update/save path with two smoke steps, train an independent adapter for 100 steps, select on validation, lock the budget, open internal and external final tests once, export, and publish the verified result.

Read the reports in numeric order. Experiment 001 is a disclosed diagnostic that caused data and protocol corrections and is excluded from quality claims. Experiments 002–004 describe the clean run from the GitHub-merged source. For downstream analysis, use the JSON files in [`evidence/`](evidence/) rather than transcribing rounded Markdown values.

The authoritative layers are:

1. Raw local run records under `logs/20260904T021845Z-minibug-rl-main/` and trainer state under `artifacts/runs/`; these contain unrounded candidate-level observations but are intentionally not committed.
2. [`evidence/run-summary.json`](evidence/run-summary.json), which records exact aggregate values, artifact SHA-256 digests, identities, definitions, and limitations.
3. [`evidence/task-scores.json`](evidence/task-scores.json), which records every internal paired greedy score and a compact lossless representation of every external binary score.
4. The experiment narratives, which interpret rather than replace the machine-readable evidence.

The resulting model is [`BurnyCoder/qwen2.5-coder-0.5b-swe-rl`](https://huggingface.co/BurnyCoder/qwen2.5-coder-0.5b-swe-rl) at immutable revision [`5b6e22a4c6c01bec95d10e93a0fc78666eb9c543`](https://huggingface.co/BurnyCoder/qwen2.5-coder-0.5b-swe-rl/tree/5b6e22a4c6c01bec95d10e93a0fc78666eb9c543). The code used for the clean experiment was merged on GitHub as [`7027bc55baecc00fad51cbe2b8f030dca2c91c1e`](https://github.com/BurnyCoder/llm-rl-software-engineering/commit/7027bc55baecc00fad51cbe2b8f030dca2c91c1e) before the run began.

## Experiment index

| Record | Role | Decision or result |
|---|---|---|
| [001 — pre-fix diagnostic smoke](experiments/001-pre-fix-smoke.md) | Hardware and pipeline reconnaissance | Training path worked, but curriculum and protocol review required a clean restart. |
| [002 — corrected smoke](experiments/002-fixed-smoke.md) | Prespecified systems gate | Two steps changed and reloaded the adapter within 1.61 GB peak VRAM; proceed. |
| [003 — main 100-step run](experiments/003-main-100-step.md) | Primary training experiment | Validation learning/reliability gate passed; lock checkpoint 100 without extension. |
| [004 — locked evaluation](experiments/004-locked-evaluation.md) | Untouched internal and frozen external evaluation | Stronger output-format reliability, but correctness differences remain uncertain. |

## Evidence index

- [`run-summary.json`](evidence/run-summary.json): run/model/data identities, hardware, dependencies, training diagnostics, aggregate evaluations, caveats, and raw-artifact digests.
- [`task-scores.json`](evidence/task-scores.json): task pairing, internal score maps, exact changes, external passed/timeout sets, and paired intervals.
- [`resolved-config.json`](evidence/resolved-config.json): byte-identical copy of the completed run's resolved configuration; `run-summary.json` records its file and canonical hashes.
- [`verification.json`](evidence/verification.json): independently executed evidence checks and explicitly unclaimed final-document checks.

## Result in one paragraph

The selected 100-step, rank-16 LoRA adapter trained 8,798,208 parameters—1.7497% of the reported 502,830,976 total—on an RTX 5070 Laptop GPU with 1.85 GB peak training VRAM. Relative to the base, it moved internal greedy full solves from 5/12 to 6/12 and HumanEvalFix pass@1 from 37/164 to 38/164. Internal invalid outputs fell from 19/60 to 1/60, but mean greedy hidden-test correctness moved slightly down from 0.6083 to 0.6042 and both final paired 95% intervals include zero. The reliable claim is a functioning consumer-hardware RL pipeline and substantially better output-format compliance, not conclusive general software-repair improvement.

## Verification boundary

The aggregate values and paired intervals were recomputed independently from candidate records, task key/order integrity was checked, raw artifact hashes were captured, and the published snapshot was re-downloaded for generation. The final [`verification.json`](evidence/verification.json) also records the 74-test repository run, static checks, documentation audit, exact public-model inference examples, and warning-free eight-page paper build.

To validate this report bundle after cloning, first parse the committed JSON:

```bash
jq empty reports/evidence/*.json
```

Then follow the repository's main README for environment setup and the full test command. Exact training reproduction requires the pinned base revision, curriculum and manifest hashes, Docker image construction, seed, and configuration recorded here. GPU kernels and sampled decoding can still vary across hardware/software stacks, so reproduction means repeating the protocol and reporting the new run rather than assuming byte-identical generations.

## Source references

- [TRL `GRPOTrainer`](https://huggingface.co/docs/trl/grpo_trainer)
- [PEFT LoRA concepts](https://huggingface.co/docs/peft/main/conceptual_guides/lora)
- [Trackio logging API](https://huggingface.co/docs/trackio/api#trackio.log)
- [HumanEvalPack dataset revision](https://huggingface.co/datasets/bigcode/humanevalpack/tree/9a41762f73a8cb23bb5811b73d5aab164efcf378)
- [Codex pass@k evaluation methodology](https://arxiv.org/abs/2107.03374)
