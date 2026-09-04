# MiniBug-RL experiment record

## Methodology and how to use these reports

MiniBug-RL tests whether executable hidden-test rewards can improve a tiny code model through a hardware-bounded GRPO/LoRA run. The experiment flow was: audit a fixed 60-task curriculum, establish a base validation baseline, prove the reward/update/save path with two smoke steps, train an independent adapter for 100 steps, select on validation, lock the budget, score each final benchmark once per compared system, export, publish, and independently verify the resulting artifact. Internal final task definitions were prepared before the lock, but no final model generations or outcomes entered selection; HumanEvalPack loading and model scoring occurred afterward. The exact ordering is checked in [`evidence/verification.json`](evidence/verification.json).

Read the reports in numeric order. Experiment 001 is a disclosed diagnostic that caused data and protocol corrections and is excluded from quality claims. Experiments 002–004 describe the authoritative completed run from the GitHub-merged source. For downstream analysis, use the JSON files in [`evidence/`](evidence/) rather than transcribing rounded Markdown values.

The authoritative layers are:

1. Raw local run records under `logs/20260904T021845Z-minibug-rl-main/` and trainer state under `artifacts/runs/`; these contain unrounded candidate-level observations but are intentionally not committed.
2. [`evidence/run-summary.json`](evidence/run-summary.json), which records exact aggregate values, artifact SHA-256 digests, identities, definitions, and limitations.
3. [`evidence/task-scores.json`](evidence/task-scores.json), which records every internal paired greedy score and a compact lossless representation of every external binary score.
4. The experiment narratives, which interpret rather than replace the machine-readable evidence.

The resulting model is [`BurnyCoder/qwen2.5-coder-0.5b-swe-rl`](https://huggingface.co/BurnyCoder/qwen2.5-coder-0.5b-swe-rl). Its experiment artifacts originated at immutable revision [`5b6e22a4…`](https://huggingface.co/BurnyCoder/qwen2.5-coder-0.5b-swe-rl/tree/5b6e22a4c6c01bec95d10e93a0fc78666eb9c543), and its README-only factual correction is at [`8dc6fa7d…`](https://huggingface.co/BurnyCoder/qwen2.5-coder-0.5b-swe-rl/tree/8dc6fa7df8a09e157e3a5be1ec17b5d8b8aa4f43). The code used for the authoritative completed experiment was merged on GitHub as [`7027bc55baecc00fad51cbe2b8f030dca2c91c1e`](https://github.com/BurnyCoder/llm-rl-software-engineering/commit/7027bc55baecc00fad51cbe2b8f030dca2c91c1e) before the run began.

## Experiment index

| Record | Role | Decision or result |
|---|---|---|
| [001 — pre-fix diagnostic smoke](experiments/001-pre-fix-smoke.md) | Hardware and pipeline reconnaissance | Training path worked, but curriculum and protocol review required a clean restart. |
| [002 — corrected smoke](experiments/002-fixed-smoke.md) | Source-committed systems gate | Two steps changed and reloaded the adapter within 1.61 GB peak VRAM; proceed. |
| [003 — main 100-step run](experiments/003-main-100-step.md) | Primary training experiment | Validation learning/reliability gate passed; lock checkpoint 100 without extension. |
| [004 — locked evaluation](experiments/004-locked-evaluation.md) | Selection-held-out internal and frozen external evaluation | Fewer invalid outputs were observed, but correctness differences remain uncertain. |

## Evidence index

- [`run-summary.json`](evidence/run-summary.json): run/model/data identities, hardware, dependencies, training diagnostics, aggregate evaluations, caveats, and raw-artifact digests.
- [`task-scores.json`](evidence/task-scores.json): task pairing, internal score maps, exact changes, external passed/timeout sets, and paired intervals.
- [`resolved-config.json`](evidence/resolved-config.json): byte-identical copy of the completed run's resolved configuration; `run-summary.json` records its file and canonical hashes.
- [`verification.json`](evidence/verification.json): independently executed empirical checks plus an explicit record that its original documentation certification is superseded and its original PDF check is historical.
- [`documentation-audit.json`](evidence/documentation-audit.json): current source/link checks, complete test pass, corrected PDF identity, Hugging Face card-only correction, exact file comparison, and post-correction inference.

## Result in one paragraph

The selected 100-step, rank-16 LoRA adapter trained 8,798,208 parameters—1.7497% of the reported 502,830,976 total—on the recorded RTX 5070 Laptop GPU with 1.85 GB peak training VRAM. Relative to the base, it moved internal greedy full solves from 5/12 to 6/12 and HumanEvalFix pass@1 from 37/164 to 38/164. Internal invalid outputs fell from 19/60 to 1/60, but mean greedy hidden-test correctness moved slightly down from 0.6083 to 0.6042 and both final paired percentile-bootstrap 95% intervals include zero. The supported claim is a functioning small-hardware RL pipeline and an observed reduction in invalid output formatting, not conclusive general software-repair improvement. Exact measurements and qualifications are in [`evidence/run-summary.json`](evidence/run-summary.json).

## Verification boundary

The aggregate values and paired intervals were recomputed independently from candidate records, task key/order integrity was checked, raw artifact hashes were captured, and the published snapshot was re-downloaded for generation. A later post-publication check byte-compared both Safetensors files; that check was not part of the original publication gate. [`verification.json`](evidence/verification.json) preserves the 74-test repository run, static checks, exact public-model inference examples, and the historical eight-page paper build while marking its incomplete documentation check as superseded. The non-destructive follow-up is recorded separately in [`documentation-audit.json`](evidence/documentation-audit.json).

To validate this report bundle after cloning, first parse the committed JSON:

```bash
jq empty reports/evidence/*.json
```

Then follow the repository's main README for environment setup and the full test command. Exact training reproduction requires the pinned base revision, curriculum and manifest hashes, Docker image construction, seed, and configuration recorded here. GPU kernels and sampled decoding can still vary across hardware/software stacks, so reproduction means repeating the protocol and reporting the new run rather than assuming byte-identical generations.

## Source references

- [TRL 1.12.0 `GRPOTrainer`](https://huggingface.co/docs/trl/v1.12.0/en/grpo_trainer)
- [PEFT 0.20.0 LoRA API](https://huggingface.co/docs/peft/v0.20.0/en/package_reference/lora)
- [Trackio 0.37.0 logging API](https://huggingface.co/docs/trackio/v0.37.0/en/api#trackio.log)
- [Pinned Qwen2.5-Coder-0.5B-Instruct card](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct/blob/ea3f2471cf1b1f0db85067f1ef93848e38e88c25/README.md)
- [Pinned DebugArena card](https://huggingface.co/BharathVikas/debugarena/blob/e4be134207727d684b8a7edfbc7934159893617b/README.md)
- [Pinned HumanEvalPack dataset card](https://huggingface.co/datasets/bigcode/humanevalpack/blob/9a41762f73a8cb23bb5811b73d5aab164efcf378/README.md)
- [Pinned BigCode HumanEvalPack harness](https://github.com/bigcode-project/bigcode-evaluation-harness/blob/8fc5bae6479c4fbbb28c3f8b644f6a15b3f3b5bd/bigcode_eval/tasks/humanevalpack.py)
- [Docker CLI 29.4.1 `run` reference source](https://github.com/docker/cli/blob/055a478ea9010a19d0d4674c0d0e87ade37a4223/docs/reference/commandline/run.md)
- [Codex pass@k evaluation methodology](https://arxiv.org/abs/2107.03374)
