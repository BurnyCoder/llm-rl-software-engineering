# Experiment 003: main 100-step GRPO run

**Classification:** primary training experiment

**Run:** `20260904T021845Z-minibug-rl-main`

**Training interval:** 2026-09-04 02:25:21–02:44:57 UTC

**Outcome:** checkpoint 100 passed the validation learning/reliability gate and was locked without extension

## Domain, question, and hypothesis

The domain is outcome-supervised reinforcement learning for small language-model, single-function Python repair. Each example asks the model to replace one short buggy function. Four completions form a GRPO comparison group. Every completion receives the same composite reward definition: host-side parser/policy rejects receive fixed penalties, deterministic cache hits reuse a prior breakdown, and valid cache misses obtain hidden-test results from a fresh Docker container.

The primary question was: can a 0.5B instruction-tuned code model acquire measurable repair behavior in 100 local GRPO steps on an 8 GB laptop GPU without persistent reward collapse or unusable generation truncation?

Before final model scoring, the operational hypothesis was defined by the validation gate pre-specified in producing commit [`7027bc5`](https://github.com/BurnyCoder/llm-rl-software-engineering/commit/7027bc55baecc00fad51cbe2b8f030dca2c91c1e): relative to the base, the selected adapter must either improve mean paired greedy hidden-test fraction by at least 0.05 or gain at least one net greedy full solve, and its total candidate failure rate must not be worse. Internal final task definitions had already been structurally checked, but no final model generations, scores, or outcomes—and no HumanEvalPack rows—entered this hypothesis test or the training-budget decision.

## Existing information and design choices

The versioned [TRL 1.12.0 GRPO documentation](https://huggingface.co/docs/trl/v1.12.0/en/grpo_trainer) provides the `GRPOConfig`/`GRPOTrainer`, grouped generations, reward-function, PEFT, and sampling interfaces used here. [PEFT 0.20.0's LoRA API](https://huggingface.co/docs/peft/v0.20.0/en/package_reference/lora) documents the low-rank adapter configuration used to freeze base weights and limit trainable parameters.

The experiment used the 0.49B-parameter base named in the pinned [DebugArena card](https://huggingface.co/BharathVikas/debugarena/blob/e4be134207727d684b8a7edfbc7934159893617b/README.md), [`Qwen/Qwen2.5-Coder-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct/blob/ea3f2471cf1b1f0db85067f1ef93848e38e88c25/README.md), but used the installed standard Transformers/PEFT backend instead of depending on Unsloth. Preflight measured BF16 support and 6,225,395,712 free bytes after model loading; the completed run therefore did not use 4-bit quantization. This avoids mixing a quantization change into this measured run.

The locked settings were:

| Parameter | Value | Reason for the starter experiment |
|---|---:|---|
| Seed | 42 | One seeded recorded run; exact determinism across stacks is not claimed. |
| Steps | 100 | Small enough for one local run, long enough to revisit 36 training tasks for 2.78 reported epochs. |
| Generations per group | 4 | Provides within-prompt reward comparison while fitting memory. |
| Batch / accumulation | 1 / 4 | Keeps instantaneous memory low while accumulating four optimizer microsteps. |
| Learning rate | `1e-5` | Matches the value shown in the pinned DebugArena starter recipe. |
| Sampling | temperature 0.9, top-p 0.95 | Allows stochastic variation among group candidates; diversity was not measured. |
| Prompt / completion cap | 512 / 256 | Covers the measured 146–240-token training prompts and bounds rollout cost. |
| LoRA rank / alpha | 16 / 32 | Trains 8,798,208 parameters, 1.7497% of the reported 502,830,976 total. |
| Warmup / max grad norm | 0.05 / 0.1 | Bounds the early schedule and update norm. |
| Checkpoint interval | 25 | Creates four review points without excessive storage. |

These are feasibility settings, not a hyperparameter optimum. No sweep was conducted.

## Training observations

The trainer completed all 100 steps in 1,174.0846 seconds. Peak allocated VRAM was 1,849,278,976 bytes. All 336 tracked trainable tensors changed, with aggregate adapter L2 delta `0.2811997439`. Mean trainer loss was `0.0179620331`; because GRPO loss is a relative policy objective that can be negative per step, its small magnitude is not itself a correctness score. Across the two-step smoke and main training, 404 rollout records comprised 38 host-side structural rejects, 175 deterministic reward-cache hits, and 191 valid cache misses executed in fresh containers; the complete count is recorded in [`run-summary.json`](../evidence/run-summary.json).

The persisted reward history gives a clearer training diagnostic:

| Steps | Mean group reward | Mean logged loss | Zero-reward-std steps | Steps with clipping | Maximum clipped ratio |
|---|---:|---:|---:|---:|---:|
| 1–25 | 1.1625 | 0.05290 | 6 | 3 | 0.25 |
| 26–50 | 1.5740 | 0.01664 | 13 | 1 | 0.25 |
| 51–75 | 1.5230 | 0.00368 | 9 | 0 | 0.00 |
| 76–100 | 1.5355 | -0.00138 | 9 | 0 | 0.00 |

Across all steps, mean reward was `1.44875`, with observed range `-0.1375` to `2.1`. Thirty-seven individual steps had zero reward standard deviation, but the longest consecutive streak was eight, below the stop threshold of twenty. The final streak was zero, `stopped_for_reward_collapse` was false, and no `reward_collapse` event appears in the log. The field name `reward_zero_variance_streak` therefore describes the final diagnostic state; it is not evidence that collapse occurred.

Four early steps had one of four completions reach the token ceiling. No completion clipping occurred in steps 51–100. The maximum clipped ratio of 0.25 is therefore bounded and does not indicate whole-batch truncation in the main run.

## Validation test of the hypothesis

After training, both independent adapters were evaluated on the same 12-task validation set with one greedy and four sampled candidates per task:

| Validation metric | Untouched base | Two-step smoke | 100-step train |
|---|---:|---:|---:|
| Greedy full solves | 7/12 | 7/12 | 8/12 |
| Greedy hidden-test fraction | 0.7708 | 0.7708 | 0.8333 |
| Observed sampled success at 4 | 8/12 | 7/12 | 10/12 |
| Sampled hidden-test fraction | 0.4688 | 0.4844 | 0.7448 |
| Invalid candidates | 19/60 | 18/60 | 2/60 |
| Runtime errors | 2/60 | 1/60 | 3/60 |
| Policy violations | 0/60 | 0/60 | 1/60 |
| Total failures | 21/60 | 19/60 | 6/60 |

The deterministic selection key was `(greedy_hidden_test_fraction, greedy_pass_at_1)`, so the 100-step adapter beat the smoke adapter. Against base, its mean paired greedy hidden-fraction difference was `+0.0625`; a 10,000-sample paired percentile bootstrap with seed 42 gave a 95% interval of `[-0.1458, +0.2917]`. It gained two full solves (`validation_transpose_042`, `validation_longest_word_040`), lost one (`validation_mode_046`), and therefore gained one net solve. Failure rate fell from 0.35 to 0.10.

The predefined gate passed through both improvement clauses—mean difference at least 0.05 and net solved gain at least one—and through the reliability clause. The wide interval still includes zero, so validation supports the operational gate but is not conclusive population-level evidence.

## Budget decision and refined hypothesis

The last two reward windows were close, 1.5230 then 1.5355. Since checkpoint 100 had already passed the validation gate, extending to 200 steps after inspecting these values would have added an unplanned adaptive choice without clear diagnostic justification. At `20260904T025822Z`, the experiment therefore locked `checkpoint-100`. The decision artifact's legacy field `final_benchmarks_opened_before_decision: false` meant that no final model generations, scores, or outcomes were available; internal final data had already been loaded and structurally validated, while HumanEvalPack loading occurred later.

The refined question for Experiment 004 became narrower: does the validation-selected checkpoint retain any improvement on internal tasks held out from optimization and selection and on a frozen public repair benchmark, and which failure modes change? No further weight update or hyperparameter change followed the lock.

## Limitations

- This is one seed and one small synthetic training curriculum, not a learning-curve or scaling study.
- The validation set drove checkpoint selection and the stopping decision; it is not final unbiased evidence.
- Reward is a shaped task outcome, so higher training reward can reflect format compliance as well as semantic repair.
- A four-generation observed success fraction is not an unbiased pass@4 estimator. See the [pass@k estimator discussion](https://arxiv.org/abs/2107.03374).
- The bootstrap interval is wide because validation has only 12 paired tasks.

## Evidence

- [Exact resolved configuration](../evidence/resolved-config.json)
- [Measured run summary](../evidence/run-summary.json)
- [Per-task validation scores](../evidence/task-scores.json)
- Local training-budget decision `logs/20260904T021845Z-minibug-rl-main/training-budget-decision.json`
- Local trainer metrics `logs/20260904T021845Z-minibug-rl-main/metrics.jsonl`
- Local checkpoint state `artifacts/runs/20260904T021845Z-minibug-rl-main/main/checkpoints/checkpoint-100/trainer_state.json`
- Local train-validation records `logs/20260904T021845Z-minibug-rl-main/evaluation-train-validation.json`
