# Experiment 004: locked internal and external evaluation

**Classification:** final evaluation after model and budget lock

**Selected model:** 100-step `train` LoRA adapter

**Published result:** [`BurnyCoder/qwen2.5-coder-0.5b-swe-rl@5b6e22a`](https://huggingface.co/BurnyCoder/qwen2.5-coder-0.5b-swe-rl/tree/5b6e22a4c6c01bec95d10e93a0fc78666eb9c543)

**Outcome:** fewer invalidly structured outputs were observed; correctness differences are small and statistically uncertain

## Question and locked protocol

The final question was whether the validation-selected 100-step adapter transferred to tasks whose model generations, scores, and outcomes did not enter training, checkpoint selection, hyperparameter changes, or the stopping decision.

The ordering is part of the evidence:

1. Pre-decision smoke and train validation completed at 02:57:58 UTC.
2. `checkpoint-100` was locked at 02:58:22 UTC. Its legacy `final_benchmarks_opened_before_decision: false` field meant that no final model outcomes were available; internal final task definitions and hidden cases had already been loaded and structurally checked.
3. The first recorded internal-final model generation was at 03:06:05 UTC. The base evaluation completed at 03:08:29 UTC and selected-adapter evaluation at 03:11:27 UTC.
4. HumanEvalPack was loaded in this post-lock phase. Its first recorded generation was at 03:11:35 UTC; base evaluation completed at 03:30:29 UTC and selected-adapter evaluation at 04:05:43 UTC.
5. Each base/selected final benchmark was model-scored once. Only after those results were written did the phase record completion, export the selected adapter and merged model, and publish the result.

The internal protocol used the 12 reserved MiniBug test tasks. Each system produced one deterministic greedy candidate and four temperature-0.9/top-p-0.95 sampled candidates per task, for 60 candidate records. The primary paired score was the greedy fraction of hidden assertions passed.

The external protocol used all 164 Python HumanEvalFix tasks from [`bigcode/humanevalpack`](https://huggingface.co/datasets/bigcode/humanevalpack/tree/9a41762f73a8cb23bb5811b73d5aab164efcf378) at revision `9a41762f73a8cb23bb5811b73d5aab164efcf378`, prompt variant `humanevalfixdocs-python`, one greedy candidate, 256 new tokens, and a three-second Docker timeout. External task scores were binary pass/fail indicators.

All base/selected key sets and record orders matched. Internal artifacts contained exactly one greedy index 0 and sampled indices 0–3 for every task; external artifacts contained exactly `Python/0` through `Python/163`. The paired percentile bootstrap resampled after-minus-before task differences 10,000 times with `random.Random(42)` and the repository's rounded empirical 2.5th/97.5th percentile indices.

## Final internal results

| Metric | Base | Selected adapter | Difference |
|---|---:|---:|---:|
| Greedy full solves | 5/12 (0.4167) | 6/12 (0.5000) | +1 task |
| Greedy hidden-test fraction | 0.6083 | 0.6042 | -0.0042 |
| Observed sampled success at 4 | 10/12 (0.8333) | 10/12 (0.8333) | 0 tasks |
| Sampled hidden-test fraction | 0.4635 | 0.6281 | +0.1646 |
| Invalid structure | 19/60 | 1/60 | -18 candidates |
| Runtime error | 4/60 | 7/60 | +3 candidates |
| Timeout | 0/60 | 1/60 | +1 candidate |
| Policy violation | 0/60 | 1/60 | +1 candidate |
| Total candidate failures | 23/60 | 10/60 | -13 candidates |

The paired mean greedy hidden-fraction difference was `-0.0041667`; its paired percentile-bootstrap 95% interval was `[-0.2916667, +0.2625]`. The point estimate is therefore essentially flat and the interval includes both material harm and material benefit.

Task-level greedy changes explain why solve count and average partial correctness move in different directions:

- Full-solve gains: `test_parse_bool_051` 0.8→1.0, `test_wrap_index_054` 0.25→1.0, and `test_pair_sums_055` 0.25→1.0.
- Partial improvement without a full solve: `test_nearest_multiple_052` 0.5→0.75.
- Full-solve losses: `test_alternating_053` 1.0→0.25 and `test_trace_059` 1.0→0.0.
- Partial regression: `test_hamming_049` 0.25→0.0.
- Five tasks tied exactly.

The improvements sum to `+1.95` task-fractions and regressions to `-2.00`, yielding `-0.05 / 12 = -0.0041667`. Binary full solves have three gains and two losses, hence the simultaneous net gain of one solved task.

The strongest observed internal change is structural reliability: invalid generations fell from 19 to 1, and total failures from 23 to 10. That result needs qualification because runtime errors rose from four to seven and one timeout and one policy violation appeared. In this evaluation the selected adapter emitted parseable single-function answers more consistently; the single run does not establish a general causal effect or show that every newly parseable answer executed correctly.

The sampled hidden fraction rose, but observed at-least-one-of-four full success stayed 10/12. No paired interval was specified or reported for the sampled metric, so it remains secondary descriptive evidence rather than the headline.

## Frozen HumanEvalFix results

| Metric | Base | Selected adapter | Difference |
|---|---:|---:|---:|
| Passed | 37/164 | 38/164 | +1 task |
| Pass@1 | 0.22561 | 0.23171 | +0.00610 |
| Failed, excluding timeout | 124 | 122 | -2 |
| Timeout | 3 | 4 | +1 |

The paired percentile-bootstrap 95% interval for pass@1 difference was `[-0.0182927, +0.0365854]`. It includes zero. Three tasks changed from fail to pass—`Python/69`, `Python/78`, and `Python/98`—while `Python/62` and `Python/86` changed from pass to fail. Of the remaining 159 binary ties, `Python/80` changed from ordinary failure to timeout; 158 tasks retained the same status.

This is a net improvement of one task or 0.61 percentage points, accompanied by one extra timeout. It is compatible with a small positive transfer effect, no effect, or a small negative effect under the recorded interval.

## Conclusion

The precise supported headline is:

> After 100 GRPO steps, the selected 0.5B LoRA solved one more selection-held-out internal repair greedily (5/12→6/12) and one more frozen HumanEvalFix task (37/164→38/164). Internal candidate failures were 23/60 for base and 10/60 for the selected adapter, corresponding primarily to 18 fewer malformed outputs. Average greedy internal hidden-test correctness was essentially unchanged (60.83%→60.42%; paired percentile-bootstrap 95% interval -29.17 to +26.25 percentage points), and HumanEvalFix's +0.61-point difference was also uncertain (paired percentile-bootstrap 95% interval -1.83 to +3.66 points).

This experiment demonstrates that a complete single-function software-repair RL workflow—executable rewards, LoRA updates, locked selection, held-out-from-selection evaluation, export, and publication—ran on the recorded laptop hardware. It does not establish a statistically conclusive general repair-capability gain or repository-scale software-engineering ability.

## Resulting model and artifact verification

The selected adapter was exported both as a 35,237,104-byte PEFT adapter and as a merged Transformers model. The adapter weight SHA-256 is `37f5f71fae87a4fcc5b3df40d325544271d28cf7b55d6a0262c6096d6233b1b2`. The original export reload outputs matched, and a clean redownload of Hub revision `5b6e22a4c6c01bec95d10e93a0fc78666eb9c543` completed generation. The original publication phase did not hash-compare weights. A later independent check found identical local/redownloaded SHA-256 values for merged `model.safetensors` (`e6b2d3884e72fe8322eaa5fe181ce530593f710b8b4efdcf2012b911d2416a70`) and `adapter/adapter_model.safetensors` (`37f5f71fae87a4fcc5b3df40d325544271d28cf7b55d6a0262c6096d6233b1b2`). The Hub revision contains the adapter, merged model, tokenizer, training configuration, model card, and complete comparison result.

## Limitations and next experiment

- Twelve internal tasks give a very wide paired interval and make one-task changes look large in pass@1.
- One seed cannot separate a training effect from optimizer and generation variance.
- Internal tasks are short, synthetic, import-free functions; they do not approximate repository-scale issue resolution.
- This evaluation uses only HumanEvalPack's Python function-repair subset.
- The sampled four-draw success fraction is observational, not the unbiased estimator discussed in the [Codex pass@k paper](https://arxiv.org/abs/2107.03374).
- Timeouts are counted separately from ordinary external failures and increased from three to four.

A clean follow-up should preregister multiple seeds and a larger paired external suite, retain a fixed training budget, and treat formatting, runtime correctness, and semantic test success as separate endpoints. It should not tune against these already model-scored final sets.

## Evidence

- [Machine-readable run summary](../evidence/run-summary.json)
- [Per-task paired scores](../evidence/task-scores.json)
- [Independent verification record](../evidence/verification.json)
- Local authoritative comparison `logs/20260904T021845Z-minibug-rl-main/comparison.json`
- Local base internal records `logs/20260904T021845Z-minibug-rl-main/evaluation-base-test-final.json`
- Local selected internal records `logs/20260904T021845Z-minibug-rl-main/evaluation-selected-test-final.json`
- Local base HumanEvalFix records `logs/20260904T021845Z-minibug-rl-main/external-evaluation-base-humanevalfix.json`
- Local selected HumanEvalFix records `logs/20260904T021845Z-minibug-rl-main/external-evaluation-selected-humanevalfix.json`
