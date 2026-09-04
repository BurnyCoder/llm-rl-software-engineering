# Experiment 002: corrected two-step smoke gate

**Classification:** systems gate pre-specified in producing commit `7027bc5`

**Run:** `20260904T021845Z-minibug-rl-main`

**Source:** [`7027bc55baecc00fad51cbe2b8f030dca2c91c1e`](https://github.com/BurnyCoder/llm-rl-software-engineering/commit/7027bc55baecc00fad51cbe2b8f030dca2c91c1e)

**Outcome:** passed; the 100-step allocation proceeded

## Question and hypothesis

After correcting the curriculum and execution protocol, the question was whether the exact code committed to GitHub could complete the hardware, dataset, sandbox, reward, LoRA update, logging, save, and reload path before committing approximately 20 minutes of GPU time to the main trainer.

The hypothesis was deliberately narrow: two GRPO steps should change trainable tensors, produce finite persisted metrics, save a reloadable adapter, avoid the 20-log reward-collapse stop, and stay inside the available 8 GB GPU. It did not require a quality improvement after two steps.

## Locked inputs

The final run began 33 seconds after its source became GitHub merge commit `7027bc5`. The state recorded an empty tracked diff against that commit for `src`, `sandbox`, `configs`, `pyproject.toml`, and `uv.lock`; it did not establish that other tracked paths or untracked files were clean. It also recorded the following identities:

- canonical configuration: `b6f27e3ab14cec6fa0f8badb2e630427a2abe5f91629ec20c441fa396c4b0e35`;
- corrected 60-task curriculum: `f5049138a16269a064ec3488481223bc0b611fd416ab88ee3bec5f84291e0aeb`;
- schema-v2 split manifest: `9e30a088c50634c78dbb4e547e97dc4bf20b4518df5b88f9a00539e4c8968549`;
- prepared sandbox: `sha256:a869cd1dffb8c87afad1bb1302106cb9f5cb580641c7391bb73f4ab077f140d9`.

Preparation reported 36 training, 12 validation, and 12 final-test tasks. It structurally loaded all 60 tasks and ran each original buggy function against hidden tests, including the internal final split, before training. This preparation did not produce model scores for the final split. It also ran the structural cross-split curriculum audit, passed the internal sandbox canary, and passed a synthetic canary for the external assertion-script protocol; the latter did not load HumanEvalPack. The same sandbox digest is present in smoke training and every subsequent evaluation summary.

The base model was `Qwen/Qwen2.5-Coder-0.5B-Instruct` at immutable revision `ea3f2471cf1b1f0db85067f1ef93848e38e88c25`. The smoke phase derived its bounded settings from the committed main configuration: two steps, two generations, batch size one, gradient accumulation two, maximum completion length 128, seed 42, BF16, learning rate `1e-5`, rank-16 LoRA, and alpha 32. This follows the same core [TRL 1.12.0 `GRPOTrainer`](https://huggingface.co/docs/trl/v1.12.0/en/grpo_trainer) and [PEFT 0.20.0 LoRA](https://huggingface.co/docs/peft/v0.20.0/en/package_reference/lora) APIs as the main run.

## Results

| Measurement | Result |
|---|---:|
| Global steps | 2 |
| Mean training loss | 0.1007812619 |
| Trainable / reported total parameters | 8,798,208 / 502,830,976 |
| Trainable share | 1.7497% |
| Changed trainable tensors | 168 |
| Adapter L2 change | 0.0146747118 |
| Peak training VRAM | 1,610,769,920 bytes |
| Maximum reward standard deviation | 1.6970562935 |
| Maximum clipped-completion ratio | 1.0 |
| Final zero-variance streak | 1 |
| Reward-collapse stop | false |
| Reloaded adapter | verified |

The full step records explain the maximum clipped ratio. Step 1 generated terminated completions of 40 and 83 tokens, reward mean `0.9`, and reward standard deviation `1.6971`. At step 2 both generations reached the 128-token smoke ceiling, so `completions/clipped_ratio` was 1.0; both were invalid, giving reward `-0.3` and zero standard deviation. This isolated bad batch reset neither the adapter proof nor the safety gate: the configured stop requires 20 consecutive logs whose zero-standard-deviation group fraction exceeds 0.8.

The filter passed only defined metric values to Trackio. In the separately persisted `metrics.jsonl`, the undefined step-2 clipping extrema were absent rather than serialized as null, every numeric value was finite, and the training phase completed. The structured log contains no `run_failed`, `Traceback`, or `reward_collapse` event. Because process stderr and an immutable Trackio export are absent from the evidence bundle, this supports the local metric and completion claims, not remote Trackio completeness or archival of every third-party diagnostic stream.

When both adapters were later evaluated on the validation split, the smoke adapter matched the base on deterministic quality:

| Validation metric | Base | Two-step smoke |
|---|---:|---:|
| Greedy full solves | 7/12 | 7/12 |
| Greedy hidden-test fraction | 0.7708 | 0.7708 |
| Observed sampled success at 4 | 8/12 | 7/12 |
| Sampled hidden-test fraction | 0.4688 | 0.4844 |
| Candidate failures | 21/60 | 19/60 |

The sampled quantities are secondary observations over four recorded draws per task. “Observed sampled success at 4” is not the unbiased pass@k estimator described in the [Codex evaluation paper](https://arxiv.org/abs/2107.03374).

## Interpretation and decision

The systems hypothesis passed. The adapter changed, reloaded, stayed well below available VRAM, exercised real hidden-test rewards in the immutable sandbox, and left an auditable metric stream. Matching base greedy validation performance is acceptable because two steps were never a learning threshold.

The main 100-step run therefore proceeded from the base model, not from the smoke adapter. Keeping the smoke and main adapters independent prevents a hidden two-step warm start and makes the configured 100-step budget exact.

## Evidence

- [Resolved configuration](../evidence/resolved-config.json)
- [Run summary and artifact digests](../evidence/run-summary.json)
- Local structured run log `logs/20260904T021845Z-minibug-rl-main/run.log`, smoke events 76–84
- Local persisted metrics `logs/20260904T021845Z-minibug-rl-main/metrics.jsonl`, smoke records 1–3
- Local smoke validation `logs/20260904T021845Z-minibug-rl-main/evaluation-smoke-validation.json`
- Local base validation `logs/20260904T021845Z-minibug-rl-main/evaluation-base-validation.json`
