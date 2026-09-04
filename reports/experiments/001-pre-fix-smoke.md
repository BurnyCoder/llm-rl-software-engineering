# Experiment 001: pre-fix diagnostic smoke

**Classification:** diagnostic only; excluded from the final experiment and every model-quality comparison

**Run:** `20260904T011842Z-minibug-rl-main`

**Source:** [`d3990b7ad419aa211f6fcf201c53303062111088`](https://github.com/BurnyCoder/llm-rl-software-engineering/commit/d3990b7ad419aa211f6fcf201c53303062111088)

**Outcome:** preflight, preparation, baseline, and two GRPO steps completed; the 100-step phase was deliberately not started

## Question and hypothesis

The initial question was operational: can the pinned 0.5B code model load in BF16, generate repairs, receive Docker-isolated hidden-test rewards, update LoRA tensors, save an adapter, and reload it on this laptop GPU?

The hypothesis was that a two-step smoke would exercise that complete training path within roughly 2 GB of VRAM. It was not designed to measure learning.

## Method

The run used `Qwen/Qwen2.5-Coder-0.5B-Instruct` at revision `ea3f2471cf1b1f0db85067f1ef93848e38e88c25`, seed 42, BF16, rank-16 LoRA with alpha 32, and the local Transformers/PEFT backend of TRL's [`GRPOTrainer`](https://huggingface.co/docs/trl/grpo_trainer). The smoke derivation reduced the main profile to two steps, two generations, batch size one, gradient accumulation two, and 128 completion tokens. Generated code executed only in Docker.

The run identity recorded:

- config SHA-256 `b6f27e3ab14cec6fa0f8badb2e630427a2abe5f91629ec20c441fa396c4b0e35`;
- then-current curriculum SHA-256 `d1819fd3e89c22b850cbc8282f029db641cdb3d3ac954c9bae5a60d83dce5a68`;
- then-current manifest SHA-256 `6fb6d457a3322211a74c06610ae2a39ba883031049afa280511370c512a521c2`;
- a clean source-diff hash, `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

## Observations

The hardware path worked. Preflight found an NVIDIA GeForce RTX 5070 Laptop GPU with 8,546,484,224 bytes of VRAM, CUDA 13.0, BF16 support, and Docker 29.4.1. Preparation built sandbox image `sha256:b360c7eff7e2a4e9d6395aa3b8c6377cf2758694bc5a64947faa2778482f4cf9` and passed its canary.

The two-step trainer completed without a structured `run_failed` event:

| Measurement | Value |
|---|---:|
| Global steps | 2 |
| Trainable / reported total parameters | 8,798,208 / 502,830,976 |
| Trainable share | 1.7497% |
| Changed trainable tensors | 168 |
| Adapter L2 change | 0.0146748621 |
| Mean training loss | 0.1007812619 |
| Peak training VRAM | 1,610,769,920 bytes |
| Maximum clipped-completion ratio | 1.0 |
| Final zero-variance streak | 1 |
| Reward-collapse stop | false |
| Adapter reload | verified |

Step 1 had mean reward `0.9` with reward standard deviation `1.6971`. Both step-2 candidates reached the 128-token ceiling, were structurally invalid, and received reward `-0.3`; that made the step-2 clipped ratio and zero-standard-deviation fraction both 1.0. A single zero-variance step is not a collapse under the 20-log stopping rule.

The run also measured a validation baseline of 9/12 greedy solves and mean greedy hidden fraction 0.8333. Those numbers are retained as diagnostic history only and must not be compared with the final run because the curriculum changed after this audit.

## Post-smoke audit and intervention

The smoke succeeded as a systems test, but review found that the evidence protocol was not yet strong enough for the allocated run:

1. The old manifest checked literal-preserving AST uniqueness rather than alpha- and literal-normalized structural overlap across splits.
2. `validation_leap_037` exposed the decisive century case in its public examples while the buggy function did not pass those examples; the corrected curriculum keeps public behavior compatible with the bug and reserves the divisible-by-400 distinction for hidden tests.
3. `validation_shift_char_038` exposed the wraparound input publicly; the corrected version moves that discriminating input to hidden tests.
4. A `train_occurrences_016` hidden case used booleans, where Python's boolean/integer equality and object identity can accidentally reward the buggy `is` implementation; the replacement nested-list case tests equality versus identity without that alias.
5. The two-step terminal output exposed a Trackio conversion warning when TRL supplied undefined extrema. That stderr warning was not persisted in the structured run log, so this report does not present it as log-derived evidence. The subsequent implementation retained Transformers' Trackio lifecycle while dropping only `None` metrics before `trackio.log`, consistent with the [`trackio.log(metrics, step)` API](https://huggingface.co/docs/trackio/api#trackio.log).
6. The sandbox tag was not yet propagated as an immutable image digest through every reward and evaluation artifact, and there was no frozen external HumanEvalFix evaluation.

The fixes were implemented in [`332b7f6`](https://github.com/BurnyCoder/llm-rl-software-engineering/commit/332b7f6d55db266b2174f1c460fa048a688e38e2) and [`b173192`](https://github.com/BurnyCoder/llm-rl-software-engineering/commit/b17319222a1a906bc11c95f77aae06e65ae33448). They were reviewed and merged to GitHub as [`7027bc5`](https://github.com/BurnyCoder/llm-rl-software-engineering/commit/7027bc55baecc00fad51cbe2b8f030dca2c91c1e) before the final run began.

## Interpretation and decision

The operational hypothesis was supported: the complete reward/update/save/reload loop fit the hardware and changed the adapter. The experiment-quality hypothesis was not tested. Because the data and evaluation protocol were revised after seeing this run, continuing its checkpoint would have mixed incompatible evidence and risked leakage. The correct decision was to stop, classify the run as diagnostic, merge the protocol fixes, and start a new clean run from step zero.

## Evidence

- Local structured diagnostic log `logs/20260904T011842Z-minibug-rl-main/run.log` — SHA-256 `da4c488b4db1fdd473cbe8456a66479acdb10904fc9518a4cd8fd44e0cb9ef36`
- Local diagnostic state `logs/20260904T011842Z-minibug-rl-main/state.json` — SHA-256 `eab9127b049a2e14b3a29c8210b638c4a069f272804ffff376cc0b6e6fa201b7`
- Local diagnostic baseline `logs/20260904T011842Z-minibug-rl-main/evaluation-base-validation.json` — SHA-256 `1689182f82fd16a1231d16ea9ee94ecaf967dd724ed2052ee7ce2dfb628afdde`
- Local diagnostic metrics `logs/20260904T011842Z-minibug-rl-main/metrics.jsonl` — SHA-256 `e55471d5d45254c1443dc0e65ff9d6312fbc197b334b53f470edf9a5d3d0fa92`
- [Final fix diff](https://github.com/BurnyCoder/llm-rl-software-engineering/compare/d3990b7ad419aa211f6fcf201c53303062111088...7027bc55baecc00fad51cbe2b8f030dca2c91c1e)

Raw logs and transient model artifacts are local and intentionally untracked. Their hashes make this diagnostic record tamper-evident without committing those large run products to Git.
