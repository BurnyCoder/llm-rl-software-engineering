# Methodology: verifiable-reward learning for tiny Python repair

## Research question and scope

The experiment asks a deliberately narrow question: can a 0.49B-parameter code model
receive measurable, non-collapsed policy updates from hidden unit-test rewards on the
recorded 8 GB laptop GPU, and does the selected adapter retain or improve correctness
on tasks that did not supply its training rewards?

“Software-engineering RL” here means replacing one buggy, import-free Python function.
It does not mean editing a repository, running a build system, resolving dependencies,
or acting autonomously on SWE-bench. This restriction makes functional correctness
executable and verifiable, keeps the experiment affordable, and makes failures
attributable to parsing, runtime, or functional behavior.

The design was inspired by the short pinned
[`debugarena` GRPO example](https://huggingface.co/BharathVikas/debugarena/blob/e4be134207727d684b8a7edfbc7934159893617b/README.md), but it is not an exact reproduction
of DebugArena or its linked public [Kaggle notebook](https://www.kaggle.com/code/bharathvikas/debugarena).
That snippet establishes the useful
`GRPOTrainer(model, reward_funcs, args, train_dataset)` shape; MiniBug-RL supplies the
omitted frozen data, LoRA configuration, reward semantics, resource-limited isolated
Docker executor, selection rule, held-out comparisons, artifact verification, and exact
revisions. The actual
implementation follows the public
[`GRPOTrainer` documentation for TRL 1.12.0](https://huggingface.co/docs/trl/v1.12.0/en/grpo_trainer),
the corresponding
[`GRPOTrainer` source](https://github.com/huggingface/trl/blob/v1.12.0/trl/trainer/grpo_trainer.py),
and PEFT 0.20's [LoRA API](https://huggingface.co/docs/peft/v0.20.0/package_reference/lora).
The algorithmic references are the primary [GRPO](https://arxiv.org/abs/2402.03300) and
[LoRA](https://arxiv.org/abs/2106.09685) papers.

## Fixed inputs

### Base policy

The policy starts from
[`Qwen/Qwen2.5-Coder-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct/blob/ea3f2471cf1b1f0db85067f1ef93848e38e88c25/README.md)
at commit `ea3f2471cf1b1f0db85067f1ef93848e38e88c25`. The pinned model card describes an
instruction-tuned causal language model with 0.49B total parameters, 24 layers, grouped
query attention, and a 32,768-token context window. MiniBug-RL loads both tokenizer and
model at that commit with `trust_remote_code=False`, BF16, PyTorch SDPA, and no
quantization. The training prompt ceiling is 512 tokens and the completion ceiling is
256 tokens; the checked [run summary](../reports/evidence/run-summary.json) records
training prompts of 146–240 tokens. The corresponding model-family reference is the
[Qwen2.5-Coder report](https://arxiv.org/abs/2409.12186).

### Project-authored curriculum

`data/minibug_tasks.json` contains 60 repository-authored records declared clean-room
and no canonical repaired programs. The repository does not independently prove
originality:

| Split | Tasks | Role |
|---|---:|---|
| Train | 36 | Supplies model prompts and hidden execution rewards. |
| Validation | 12 | Measures the base, compares smoke/main candidates, and selects one. |
| Final test | 12 | Structurally validated before training; model-scored only after the checkpoint/budget lock and never used for selection. |

Each task has a natural-language specification, one buggy function, exactly two public
examples, and four or five hidden calls in the frozen file (the validator schema permits
four through eight). Public examples are model-visible behavior, not gold code. Hidden
calls and expected values are separate reward columns and never enter the prompt.
`data/split_manifest.json` binds every record to a split with a
canonical SHA-256 and a literal/name-normalized AST fingerprint. Preparation rejects
changed hashes, missing IDs, split drift, and structurally duplicated templates that
cross splits. [The data contract](../data/README.md) owns the field-level schema.

This controls leakage introduced by this repository; it cannot prove that the
pretrained base never encountered similar tasks or public benchmarks.

### Prompt contract

The internal prompt uses Qwen's native chat template, following the installed
[Transformers 5.16.1 tokenizer implementation](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/tokenization_utils_base.py).
It contains:

1. a system instruction to return exactly one complete, pure replacement function;
2. the behavioral specification;
3. the buggy function;
4. two public examples serialized as JSON calls; and
5. the requested function name.

Current code logs each complete raw prompt/completion pair before cache lookup, parsing,
or execution, then appends a linked outcome record; an exception outcome is written
before an infrastructure exception is re-raised. By contrast, producing commit
`7027bc55…` logged successful reference-run pairs in full only after scoring, so an
exception before that call could omit the current pair. The parser optionally removes
one surrounding Python Markdown fence, requires exactly one top-level function with the
requested name, and rejects imports, selected dynamic builtins, dunder attributes, and
direct `__builtins__` access. These AST rules stabilize the task format and reduce attack
surface; they are not the execution security boundary.

## Reward definition

For a valid candidate with `m` hidden cases and `c` correct outputs, the normal
functional reward is `R = c / m + 1[c = m] + 0.1`, where `1[c = m]` is one only when
every hidden case passes and is zero otherwise.

This yields `0.1` for a structurally valid function that passes no hidden case and `2.1`
for a complete repair. Exceptional paths are explicit and mutually attributable:

| Outcome | Reward | Reason |
|---|---:|---|
| Valid and executable | pass fraction + full-solve bonus + `0.1` | Dense functional credit plus a sparse completion target. |
| Invalid syntax/shape/name | `-0.3` | Formatting failure, with no container execution. |
| Runtime error or timeout | `-0.15` | Valid structure `+0.1`, execution penalty `-0.25`. |
| Prohibited AST construct | `-1.0` | Strong policy penalty, with no container execution. |
| Sandbox infrastructure fault | no model score; abort | A broken daemon/image/protocol is not evidence of model quality. |

Expected hidden outputs stay in the host process. Only candidate source, function name,
and call inputs cross the internal Docker boundary; actual outputs return for strict
host comparison. Boolean equality is type-sensitive so `True` cannot silently pass for
`1`. The deterministic reward cache hashes task ID, exact raw completion, function name,
and the entire hidden suite. A cache hit reuses the full component breakdown but never
adds expected values to generation metadata.

TRL documents that a custom GRPO reward callable receives prompts, completions, and
extra dataset columns and returns one scalar per completion in
[`Using a custom reward function`](https://huggingface.co/docs/trl/v1.12.0/en/grpo_trainer#using-a-custom-reward-function).
MiniBug-RL uses one callable named `hidden_unit_test_reward`; no lint score or learned
reward model is mixed into the objective.

## Policy update

The selected profile is a standard Transformers + PEFT BF16 LoRA run, not Unsloth and
not 4-bit QLoRA. PEFT attaches LoRA to `all-linear` modules with rank 16, alpha 32,
dropout 0, and no trained bias; PEFT's `all-linear` selector excludes the output layer
for a `PreTrainedModel`. PEFT defines ordinary LoRA scaling as alpha/rank, which
is `2` here, and documents that its default initialization starts as a no-op adapter in
the [LoRA 0.20 configuration reference](https://huggingface.co/docs/peft/v0.20.0/package_reference/lora);
the executed release is anchored by the tagged
[`LoraConfig` 0.20.0 source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/lora/config.py).
The attached policy had 8,798,208 trainable parameters out of 502,830,976
(`1.749734686%`).

The main GRPO configuration is:

| Field | Value | Experimental role |
|---|---:|---|
| Training tasks | 36 | Only the train split supplies policy rewards. |
| Optimizer steps | 100 | Fixed initial budget used for the completed result. |
| Generations per prompt | 4 | One relative-reward group. |
| Per-device batch | 1 | Fits the single GPU. |
| Gradient accumulation | 4 | Effective batch 4, divisible by the generation group. |
| Learning rate | `1e-5` | Small adapter update. |
| Warmup ratio | `0.05` | Passed as ratio-valued `warmup_steps` under the installed Transformers behavior. |
| Maximum gradient norm | `0.1` | Gradient clipping. |
| Sampling | temperature `0.9`, top-p `0.95`, top-k `0` | Training rollouts only. |
| Prompt/completion limits | 512 / 256 | Prompt lengths are validated; completions are bounded. |
| GRPO KL coefficient | `beta=0` | No reference-policy KL term or reference model. |
| Loss | `dr_grpo` | Uses the fixed maximum-completion-length normalizer. |
| Reward scaling | `none` | No group-standard-deviation scaling. |
| Truncation | mask truncated completions | Ceiling-hit rollouts are excluded from the loss. |
| Memory controls | BF16, gradient checkpointing, no vLLM, no KV cache | Fits the measured laptop GPU path. |
| Seed | 42 | Model, data, rollout, and evaluation seed root. |
| Checkpoints | every 25 steps; retain two | Supports bounded restart storage. |

These fields correspond to the installed
[`GRPOConfig` 1.12.0 source](https://github.com/huggingface/trl/blob/v1.12.0/trl/trainer/grpo_config.py).
Transformers 5.16.1 documents ratio-valued step fields in its tagged
[`TrainingArguments` source](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/training_args.py#L2115-L2122).
`dr_grpo`, no reward scaling, and `beta=0` describe the executed objective; they should
not be read as a general claim that those choices dominate alternatives.

Before the main allocation, a derived smoke configuration runs two steps with two
generations, accumulation 2, a 128-token completion ceiling, and checkpointing every
step. It must finish, change LoRA tensors, save, and reload. The main run must reach
exactly step 100, have finite reported loss, and change at least one trainable tensor.
An audit callback requests early stop when more than 80% of reward groups have zero
standard deviation for 20 consecutive logs; any early stop then fails the exact-step
postcondition rather than masquerading as completion.

The checked [run evidence](../reports/evidence/run-summary.json) records that the main
run reached step 100, reported training loss
`0.017962033050134777`, changed all 336 observed trainable tensors, had aggregate LoRA
delta L2 `0.28119974388470714`, and did not trigger reward-collapse stopping. These are
optimization diagnostics on seen tasks, not generalization metrics. GRPO's centered
surrogate loss may be positive, zero, or negative and must not be interpreted like a
supervised cross-entropy score.

## Validation selection and learning gate

Base validation is measured before training. After smoke and main training, candidates
are ranked lexicographically by:

1. greedy mean hidden-test fraction;
2. greedy full-solve rate as the tie-breaker.

The main adapter won: its selection pair was `(0.833333, 0.666667)`, versus smoke
`(0.770833, 0.583333)`. It was then compared with the untouched base, which had greedy
hidden-test fraction `0.770833` and solved 7/12 tasks.

Learning succeeds only when both conditions hold:

- paired mean validation improvement is at least `0.05` **or** at least one additional
  validation task is fully solved; and
- the combined invalid-structure, timeout, runtime-error, and policy-violation rate over
  all 60 greedy-plus-sampled candidates does not exceed the base rate.

The gate was pre-specified in producing source commit
[`7027bc55…`](https://github.com/BurnyCoder/llm-rl-software-engineering/blob/7027bc55baecc00fad51cbe2b8f030dca2c91c1e/src/minibug_rl/phases/evaluate.py)
before the authoritative execution; this is a source-committed operational rule, not a
formal preregistration. For the selected main adapter, paired mean hidden-fraction
change was `+0.0625`, solved
tasks changed 7/12 to 8/12, and candidate failure rate changed 21/60 (`0.35`) to 6/60
(`0.10`). Therefore the recorded gate passed. The gate is an operational decision rule,
not a significance test.

## Internal evaluation and uncertainty

Each internal policy receives identical task ordering and a per-task seed of
`42 + task_index`. Each task produces one greedy candidate plus four candidates sampled
at temperature `0.9`, top-p `0.95`, and top-k `0`. Transformers defines
`do_sample=False` as greedy decoding and exposes `max_new_tokens` for completion limits
in its tagged generation implementation;
the exact installed behavior is anchored by the tagged
[`GenerationConfig` 5.16.1 source](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/generation/configuration_utils.py).

Metrics have deliberately different denominators:

- `greedy_pass_at_1`: fraction of tasks whose single greedy repair passes every hidden
  case;
- `observed_sampled_success_at_4`: observed fraction of tasks with at least one success
  among exactly four samples; historical reference artifacts use the legacy key
  `sampled_pass_at_4`, and neither name denotes the unbiased pass@k estimator for a
  larger sample pool;
- hidden-test fraction: macro mean of each candidate's within-task pass fraction;
- failure rates: candidate counts divided by all greedy and sampled candidates.

The frozen final internal comparison was 5/12 to 6/12 greedy solves. The macro greedy
hidden fraction was `0.608333` for base and `0.604167` for the adapter, even though both
passed 30/49 raw hidden cases; macro averaging weights tasks equally, while the raw count
weights the one five-case task more. Observed sampled success@4 remained 10/12, while the
sampled hidden fraction changed `0.463542` to `0.628125`.

Paired uncertainty resamples per-task **after-minus-before** scores 10,000 times with
`random.Random(42)`. It reports the arithmetic mean and deterministic empirical 2.5th
and 97.5th percentiles: a paired percentile-bootstrap 95% interval. The final internal
mean difference was `-0.004167`, interval
`[-0.291667, 0.262500]`. Pairing is essential because each model sees the same tasks;
independent resampling would discard that structure.

## Frozen external evaluation

Only after validation has fixed the selected adapter does evaluation load the Python
configuration of
[`bigcode/humanevalpack`](https://huggingface.co/datasets/bigcode/humanevalpack/blob/9a41762f73a8cb23bb5811b73d5aab164efcf378/README.md)
at commit `9a41762f73a8cb23bb5811b73d5aab164efcf378`. All 164 ordered IDs
`Python/0` through `Python/163` are required. No HumanEvalPack examples, model
candidates, observed outcomes, or scores enter training, reward computation, checkpoint
or budget selection, restarts, or hyperparameter tuning. Its pinned card and harness
source define the frozen external protocol; this narrower statement does not claim that
prior knowledge of the public benchmark could not influence protocol design. The
dataset's primary reference is the [OctoPack paper](https://arxiv.org/abs/2308.07124).

The protocol follows the pinned BigCode
[`HumanEvalFixDocs` implementation](https://github.com/bigcode-project/bigcode-evaluation-harness/blob/8fc5bae6479c4fbbb28c3f8b644f6a15b3f3b5bd/bigcode_eval/tasks/humanevalpack.py):

- prompt variant `humanevalfixdocs-python`, `instruct` mode;
- buggy program, the instruction `Fix bugs in <entry_point>.`, then the original
  completion prefix;
- no Qwen chat template for this benchmark-specific raw prompt;
- one greedy completion per task, sample index zero, maximum 256 new tokens;
- the harness's Python stop strings and first-new-top-level-block truncation;
- the exact common import prelude, including NumPy, before candidate execution; and
- canonical assertions visible only to the isolated scorer, never the model prompt.

The project uses a resource-limited, host-enforced 3-second wall-clock deadline around
container execution, whereas the pinned BigCode Python harness uses 10 seconds. That
deliberate resource-policy difference is recorded
in every result. The score must therefore be labeled “MiniBug sandbox,
HumanEvalFixDocs-Python instruct, greedy n=1,” not presented as a directly comparable
stock-harness leaderboard number.

Base passed 37/164 (`0.225610`) and the selected adapter 38/164 (`0.231707`). The paired
binary mean difference was `+0.006098`, with paired percentile-bootstrap 95% interval
`[-0.018293, 0.036585]`. Three tasks improved and two regressed. The point estimate is
positive, but the interval includes zero and is not evidence of a reliable population
gain. The standard pass@k estimator and its execution warning are documented by the
official Hugging Face
[`code_eval` implementation at `a7dd338…`](https://github.com/huggingface/evaluate/blob/a7dd338386a4fae9a1767e05eb9ef9479513d9e8/metrics/code_eval/code_eval.py).

## Interpretation and validity limits

The checked [run summary](../reports/evidence/run-summary.json) and
[task scores](../reports/evidence/task-scores.json) establish an end-to-end mechanism:
real LoRA tensors changed under
TRL GRPO; a validation-only rule selected the main adapter; invalid output formatting
decreased on the small internal sets; adapter and merged exports reloaded; and a public
immutable Hub snapshot generated successfully.

It does **not** establish a statistically reliable correctness improvement. Every
reported paired percentile-bootstrap 95% interval—validation, internal final test, and external
HumanEvalFix—includes zero. Other limitations are one training seed, 36 reward-bearing
tasks, a synthetic single-function domain, a selected internal validation set of only
12 tasks, observed sampled success@4 rather than a large-sample estimator, a custom
external timeout, possible public-benchmark exposure during base pretraining, and no
repository-level evaluation. The honest conclusion is a functioning small-hardware RL
pipeline with fewer malformed internal candidates and inconclusive generalization gains.

Machine-readable candidate records and exact aggregates are published in
[`results.json`](https://huggingface.co/BurnyCoder/qwen2.5-coder-0.5b-swe-rl/blob/5b6e22a4c6c01bec95d10e93a0fc78666eb9c543/results.json).
