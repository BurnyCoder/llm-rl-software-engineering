# Reproducibility contract and evidence map

## What “reproducible” means here

MiniBug-RL separates three claims that are often conflated:

1. **Artifact identity:** the exact published bytes can be retrieved by immutable Hub
   commit and checked against recorded SHA-256 values.
2. **Procedural reproducibility:** the recorded source, locked dependencies, config,
   data, model, benchmark, sandbox, seeds, and commands are sufficient to repeat the
   same experiment design.
3. **Numerical reproducibility:** fixed seeds control the recorded RNG streams and improve
   repeatability, but sampled decoding and GPU kernels are not guaranteed bit-identical,
   even on the same stack.

The first two claims are satisfied for the completed run. The third is deliberately
bounded rather than overstated. Transformers 5.16.1 implements revision selection in
tagged [`from_pretrained` source](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/modeling_utils.py),
Hugging Face Hub 1.30 documents immutable revision downloads through
[`snapshot_download`](https://huggingface.co/docs/huggingface_hub/v1.30.0/package_reference/file_download#huggingface_hub.snapshot_download),
and PyTorch documents the broader nondeterminism boundary in its official
[reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html).

## Canonical completed run

| Identity | Exact value |
|---|---|
| Run ID | `20260904T021845Z-minibug-rl-main` |
| Git source | `7027bc55baecc00fad51cbe2b8f030dca2c91c1e` |
| Recorded runtime-scope tracked diff SHA-256 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` (empty bytes) |
| Resolved public config SHA-256 | `b6f27e3ab14cec6fa0f8badb2e630427a2abe5f91629ec20c441fa396c4b0e35` |
| `uv.lock` SHA-256 | `38568853e845d1b5ff58bd13ac43ee26b1853bd9f5b903fe73ee3d12a73201c8` |
| Curriculum file SHA-256 | `f5049138a16269a064ec3488481223bc0b611fd416ab88ee3bec5f84291e0aeb` |
| Split manifest file SHA-256 | `9e30a088c50634c78dbb4e547e97dc4bf20b4518df5b88f9a00539e4c8968549` |
| Base model | `Qwen/Qwen2.5-Coder-0.5B-Instruct` |
| Base revision | `ea3f2471cf1b1f0db85067f1ef93848e38e88c25` |
| HumanEvalPack revision | `9a41762f73a8cb23bb5811b73d5aab164efcf378` |
| BigCode protocol reference | `bigcode-evaluation-harness@8fc5bae6479c4fbbb28c3f8b644f6a15b3f3b5bd` |
| Prepared Docker image ID | `sha256:a869cd1dffb8c87afad1bb1302106cb9f5cb580641c7391bb73f4ab077f140d9` |
| Result model repository | `BurnyCoder/qwen2.5-coder-0.5b-swe-rl` |
| Original result Hub commit | `5b6e22a4c6c01bec95d10e93a0fc78666eb9c543` |
| Corrected-card Hub commit | `8dc6fa7df8a09e157e3a5be1ec17b5d8b8aa4f43` |

The source commit is public at
[`7027bc55…`](https://github.com/BurnyCoder/llm-rl-software-engineering/commit/7027bc55baecc00fad51cbe2b8f030dca2c91c1e).
The result is browsable at the immutable
[`5b6e22a…` model tree](https://huggingface.co/BurnyCoder/qwen2.5-coder-0.5b-swe-rl/tree/5b6e22a4c6c01bec95d10e93a0fc78666eb9c543).
That commit remains the original experiment artifact identity. A later README-only
correction was merged at [`8dc6fa7d…`](https://huggingface.co/BurnyCoder/qwen2.5-coder-0.5b-swe-rl/tree/8dc6fa7df8a09e157e3a5be1ec17b5d8b8aa4f43); the checked
[documentation audit](../reports/evidence/documentation-audit.json) records the exact
file-set comparison and successful inference without rewriting the original run record.

The state identity hashes `RunConfig.public_dict()` as compact, key-sorted UTF-8 JSON;
hashes the curriculum and manifest as raw file bytes; and records `git rev-parse HEAD`
plus a binary diff over `src`, `sandbox`, `configs`, `pyproject.toml`, and `uv.lock`.
Because `public_dict()` contains resolved absolute paths, its config digest is sensitive
to checkout location; the recorded value identifies this run rather than serving as a
location-independent TOML digest. The empty diff digest proves no tracked difference in
the listed runtime-source paths. It does not cover other tracked paths or detect an
untracked file, which is why the operator guide also requires a clean Git status and a
pushed commit.

## Locked software and measured environment

`uv.lock` is the complete resolver artifact. The principal direct packages used by the
run are:

| Component | Version |
|---|---:|
| Python project requirement | `>=3.12` |
| PyTorch | `2.14.0+cu130` |
| Transformers | `5.16.1` |
| TRL | `1.12.0` |
| PEFT | `0.20.0` |
| Datasets | `5.0.1` |
| Accelerate | `1.14.0` |
| huggingface-hub | `1.30.0` |
| Trackio | `0.37.0` |

The run's persisted [environment record](../reports/evidence/run-summary.json) reports
PyTorch `2.14.0+cu130`, CUDA runtime 13.0, NVIDIA GeForce RTX 5070 Laptop GPU, driver
591.74, compute capability 12.0, 8,151 MiB via `nvidia-smi`, BF16 support, and Docker
server 29.4.1. Host OS build, CPU, firmware, Python patch version, and uv executable
version were not captured; no exact historical value is inferred for those fields.

The Docker build separately pins:

- `python:3.12-slim` by digest
  `sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea`;
- `numpy==2.5.2`; and
- the checked `sandbox/runner.py` from the recorded source commit.

The final image ID is platform-specific and was resolved after the build. All reward and
evaluation summaries record that immutable ID rather than the mutable local tag. The
NumPy version is fixed but its wheel hash is not locked in the Dockerfile, so an exact
future image rebuild is less strongly pinned than reuse of the already prepared image.

## Published byte checks

The local Hub staging tree and the cleanly downloaded Hub snapshot had matching SHA-256
values for these central files:

| File at Hub commit `5b6e22a…` | SHA-256 |
|---|---|
| `model.safetensors` | `e6b2d3884e72fe8322eaa5fe181ce530593f710b8b4efdcf2012b911d2416a70` |
| `adapter/adapter_model.safetensors` | `37f5f71fae87a4fcc5b3df40d325544271d28cf7b55d6a0262c6096d6233b1b2` |
| `results.json` | `158046a436c2cdff5c036ef4a606ac450f98c67a78f5916019cb4d22ae44fbfa` |
| `training_config.json` | `c08ee82cb72069a0490909250cde227cb4aa49723932813c6bd91c18fb9739f0` |
| `tokenizer.json` | `3fd169731d2cbde95e10bf356d66d5997fd885dd8dbb6fb4684da3f23b2585d8` |
| `config.json` | `16b94db246860669503a772467b8f895f585ac2cdb056c3496ff0de264c2911a` |

The Hub commit is the primary immutable identity. File digests provide an independent
local check and catch a mistaken directory or incomplete copy.

```bash
# Download the immutable public snapshot with the Hub CLI supplied by huggingface-hub.
uv run hf download BurnyCoder/qwen2.5-coder-0.5b-swe-rl \
  --revision 5b6e22a4c6c01bec95d10e93a0fc78666eb9c543 \
  --local-dir artifacts/reference-model
# Compare the merged model with the recorded digest above.
sha256sum artifacts/reference-model/model.safetensors
# Compare the separately reloadable LoRA delta with its independent digest.
sha256sum artifacts/reference-model/adapter/adapter_model.safetensors
# Confirm that measured records, rather than a hand-entered table, have the expected bytes.
sha256sum artifacts/reference-model/results.json
# Confirm the resolved public training configuration copied into the publication.
sha256sum artifacts/reference-model/training_config.json
```

The CLI behavior is documented in Hugging Face's
[`hf download` guide](https://huggingface.co/docs/huggingface_hub/v1.30.0/guides/cli#hf-download).
Downloading into `artifacts/` keeps the large result ignored by Git.

## Run-directory evidence schema

Every run invocation appends to the same timestamped directory. Secret token values are
never serialized. Public operator settings can be: an `HF_MODEL_REPO` override becomes
the resolved `hub_model_id`, while
[`TRACKIO_SPACE_ID`](https://huggingface.co/docs/trackio/v0.37.0/track) controls the
separate Trackio integration and is not part of `RunConfig`.

| Path | Format and authority |
|---|---|
| `resolved-config.json` | Full non-secret configuration after TOML path resolution and the optional `HF_MODEL_REPO` override. |
| `state.json` | Atomic resumable phase state. `_identity` binds config, source, data, and manifest before prior work can be reused. |
| `environment.json` | Preflight hardware, Docker, base-revision resolution, prompt range, and real generation evidence. |
| `run.log` | Append-only JSON Lines lifecycle stream plus generation and linked outcome records. It may contain repeated phase attempts after a resume. |
| `generations.jsonl` | Append-only generation projection. Current records write `generation_id`, UTC timestamp, task ID, split, complete prompt, and complete raw completion before scoring. Scored paths then append a `generation_outcome` carrying the same ID and outcome metadata; unscored smoke generations stand alone. |
| `metrics.jsonl` | Append-only numeric trainer snapshots keyed by UTC timestamp and global step. |
| `metrics.csv` | Long-form `timestamp,step,metric,value` projection for ordinary plotting tools. |
| `reward-cache.sqlite3` | Deterministic reward breakdown cache keyed by exact completion, task/callable, and hidden suite. This is local working state, not the reporting source. |
| `evaluation-*.json` | Internal summary, greedy task-score mapping, and all 60 candidate records for that policy/split. |
| `external-evaluation-*.json` | External summary, 164 binary task scores, and ordered candidate records including reconstructed source. |
| `comparison.json` | Selected candidate, learning gate, base/selected final results, external results, and paired intervals in one object. |
| `training-budget-decision.json` | Run-specific operator decision that locked checkpoint 100 before final benchmarks. It is evidence from this run, not a generic CLI output promised for every reproduction. |

Current `RunLogger.generation` intentionally does not truncate prompts or completions.
The same pre-score generation record is written to `run.log` and `generations.jsonl`,
while terminal output prints the raw multiline pair immediately; later outcome records
correlate through `generation_id`. Producing commit `7027bc55…` instead wrote one combined
record after successful scoring. Its historical records remain readable, but an
exception before logging could omit that current pair. Runner stdout, stderr, and
diagnostics have separate security limits and should not be confused with model-text
logging. The
[`JSON Lines specification`](https://jsonlines.org/) explains why each append-only record
remains independently parseable.

Raw run logs are ignored by Git and were not uploaded with the model because they are
large and may be sensitive in adaptations of this project. The public Hub
[`results.json`](https://huggingface.co/BurnyCoder/qwen2.5-coder-0.5b-swe-rl/blob/5b6e22a4c6c01bec95d10e93a0fc78666eb9c543/results.json)
contains the selected comparison and candidate-level scored evidence; it is not a
substitute for the private append-only training trace.

## Training and artifact directories

For run ID `<run-id>`, the training phase writes:

| Path | Contents |
|---|---|
| `artifacts/runs/<run-id>/smoke/checkpoints/` | Retained smoke checkpoints with optimizer, scheduler, RNG, trainer state, tokenizer, and adapter. |
| `artifacts/runs/<run-id>/smoke/adapter/` | Final two-step smoke adapter and tokenizer. |
| `artifacts/runs/<run-id>/main/checkpoints/` | Last two retained main checkpoints; the completed run retained 75 and 100. |
| `artifacts/runs/<run-id>/main/adapter/` | Step-100 main adapter, later selected by validation. |
| `trainer_log_history.json` | Complete Transformers/TRL log-history list for that training allocation. |
| `training_result.json` | Terminal step, loss, image ID, prompt range, parameter counts, peak VRAM, tensor delta, and collapse/clipping diagnostics. |

Export moves any older `artifacts/final` tree to a timestamped `.previous-*` sibling,
then writes:

- `adapter/`: copied selected LoRA adapter and tokenizer;
- `merged/`: standard merged Transformers weights, configuration, and tokenizer;
- `hub/`: the exact regular-file allowlist declared in `artifacts.py`: loadable merged
  root files, the loadable adapter subtree, generated model card, `results.json`, and
  `training_config.json` (trainer state such as `training_args.bin` is excluded); and
- `export_result.json`: source commit, paths, and one-prompt reload comparison result.

The export check loads the adapter form and merged form independently and requires both
to generate; it records whether one greedy verification completion is identical. For the
completed run, `reload_outputs_match` is true. This is a useful integration check, not a
formal proof that every possible output or floating-point operation is identical.

Current publication uses `HfApi.create_repo(..., exist_ok=True)` and `upload_folder`,
retains the returned commit OID, and follows the Hub 1.30
[`upload guide`](https://huggingface.co/docs/huggingface_hub/v1.30.0/guides/upload).
It locks the write to the inspected parent commit, replaces older remote files, and
requires the returned remote tree to contain the reviewed allowlist with no extra file
except Hub-managed `.gitattributes`. It then downloads that exact OID into
`artifacts/hub-verification/<oid>/`, requires the same relative paths and SHA-256 values
for every local and downloaded Safetensors file, loads only local files, and completes
one greedy generation. Producing commit `7027bc55…`
gated the reference publication on exact-revision redownload, load, and non-empty
generation; the byte equality in this document was established by the subsequent
independent verification rather than that historical phase. The recorded reference
outcome is `redownload_generation_completed: true`.

## Resume semantics

The supported recovery unit is a phase:

- a recorded phase is skipped unless `--force` is explicit;
- an individual phase fails before work if its prerequisite state is absent;
- reopening a run ID recomputes identity and rejects config/data/source mismatch;
- main and smoke training use the last checkpoint found in their checkpoint directory;
- an existing completed training result causes a forced rerun to use a timestamped
  sibling rather than overwrite evidence;
- final internal test sides can be reused only when their summary selectors, configured
  task count, and exact configured test-task ID set match; and
- external evidence is reused only after strict validation of all 164 ordered records,
  sample indices, statuses, aggregates, exact score keys, model, base/data revision,
  prompt variant, token limit, image ID, and timeout.

Append-only logs may therefore contain duplicate records from interrupted or deliberate
reruns. The atomic standalone result JSON and `state.json` value are the canonical
completed phase view; preserve the log timestamps when reconstructing attempt history.

## Determinism and comparison rules

The seed root is 42. Training sets both trainer and data seeds. Evaluation resets to
`42 + task_index` for each task so base and adapter see the same task-local random stream.
Greedy decoding uses `do_sample=False`; internal sampled evaluation uses four returns at
temperature 0.9, top-p 0.95, top-k 0. External evaluation is greedy with one return.

All paired intervals sort identical task IDs, compute after-minus-before per task, then
draw 10,000 bootstrap resamples with `random.Random(42)`. Internal intervals operate on
greedy hidden-test fractions; the external interval operates on binary pass values.
This algorithm is reproducible from `results.json` without rerunning a model or executing
a candidate.

Seeds do not eliminate every nondeterministic GPU kernel, numeric change between CUDA
stacks, or timing-dependent timeout. A repeat should first verify identities and record
its own artifacts, then report any numerical difference rather than replacing the
reference result. Bit-identical model bytes are expected only when the full training
execution is itself bit-identical; downloading the pinned published commit is the way to
obtain the exact reference bytes.

## Minimum audit before making a claim

1. Verify Git status, commit, config/data/manifest hashes, base revision, and Docker image
   ID.
2. Confirm `global_step == 100`, finite training metrics, changed LoRA tensors, and saved
   adapter files.
3. Confirm validation chose `train` and record the gate operands, not only the boolean.
4. Confirm internal and external base/selected task IDs pair exactly and all aggregate
   counts recompute from records.
5. Treat timeout and candidate runtime failures separately from infrastructure errors;
   any infrastructure error invalidates that measurement.
6. Quote paired intervals alongside point estimates. All reference 95% intervals include
   zero.
7. State that the external deadline is 3 seconds rather than the BigCode harness's 10
   seconds and that upstream pretraining contamination cannot be ruled out.
8. Verify both local artifact forms, retain the Hub commit from upload, download that
   commit, require identical relative paths and SHA-256 values for every Safetensors file,
   and generate a non-empty completion from the downloaded snapshot.

Following this checklist supports an auditable repeat. It does not turn a one-seed,
small-task experiment into a general claim about software-engineering performance.
