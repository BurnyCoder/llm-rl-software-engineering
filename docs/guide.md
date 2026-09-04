# Exact guide: train, evaluate, export, and load MiniBug-RL

This is the command-level path from a fresh checkout to a real 0.49B-parameter
Python-repair model. The already completed result is
[`BurnyCoder/qwen2.5-coder-0.5b-swe-rl`](https://huggingface.co/BurnyCoder/qwen2.5-coder-0.5b-swe-rl/tree/5b6e22a4c6c01bec95d10e93a0fc78666eb9c543),
pinned to Hub commit `5b6e22a4c6c01bec95d10e93a0fc78666eb9c543`. It contains a directly
loadable merged model at the repository root and the selected LoRA adapter under
`adapter/`. The publication phase downloaded that exact Hub commit into a fresh local
directory and completed a greedy generation from it.

This guide owns commands and operator checkpoints. The experimental choices and reward
are explained in [methodology.md](methodology.md), the isolation boundary in
[security.md](security.md), and the hashes and evidence files in
[reproducibility.md](reproducibility.md).

## 1. Use the tested machine shape

The persisted evidence for the completed run records an NVIDIA GeForce RTX 5070 Laptop
GPU with 8,151 MiB reported VRAM, CUDA 13.0, driver 591.74, BF16, and Docker Engine
29.4.1. Main training peaked at 1,849,278,976 allocated GPU bytes. The host OS build was
not persisted, so it is not part of the measured environment claim. These are
observations, not a claim that every other GPU or driver is equivalent. The `preflight`
phase deliberately fails unless CUDA and BF16 work, Docker is reachable, the pinned
model can generate, all prompts fit, and at least 768 MiB remains free after model
loading.

Install these host prerequisites before cloning:

- Git.
- Python 3.12 and [uv](https://docs.astral.sh/uv/getting-started/installation/). The
  project declares Python `>=3.12`; the completed workspace used Python 3.12.3 and uv
  0.11.27.
- An NVIDIA driver visible to PyTorch through `nvidia-smi`.
- Docker Engine or Docker Desktop configured for Linux containers. Generated programs
  run in CPU-only containers; Docker GPU access is neither requested nor needed.
- Enough disk for the base cache, checkpoints, and roughly 1 GB merged model.

## 2. Check out the exact public source

The measured run used source commit
[`7027bc55baecc00fad51cbe2b8f030dca2c91c1e`](https://github.com/BurnyCoder/llm-rl-software-engineering/commit/7027bc55baecc00fad51cbe2b8f030dca2c91c1e).
Use the detached commit for a strict reproduction instead of a moving branch.

```bash
# Git's clone command obtains the public source and its commit graph.
git clone https://github.com/BurnyCoder/llm-rl-software-engineering.git
# Every later relative path assumes the repository is the working directory.
cd llm-rl-software-engineering
# A detached checkout prevents a future branch update from changing the experiment.
git checkout --detach 7027bc55baecc00fad51cbe2b8f030dca2c91c1e
# Empty output confirms that no local source or data edit is entering the run.
git status --short
# Exit status zero confirms that the measured commit is present in public origin history.
git merge-base --is-ancestor HEAD origin/main
```

Git documents detached commits and object identity in
[`git-checkout`](https://git-scm.com/docs/git-checkout) and ancestry testing in
[`git-merge-base`](https://git-scm.com/docs/git-merge-base).

If you intentionally change the experiment, create a branch in a fork, commit, and push
it before training. Do not start the costly phase from an unpushed working tree. The
pipeline records the commit and a tracked source-diff hash, but a public commit is the
recoverable source of truth. The original measured commit above was already pushed and
merged before its 100-step run began.

## 3. Recreate the locked environment

[`uv sync --locked`](https://docs.astral.sh/uv/concepts/projects/sync/) creates the local
`.venv` from `uv.lock` and errors if the lock would need to change. PyTorch comes from the explicitly
configured CUDA 13.0 wheel index.

```bash
# `--locked` turns lock drift into an error instead of silently resolving new versions.
uv sync --locked
# The first line should report Python 3.12 from the project-local environment.
uv run python --version
# This confirms that the CUDA-index PyTorch wheel imports before a long run.
uv run python -c "import torch; print(torch.__version__, torch.version.cuda)"
# Both commands must reach the same host services that `preflight` will inspect.
nvidia-smi
docker version
```

Local training and export need no secret. Publication needs a Hugging Face write token.
If publication to your own namespace is planned, configure it **before the first phase**
because the resolved Hub repository is part of the run identity. The CLI loads `.env`
without overriding existing process variables and rejects group/world-readable `.env`
files. Hugging Face documents the token variable in its
[environment-variable reference](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables).

```bash
# Start from the checked template; it contains names only, never a credential.
cp .env.example .env
# The CLI enforces this owner-only permission before reading the file.
chmod 600 .env
# Edit HF_TOKEN and change HF_MODEL_REPO to a repository in your own Hub namespace.
${EDITOR:-vi} .env
# Git must continue to ignore the secret-bearing file.
git check-ignore --verbose .env
```

Leave `.env` absent for the closest local reproduction of the published configuration,
or set only optional `TRACKIO_SPACE_ID`. Do not put tokens in TOML, shell history,
prompts, logs, or Docker build arguments. A different `HF_MODEL_REPO` changes only the
publication destination, but correctly produces a different configuration hash.

## 4. Run the fast checks

These checks do not train a model. The opt-in real-container tests expect the local
sandbox image; `prepare` builds it later, so either run the ordinary suite now and the
container tests after `prepare`, or build the image explicitly.

```bash
# Ruff checks the repository's selected correctness and style rules.
uv run ruff check .
# Mypy checks the strict package type contract.
uv run mypy src
# Pytest exercises data, prompts, rewards, phases, artifacts, and mocked isolation paths.
uv run pytest -m "not docker"
# This build uses the digest-pinned Python base and the checked runner as its only context.
docker build --file sandbox/Dockerfile --tag minibug-rl-sandbox:local sandbox
# The opt-in variable prevents ordinary test runs from starting real containers accidentally.
MINIBUG_RUN_DOCKER_TESTS=1 uv run pytest \
  tests/test_sandbox.py tests/test_external_sandbox.py
```

The official [Docker build reference](https://docs.docker.com/reference/cli/docker/buildx/build/)
defines the build invocation. Treat a failed security test as a stop condition, not as
a reason to bypass Docker.

## 5. Choose one run ID and preserve it

Every invocation opens `logs/<run-id>/state.json`. Reusing the same ID is how interrupted
work resumes. A changed config, curriculum, split manifest, source commit, or tracked
source diff is rejected rather than mixed into the old state.

```bash
# UTC plus the profile name follows the same stable naming convention as the measured run.
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-minibug-rl-main"
# Print and save this value; a new terminal must reuse the literal value, not recompute it.
printf '%s\n' "$RUN_ID"
```

The completed reference run ID was `20260904T021845Z-minibug-rl-main`. Do not reuse that
literal in a fresh clone: run logs and checkpoints are intentionally not distributed in
Git, and a distinct run ID prevents accidental evidence mixing.

## 6. Execute the phases in order

Run each line only after the previous command exits successfully. This makes the small
smoke gate and the expensive 100-step allocation visibly separate.

```bash
# Verify CUDA/BF16, Docker, model revision, prompt lengths, and one real generation.
uv run swe-rl preflight --config configs/local-8gb.toml --run-id "$RUN_ID"
# Validate all 60 records and their split manifest, build Docker, and run both canaries.
uv run swe-rl prepare --config configs/local-8gb.toml --run-id "$RUN_ID"
# Measure the untouched base on validation before any adapter update is selected.
uv run swe-rl baseline --config configs/local-8gb.toml --run-id "$RUN_ID"
# Train two steps with two generations and prove the smoke adapter reloads.
uv run swe-rl smoke --config configs/local-8gb.toml --run-id "$RUN_ID"
# Train the declared 100-step, four-generation BF16 LoRA experiment.
uv run swe-rl train --config configs/local-8gb.toml --run-id "$RUN_ID"
# Select on validation, then evaluate internal final test and frozen HumanEvalFixDocs.
uv run swe-rl evaluate --config configs/local-8gb.toml --run-id "$RUN_ID"
# Copy the adapter, merge it into the base, and verify both local forms by generation.
uv run swe-rl export --config configs/local-8gb.toml --run-id "$RUN_ID"
```

For an unattended local run, the supported equivalent is:

```bash
# `--skip-publish` completes local verification without making a remote change.
uv run swe-rl all --config configs/local-8gb.toml --run-id "$RUN_ID" --skip-publish
```

The phase runner skips state-recorded phases by default. After a crash or restart, issue
the same phase command with the same config and run ID; training also discovers the last
checkpoint under its run directory. Reserve `--force` for an intentional new
measurement. A forced completed training phase writes a timestamped rerun directory
rather than overwriting its prior result.

`evaluate` is deliberately expensive: it produces one greedy and four sampled internal
candidates per task, then one greedy external candidate for each of 164 HumanEvalFix
tasks for both policies. Do not infer a hang from sparse terminal progress; every full
prompt and completion is flushed as it is produced.

### Lower-memory diagnostic and adaptation

The checked 8 GB profile is the only profile behind the published measurements. Before
allocating 100 steps on different hardware, the existing `configs/smoke.toml` can run the
same plumbing with a two-step budget. Use a new run ID because its configuration identity
is intentionally different.

```bash
# The smoke profile has its own project name, two-step main budget, and 128-token ceiling.
SMOKE_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-minibug-rl-smoke"
# `all` still evaluates/exports, so stop after the smoke phase for a cheap allocation check.
uv run swe-rl preflight --config configs/smoke.toml --run-id "$SMOKE_RUN_ID"
# Preparation validates the same data and real Docker runner before any reward call.
uv run swe-rl prepare --config configs/smoke.toml --run-id "$SMOKE_RUN_ID"
# A base validation is the required predecessor of the training smoke gate.
uv run swe-rl baseline --config configs/smoke.toml --run-id "$SMOKE_RUN_ID"
# This must complete, update adapter tensors, save, and reload on the target GPU.
uv run swe-rl smoke --config configs/smoke.toml --run-id "$SMOKE_RUN_ID"
```

If `local-8gb.toml` still runs out of memory, copy it to a new checked profile, give it a
new `project.name`, reduce `num_generations` from 4 to 2, set
`gradient_accumulation_steps` to 2, and reduce `max_completion_length` from 256 to 192.
The effective batch must remain divisible by `num_generations`; the config loader enforces
that invariant. Commit and push the new profile before running it, and use a new run ID.
That is a new experiment and must not reuse or be reported as the published 100-step
result. The repository does not implement 4-bit loading, so do not claim a QLoRA or
Unsloth reproduction without adding, testing, and recording that backend explicitly.

### Focused troubleshooting

| Symptom | Check and resolution |
|---|---|
| `.env must not be accessible` | Run `chmod 600 .env`; never weaken the CLI check. |
| `Existing run state does not match` | Restore the recorded commit/config/data or choose a new run ID. Do not edit `state.json` to bypass identity. |
| Docker command or canary fails | Run `docker version`, rebuild the checked image, then run the opt-in Docker tests. Do not execute candidates on the host. |
| CUDA/BF16 preflight fails | Verify the NVIDIA driver and CUDA PyTorch wheel; use compatible hardware rather than disabling the BF16 check mid-run. |
| Training process was interrupted | Reissue the same phase, config, and run ID; the last compatible checkpoint is discovered automatically. |
| Evaluation appears slow | Follow `run.log` or `generations.jsonl`; external evaluation runs 328 fresh containers across both policies. |
| Hub namespace mismatch | Set `HF_MODEL_REPO` to the authenticated user's namespace before creating the run, then use a new run ID. |
| Hub download/load differs | Pin the exact returned Hub commit and verify file hashes from `reproducibility.md`; never compare against moving `main`. |

## 7. Inspect the gate before publication

The following read-only check uses only Python's standard JSON parser. It verifies the
conditions that were true in the completed run without executing any generated source.

```bash
# Pass the run directory as data rather than embedding it in the Python program.
uv run python - "logs/$RUN_ID" <<'PY'
# `json` decodes the atomic public phase state; it never imports generated code.
import json
# `sys.argv` receives the run directory supplied by the shell command above.
import sys
# `Path` keeps path construction explicit and platform-independent.
from pathlib import Path

# The sole command-line argument is the timestamped evidence directory.
run_directory = Path(sys.argv[1])
# UTF-8 is the encoding used by the run logger for every JSON artifact.
state = json.loads((run_directory / "state.json").read_text(encoding="utf-8"))
# Main training must reach the configured terminal step before evaluation is credible.
assert state["train"]["global_step"] == 100
# Validation must choose the main candidate, not the two-step smoke adapter.
assert state["evaluate"]["selected_candidate"] == "train"
# The declared learning/reliability gate must be recorded as passed.
assert state["evaluate"]["learning_success"] is True
# Export must prove that the adapter and merged form produced equal greedy smoke text.
assert state["export"]["reload_outputs_match"] is True
# Print only public outcomes; no prompt, hidden expected value, or token is read here.
print({key: state["evaluate"][key] for key in ("selected_candidate", "learning_success")})
PY
```

For the reference run, the training adapter changed 336 trainable tensors with aggregate
L2 delta `0.28119974388470714`; export reported `reload_outputs_match: true`. Numeric
performance and its uncertainty are interpreted in [methodology.md](methodology.md),
not used as a reason to tune on final test.

## 8. Publish only after local verification

Publication is an explicit remote write. `HF_TOKEN` must belong to the namespace in the
resolved `HF_MODEL_REPO`; the phase refuses a mismatch. It uploads the prepared Hub
directory, retains the returned immutable commit, downloads that commit with
[`snapshot_download`](https://huggingface.co/docs/huggingface_hub/guides/download), and
generates from the downloaded merged model.

```bash
# This is the only phase that requires HF_TOKEN and changes the Hugging Face Hub.
uv run swe-rl publish --config configs/local-8gb.toml --run-id "$RUN_ID"
```

The reference publication completed with:

- repository: `BurnyCoder/qwen2.5-coder-0.5b-swe-rl`;
- immutable Hub commit: `5b6e22a4c6c01bec95d10e93a0fc78666eb9c543`;
- `redownload_generation_completed: true`.

Hugging Face documents the exact
[`create_repo` and `upload_folder` APIs](https://huggingface.co/docs/huggingface_hub/guides/upload).
A reproducer publishing to another namespace will receive a different Hub commit even
when its model bytes match.

## 9. Load and use the actual published result

This example pins the immutable Hub commit, applies Qwen's chat template, and prints the
generated text. It does **not** execute the generated function. Transformers documents
revision-pinned loading in
[`from_pretrained`](https://huggingface.co/docs/transformers/main_classes/model) and
greedy decoding in its
[`generate` reference](https://huggingface.co/docs/transformers/main_classes/text_generation).

```python
# PyTorch supplies the measured BF16 CUDA dtype and inference-only context.
import torch

# Transformers loads the tokenizer and the merged causal language model from one Hub commit.
from transformers import AutoModelForCausalLM, AutoTokenizer

# This immutable repository revision names the concrete resulting model, not a moving tag.
MODEL_ID = "BurnyCoder/qwen2.5-coder-0.5b-swe-rl"
# Pinning the upload commit makes every Hub file selection immutable.
MODEL_REVISION = "5b6e22a4c6c01bec95d10e93a0fc78666eb9c543"
# Remote custom code is disabled because this is a standard Transformers Qwen model.
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    revision=MODEL_REVISION,
    trust_remote_code=False,
)
# The merged root loads directly; no base-model or adapter assembly is required here.
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    revision=MODEL_REVISION,
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
    trust_remote_code=False,
)
# The documented run targets a CUDA GPU and keeps model execution out of the Docker scorer.
model.to("cuda")
# Evaluation mode disables dropout for inference.
model.eval()

# The system contract matches the training prompt's single-function output requirement.
messages = [
    {
        "role": "system",
        "content": (
            "You repair one pure Python function. Return exactly one complete replacement "
            "function and nothing else. Do not import modules or use unsafe builtins."
        ),
    },
    {
        "role": "user",
        "content": (
            "Specification:\nReturn the absolute value of an integer.\n\n"
            "Buggy function:\n```python\ndef absolute_value(value):\n    return -value\n```\n\n"
            "Public examples as JSON calls:\n"
            '[\n  {"args": [-3], "kwargs": {}, "expected": 3},\n'
            '  {"args": [4], "kwargs": {}, "expected": 4}\n]\n\n'
            "Write the corrected `absolute_value` function now."
        ),
    },
]
# Qwen's native template inserts its required control tokens and assistant prefix.
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
# The prompt already contains template tokens, so no second special-token layer is added.
inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
# Inference mode avoids storing gradients while deterministic greedy decoding runs.
with torch.inference_mode():
    # One beam with sampling disabled is the same greedy mode used for pass@1 evaluation.
    generated = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
        num_return_sequences=1,
        repetition_penalty=1.0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
# Slice away prompt tokens so only the model's new repair text is decoded.
completion_tokens = generated[0, inputs["input_ids"].shape[1] :]
# Skipping control tokens yields the exact human-readable candidate text.
completion = tokenizer.decode(completion_tokens, skip_special_tokens=True)
# Printing is safe; use the repository's Docker scorer before executing any generated code.
print(completion)
```

The base Qwen card reports a 0.49B-parameter instruction-tuned causal model and documents
the same chat-template workflow:
[`Qwen/Qwen2.5-Coder-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct/blob/ea3f2471cf1b1f0db85067f1ef93848e38e88c25/README.md).

## 10. Know what this proves

Completing the guide proves that a small model can receive real GRPO updates from
container-scored unit-test rewards, be selected without final-test tuning, be exported
in adapter and merged forms, and be reloaded from an immutable public artifact. It does
not prove repository-scale software engineering ability, a statistically significant
generalization gain, absence of upstream benchmark contamination, or perfect isolation
against a container escape. Those boundaries are made explicit in the other documents.
