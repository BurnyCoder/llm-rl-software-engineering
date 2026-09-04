# MiniBug-RL contributor instructions

## Purpose and scope

Keep the smallest end-to-end software-repair RL path practically runnable before adding optional scale. MiniBug-RL targets one pure Python function at a time; do not describe it as repository-scale software engineering or general agent training.

Use `uv`, the checked local `.venv`, test-driven changes, modular phase functions, timestamped untruncated prompt/completion logs, and primary-source links. Never run model-generated or benchmark code in the host Python process. Never commit `.env`, tokens, raw run logs, caches, checkpoints, downloaded datasets, or model weights.

## Architecture ownership

The public entry point in `src/minibug_rl/cli.py` must remain a thin coordinator. `src/minibug_rl/pipeline.py` owns the ordered phase map and delegates to independently testable modules under `src/minibug_rl/phases/`:

1. `preflight` verifies CUDA BF16, the pinned base revision, Docker, prompt lengths, and Hub identity.
2. `prepare` validates the fixed curriculum, builds and resolves the Docker image ID, and runs curriculum/sandbox canaries.
3. `baseline` measures the untouched base on validation only.
4. `smoke` runs the short gradient/update/reload proof.
5. `train` performs the checked GRPO run and saves resumable checkpoints plus a LoRA adapter.
6. `evaluate` selects only on validation, then evaluates the selected immutable checkpoint on internal final and external frozen tasks.
7. `export` saves and reload-checks both adapter and merged model forms.
8. `publish` uploads the allowlisted Hub tree, re-downloads the returned immutable revision, and performs inference.

Keep implementation details in their existing modules:

- `config.py` parses public TOML only; secrets must never enter its dataclasses or serialized form.
- `task_data.py` validates curriculum and structural split manifests.
- `prompts.py` is the sole internal prompt builder.
- `parser.py` provides pre-container syntax and policy checks.
- `sandbox.py` owns the shared resource-limited Docker transport and host-only answer comparison.
- `external_eval.py` pins/converts HumanEvalFix without executing it.
- `external_sandbox.py` maps assertion-script container replies to evaluation outcomes.
- `reward.py` owns the single reward definition used by training and internal scoring.
- `training.py` owns PEFT/GRPO configuration, health checks, Trackio filtering, and adapter evidence.
- `evaluation.py`, `external_evaluation.py`, and `metrics.py` own frozen measurement and paired uncertainty.
- `artifacts.py` owns model cards, merging, local reload verification, and the publication allowlist.
- `run_logging.py` is the only prompt/output/metric logging abstraction.
- `context.py` owns immutable run identity and atomic resumable phase state.

Do not duplicate parsing, scoring, sandbox command construction, model loading, prompt rendering, or aggregation logic in a phase wrapper.

## Non-negotiable experiment invariants

- Pin the base model and external dataset with full immutable Hub commit SHAs.
- Record the resolved Docker image ID and pass that ID—not a mutable tag—to every reward/evaluation call.
- Keep all expected MiniBug outputs in the host process. Only candidate source, the public function name, and JSON call inputs may enter the internal sandbox request.
- Keep external benchmark assertions out of model prompts. They may enter only the versioned Docker assertion-script request.
- Training may use train tasks only. Validation may choose the checkpoint and budget only. Internal-final and HumanEvalFix model outcomes may not influence model choice, hyperparameters, reward design, or restarts; structural and sandbox-canary validation may cover all internal splits before training.
- Set seeds before model/adapter construction and reset evaluation seeds per task for paired comparisons.
- Preserve one greedy candidate and exactly the configured sampled indices per internal task. Label the four-sample metric “observed sampled success@4,” not an unbiased pass@k estimator.
- Treat Docker/protocol failures as invalid experiment infrastructure. Candidate syntax, assertion, runtime, policy, and timeout outcomes remain model outcomes; cleanup is best-effort and must be checked separately after abnormal exits.
- Stop rather than publish if training does not reach the expected step, produces non-finite values, changes no LoRA tensor, loses reward variance for the configured collapse streak, or fails adapter reload.
- Report all point estimates, failure categories, paired intervals, regressions, and protocol deviations. A passed point-estimate gate is not statistical significance.
- Describe Docker as defense in depth, not a perfect hostile-code security boundary.

## Data changes

`data/minibug_tasks.json` contains repository-authored tasks with no gold repairs. Any curriculum edit requires all of the following before training:

1. Add or update tests first.
2. Regenerate the structural manifest only through the reviewed curriculum tooling.
3. Verify exact IDs/counts and the 36/12/12 split.
4. Verify no normalized structural fingerprint crosses a split.
5. Verify no public case exactly duplicates a hidden case.
6. Execute every repository-authored buggy program in the real immutable container and prove at least one hidden failure per task.
7. Review prompts and logs to prove hidden expected values are absent.

The schema, fingerprints, and leakage rationale belong in `data/README.md`; do not repeat them elsewhere.

## Development and verification

Use the repository root and locked environment:

```bash
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest --cov=minibug_rl --cov-report=term-missing
```

Real-container tests are opt-in and require the locally built image:

```bash
docker build --tag minibug-rl-sandbox:local sandbox
MINIBUG_RUN_DOCKER_TESTS=1 uv run pytest -m docker
```

The pinned network test is intentionally separate because it contacts Hugging Face:

```bash
MINIBUG_RUN_NETWORK_TESTS=1 uv run pytest tests/test_external_eval.py
```

Before a real training run, push all intended source/config/data/lock changes, verify `git status --short` is empty, and verify `HEAD == origin/<branch>`. Keep the run tied to that source commit. After each material fix, push before repeating expensive execution. Do not rewrite or delete earlier experiment evidence; preserve it as diagnostic or superseded.

For code review, do one focused pass over correctness, security, leakage, reproducibility, maintainability, tests, and claims. Inspect the complete user path and logs, but do not loop indefinitely on low-value polish.

## Documentation ownership

Keep information in one canonical place and link to it:

- `README.md`: public method summary, architecture graph, headline measured result, shortest verified commands, resulting model, and navigation.
- `docs/guide.md`: exact operator runbook, phase/resume commands, expected artifacts, inference, testing, and troubleshooting.
- `docs/methodology.md`: stable model/data/reward/GRPO/selection/evaluation rationale.
- `docs/security.md`: threat model, controls, adversarial tests, residual risks, and safe adaptation.
- `docs/reproducibility.md`: versions, identities, hashes, seeds, evidence files, and determinism limits.
- `data/README.md`: task schema, structural manifest, and leakage controls.
- `reports/experiments/`: one lab-notebook record per actual attempt.
- `reports/README.md`: chronological ledger and cross-experiment conclusions.
- `reports/evidence/`: canonical checked machine-readable public evidence used by prose claims.
- `paper/`: archival scientific narrative and bibliography; generated numerical macros should come from canonical evidence.
- `output/pdf/`: compiled paper only; auxiliary LaTeX files stay ignored.
- The Hugging Face model card: artifact identity, loading, measured result, intended use, and limitations—not the repository installation guide.

Do not leave TODOs, placeholders, stale “in progress” statements, or claims unsupported by checked evidence. A documentation-only commit must not be presented as the source commit that produced an earlier model.

## Git and release workflow

Work on a named branch, split commits by coherent function, open a pull request, wait for CI, review the diff, merge it, and verify local `main` equals `origin/main`. Do not commit generated local artifacts except the intentionally tracked evidence JSON and final PDF. Scan tracked files for credential-like strings before every push.

Publication requires the user-authorized `HF_TOKEN` from a mode-600 ignored `.env`. Upload only `artifacts/final/hub`, retain the returned Hub commit SHA, re-download that exact revision to a fresh directory, require identical paths and SHA-256 values for every Safetensors file, load the root model, and log a non-empty generation before recording publication complete.
