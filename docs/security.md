# Security boundary for generated Python

## Security claim

MiniBug-RL never imports or executes model-generated Python in the trainer/evaluator
host process. Internal cache misses are parsed with the host's `ast` module, then valid
source is sent over standard input to a fresh Docker container; parser/policy rejections
stay on the host, and deterministic cache hits reuse a prior breakdown without execution.
External HumanEvalFix candidates and assertion scripts are compiled and executed only
inside a fresh container.

This is a bounded local experiment, not a hardened multi-tenant code-execution service.
Docker provides the execution boundary, but containers share the host kernel. Docker's
own [security overview](https://docs.docker.com/engine/security/) explains that daemon,
kernel, configuration, and image security all matter. The controls below reduce impact;
they do not prove that arbitrary hostile code cannot escape a vulnerable kernel or
daemon.

## Assets and trust assumptions

The design tries to protect:

- the host filesystem, user account, network, processes, GPU, model cache, and Docker
  control socket;
- `.env` and Hugging Face credentials;
- hidden expected outputs from becoming model-visible reward hints; and
- experimental integrity, so infrastructure failures cannot be counted as model
  failures.

It trusts:

- the host kernel, Docker Engine/Desktop, container runtime, and their security updates;
- the digest-pinned official Python base image and the NumPy package installed while the
  image is built;
- the checked `sandbox/runner.py` and host orchestration source at the recorded Git
  commit;
- the repository-authored task file declared clean-room and the commit-pinned
  HumanEvalPack assertions; and
- the operator not to weaken Docker defaults or mount sensitive host resources.

The Docker daemon is a high-privilege component. A user able to control it may already
have host-equivalent power; this repository is not a defense against a malicious Docker
administrator.

## Boundary and data flows

### Internal MiniBug reward

1. The model emits raw text. The current [logging implementation](../src/minibug_rl/run_logging.py)
   preserves the complete prompt and completion before scoring and later links the
   outcome by generation ID; the producing reference commit instead logged successful
   pairs after scoring.
2. The host parses the completion as Python AST and requires one correctly named
   top-level function. Static policy rules reject imports, selected dynamic builtins,
   dunder attributes, and direct `__builtins__` access.
3. The host constructs a JSON request containing normalized candidate source, function
   name, and hidden call **inputs**. Expected outputs are not in the request.
4. On a cache miss, a fresh container compiles a valid candidate with a reduced builtins
   mapping and invokes it once per input. A cache hit reuses the previously persisted
   deterministic reward breakdown.
5. The runner returns bounded JSON actual values. Only after the container exits does the
   host compare actuals with expected values and compute reward.

Python 3.12 documents that [`exec`](https://docs.python.org/3.12/library/functions.html#exec)
executes arbitrary code. The reduced builtins mapping and AST rules are defense in depth,
not a substitute for process isolation.

### External HumanEvalFix scoring

The benchmark's tests are Python assertion scripts rather than JSON calls. The host
sends candidate source, trusted setup, and trusted assertions in distinct fields of the
versioned `minibug.python-test-script.v1` protocol. The runner compiles setup and tests
first so a malformed pinned suite is an infrastructure error. It then compiles and
executes the candidate and finally executes assertions in the same container namespace.

Full Python semantics and the BigCode import prelude—including NumPy—are available in
this external path for harness compatibility. Consequently, the reduced-builtins claim
applies only to internal MiniBug calls. Docker is the sole security boundary for the
external candidate. The canonical test script never appears in the model prompt; a
test failure returns only a bounded exception class/message.

## Container controls actually used

Every executed request invokes `docker run` through an argument vector, not a shell, and
supplies candidate data through standard input without host bind mounts or volumes. The
checked [`DockerSandbox` command](../src/minibug_rl/sandbox.py) applies:

| Control | Exact setting | Effect and limit |
|---|---|---|
| Disposable instance | `--rm`, unique random name | Removes normal completed containers and lets timeout cleanup target one instance. |
| Image selection | immutable prepared `sha256:…`, `--pull never` | Prevents a mutable tag or implicit pull from changing the runner after `prepare`. |
| Network | `--network none` | Docker's [none driver](https://docs.docker.com/engine/network/drivers/none/) leaves only loopback and no external interface. |
| Filesystem | `--read-only` | Makes the container root filesystem read-only. |
| Scratch | `--tmpfs /tmp:rw,noexec,nosuid,size=16m` | Supplies a small writable temporary area that cannot execute files or honor set-ID bits. |
| Identity | image `USER 65532:65532` and run `--user 65532:65532` | Avoids root inside the container. |
| Linux capabilities | `--cap-drop ALL` | Removes the ordinary capability set. |
| Privilege escalation | `--security-opt no-new-privileges=true` | Prevents acquiring additional privileges through `execve`. |
| Process count | `--pids-limit 32` | Bounds fork/thread amplification. |
| Memory | `--memory 128m --memory-swap 128m` | Caps memory and, because both values are equal, provides no additional swap allowance. |
| CPU | `--cpus 0.5` | Limits CPU scheduling capacity. |
| Wall deadline | 3 seconds on the host | Bounds Docker startup plus candidate execution; timeout handling attempts best-effort kill and forced removal of the named container. |
| Python startup | `python -I` | Uses Python 3.12's [isolated mode](https://docs.python.org/3.12/using/cmdline.html#cmdoption-I) to ignore user site and `PYTHON*` environment customization. |

Docker documents these flags in the
[`docker container run` reference](https://docs.docker.com/reference/cli/docker/container/run/)
and memory/CPU behavior in
[`Resource constraints`](https://docs.docker.com/engine/containers/resource_constraints/).
The command creates no host bind mount, Docker volume, secret mount, Docker-socket mount,
or GPU-device request. It does create the documented 16 MiB `/tmp` tmpfs mount. No
workspace directory, home directory, model cache, `.env`, or token is added to the
request.

The image itself uses the official
[`python:3.12-slim` image at digest `78387bc…`](https://hub.docker.com/layers/library/python/3.12-slim/images/sha256-78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea)
and [`numpy==2.5.2`](https://pypi.org/project/numpy/2.5.2/). Docker recommends digest
pinning because tags are mutable in its
[build best-practices guide](https://docs.docker.com/build/building/best-practices/#pin-base-image-versions).
Preparation records the final platform-specific local image ID; the completed run used
`sha256:a869cd1dffb8c87afad1bb1302106cb9f5cb580641c7391bb73f4ab077f140d9`.

## Protocol and output bounds

The host refuses a serialized request larger than 256 KiB. The runner reads at most that
budget plus one byte, permits at most 64 internal calls, and serializes at most 16 KiB
per actual output. Ordinary Python text writes routed through the runner's redirected
`sys.stdout` and `sys.stderr` retain at most 16,384 characters each. The host rejects a
response larger than 512 KiB after collection and retains at most 16,384 characters of
a decoded Docker diagnostic. The script path retains at most 2,048 characters for an
exception message and never returns a traceback.

Those limits are not a pre-buffer bound on raw container output. In particular, an
external full-Python candidate can use a raw file-descriptor write such as `os.write(1,
...)`, bypass the Python redirect, and corrupt the single JSON protocol response. The
host's Python 3.12 [`Popen.communicate()`](https://docs.python.org/3.12/library/subprocess.html#subprocess.Popen.communicate)
collects raw stdout/stderr before applying the 512 KiB
check. Malformed or oversized protocol output is classified as infrastructure failure
and aborts that scoring path, but raw pipe buffering remains a candidate-triggerable
availability risk. Do not describe the transport as safe against arbitrary output
flooding until the protocol is separated and bounded before buffering.

These limits protect transport availability, not confidentiality from the operator:
complete model prompts and completions are intentionally stored in host logs. The
runner's output capture may truncate candidate prints, while generation logging does
not truncate model text. Those are different data paths.

## Failure classification

Failure meaning is separated so broken infrastructure cannot depress model scores:

| Class | Examples | Experiment behavior |
|---|---|---|
| Candidate structure/policy | syntax error, wrong top-level shape, prohibited internal AST | Model outcome; fixed penalty without execution. |
| Candidate runtime | exception, invalid return, assertion failure | Model outcome. |
| Candidate timeout | infinite loop or workload exceeding 3-second host deadline | Model outcome; best-effort named-container cleanup is attempted. |
| Infrastructure | Docker cannot start, image missing, invalid runner JSON, malformed trusted assertions/setup | Abort the reward/evaluation path; do not turn it into a failed test. |

The checked [verification record](../reports/evidence/verification.json) reports zero
`run_failed` events. Because scoring aborts on surfaced infrastructure failures and all
evaluation summaries completed, the narrower supported inference is that no surfaced
infrastructure error interrupted the completed run. The
[run summary](../reports/evidence/run-summary.json) classifies external timeouts as model
outcomes under the recorded 3-second policy.
Because the pinned BigCode Python evaluator ordinarily uses a 10-second timeout, the
external score is a MiniBug-sandbox measurement and not directly interchangeable with
stock harness results.

## Secret and log handling

`.env`, `.env.*`, `logs/`, `artifacts/`, checkpoints, caches, Trackio data, and coverage
output are ignored by Git. `.env.example` is the only exception and contains a blank
token plus commented account-neutral examples. The CLI refuses an existing `.env` whose
group or other permission bits are set. It logs only the authenticated Hub username
during preflight, never the supplied token. Publication sends the token only to
`huggingface_hub` host APIs; it is not part of the model prompt or Docker request.
Setting `TRACKIO_SPACE_ID` opts training into a separate remote Space, as documented by
[Trackio 0.37](https://huggingface.co/docs/trackio/v0.37.0/track), and should be treated
as an external write, not part of a local-only run.

`generations.jsonl` contains complete prompts and completions by design. Current inputs
are public or synthetic and contain no intentionally private source, but generated text
can still contain unexpected sensitive material and must be treated as potentially
sensitive. Before adapting the pipeline to private source, define retention, access
control, redaction, and upload policy; do not publish raw logs merely because aggregate
results are public. The current Hub package publishes candidate-level result evidence
but not `.env` or the raw run log.

## Residual risks and stronger deployments

The current controls do not address every hostile-code threat:

- containers share the host kernel, so a kernel/runtime escape remains possible;
- rootless Docker, user-namespace remapping, a custom seccomp profile, AppArmor/SELinux,
  and an isolated VM are not enforced by this repository;
- the Docker base is digest-pinned, but the NumPy wheel is version-pinned rather than
  hash-locked in the Dockerfile;
- `--cpus 0.5` is a CPU scheduling quota, while the separate 3-second timeout is a
  wall-clock deadline including container startup; neither is an instruction budget;
- raw file-descriptor output is not pre-capped, and `Popen.communicate()` buffers it
  before the post-hoc protocol-size check;
- timeout cleanup is best-effort and suppresses cleanup-command failures, so operators
  should check for a leftover named container after abnormal exits;
- resource limits can still create brief pressure on the Docker daemon or host;
- a compromised trusted benchmark/setup runs with full Python semantics inside the
  same container; and
- static AST denial is intentionally incomplete and must never be advertised as a
  Python sandbox.

For untrusted internet submissions, run the Docker daemon inside a disposable dedicated
VM or microVM, enable an independently reviewed seccomp/LSM policy, consider Docker's
documented [rootless mode](https://docs.docker.com/engine/security/rootless/), restrict
daemon access, export no host secrets into that VM, and destroy it after the batch. Do
not expose this runner as a network service without authentication, admission control,
rate limiting, per-tenant isolation, monitoring, and an incident response plan.

## Operator checklist

Before a run:

- verify `git status --short` is empty and the intended commit is public;
- verify `.env` is ignored and mode `600` if present;
- patch the host, Docker Engine/Desktop, WSL/kernel, and NVIDIA driver;
- inspect `sandbox/Dockerfile`, `sandbox/runner.py`, and the exact dataset revision;
- run the Docker-marked adversarial test; and
- confirm no unrelated sensitive bind mounts or daemon-wide insecure options are in use.

After `prepare`:

- read `state.json.prepare.sandbox_image_id` and ensure later summaries use the same ID;
- treat any surfaced `infrastructure_error` or malformed protocol response as a stopped
  experiment, and inspect Docker for leftovers after a timeout because cleanup is
  best-effort; and
- preserve logs as potentially sensitive local evidence, not ordinary source files.

After generation, inspect text only. Never copy a model completion into a host `python`,
REPL, notebook, test runner, or shell merely to “quickly check” it; send it through the
same isolated scorer or a stronger boundary.
