"""Global context: validate frozen data and build/test the isolated execution image.

Sources:
- https://docs.docker.com/reference/cli/docker/buildx/build/
- https://docs.python.org/3/library/hashlib.html
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from minibug_rl.context import PipelineContext
from minibug_rl.curriculum_audit import verify_buggy_programs_fail_hidden
from minibug_rl.external_eval import PythonTestScriptPayload
from minibug_rl.external_sandbox import DockerPythonTestSandbox
from minibug_rl.sandbox import run_candidate
from minibug_rl.task_data import load_tasks, validate_split_manifest


def run_prepare(context: PipelineContext) -> dict[str, Any]:
    """Verify split integrity, build the pinned image, and run a real canary."""
    logger = context.logger
    config = context.config
    logger.message("phase_start", "Preparing curriculum and Docker sandbox.", phase="prepare")
    tasks = load_tasks(config.project.data_file)
    repository_root = config.project.data_file.parent.parent
    manifest_path = config.project.data_file.with_name("split_manifest.json")
    counts = validate_split_manifest(config.project.data_file, manifest_path)
    dockerfile = repository_root / "sandbox" / "Dockerfile"
    sandbox_context = dockerfile.parent
    subprocess.run(
        [
            "docker",
            "build",
            "--file",
            str(dockerfile),
            "--tag",
            config.project.sandbox_image,
            str(sandbox_context),
        ],
        check=True,
        timeout=300,
    )
    # The canary validates JSON transport and host-only correctness comparison end to end.
    canary = run_candidate(
        "def identity(value):\n    return value",
        "identity",
        [{"args": ["sandbox-ok"], "kwargs": {}, "expected": "sandbox-ok"}],
        image=config.project.sandbox_image,
    )
    if canary.status != "success" or not canary.all_passed:
        raise RuntimeError(f"Sandbox canary failed: {canary.status}: {canary.error}")
    # Every original bug must produce normal outputs yet fail at least one hidden case.
    buggy_hidden_coverage_checked = verify_buggy_programs_fail_hidden(
        tasks,
        image=config.project.sandbox_image,
    )
    # HumanEvalFix uses a distinct assertion-script protocol and an official NumPy prelude.
    external_canary = DockerPythonTestSandbox(image=config.project.sandbox_image).execute(
        PythonTestScriptPayload(
            entry_point="sum_array",
            test_setup_source="",
            test_source="assert sum_array([1, 2, 3]) == 6",
        ),
        "import numpy as np\n\ndef sum_array(values):\n    return int(np.sum(values))",
    )
    if not external_canary.passed:
        raise RuntimeError(
            f"External sandbox canary failed: {external_canary.status}: {external_canary.error}"
        )
    # The image ID uniquely identifies the local runner used by every reward call.
    image_id = subprocess.run(
        ["docker", "image", "inspect", config.project.sandbox_image, "--format", "{{.Id}}"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    result = {
        "task_count": len(tasks),
        "split_counts": counts,
        "manifest": str(Path(manifest_path).resolve()),
        "sandbox_image": config.project.sandbox_image,
        "sandbox_image_id": image_id,
        "canary_passed": True,
        "buggy_hidden_coverage_checked": buggy_hidden_coverage_checked,
        "external_canary_passed": True,
    }
    context.record("prepare", result)
    return result
