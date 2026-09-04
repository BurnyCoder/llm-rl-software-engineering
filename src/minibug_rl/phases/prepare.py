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
    }
    context.record("prepare", result)
    return result
