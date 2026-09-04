"""Global context: persist non-secret phase state across resumable CLI invocations.

Sources:
- https://docs.python.org/3/library/dataclasses.html
- https://docs.python.org/3/library/json.html
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from minibug_rl.config import RunConfig
from minibug_rl.run_logging import RunLogger


def _sha256_bytes(value: bytes) -> str:
    """Return a stable hexadecimal digest for public reproducibility inputs."""
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    """Hash one required immutable input file without normalizing its bytes."""
    return _sha256_bytes(path.read_bytes())


def _source_identity(repository: Path) -> tuple[str, str]:
    """Identify both the Git commit and any tracked source diff used by this run."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            timeout=10,
        ).stdout.decode("utf-8").strip()
        difference = subprocess.run(
            [
                "git",
                "diff",
                "--binary",
                "HEAD",
                "--",
                "src",
                "sandbox",
                "configs",
                "pyproject.toml",
                "uv.lock",
            ],
            cwd=repository,
            check=True,
            capture_output=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        # Export remains possible outside Git, but the explicit sentinel is auditable.
        return "unavailable", "unavailable"
    return commit, _sha256_bytes(difference)


def _run_identity(config: RunConfig) -> dict[str, Any]:
    """Fingerprint config, curriculum, manifest, and source before resumable work."""
    public_config = json.dumps(
        config.public_dict(),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    data_path = config.project.data_file
    manifest_path = data_path.with_name("split_manifest.json")
    repository = data_path.parent.parent
    commit, difference = _source_identity(repository)
    return {
        "schema_version": 1,
        "config_sha256": _sha256_bytes(public_config),
        "data_sha256": _file_sha256(data_path),
        "manifest_sha256": _file_sha256(manifest_path),
        "source_commit": commit,
        "source_diff_sha256": difference,
    }


@dataclass
class PipelineContext:
    """Bundle immutable configuration, run logging, and resumable public state."""

    config: RunConfig
    logger: RunLogger
    state: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def open(cls, config: RunConfig, logger: RunLogger) -> PipelineContext:
        """Load an existing state file when resuming the same timestamped run."""
        state_path = logger.directory / "state.json"
        identity = _run_identity(config)
        if not state_path.exists():
            context = cls(config, logger, {"_identity": identity})
            logger.write_json("state.json", context.state)
            return context
        loaded = json.loads(state_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("Run state must be a JSON object")
        if loaded.get("_identity") != identity:
            raise ValueError(
                "Existing run state does not match the resolved configuration, data, or source"
            )
        return cls(config, logger, loaded)

    def record(self, phase: str, result: dict[str, Any]) -> None:
        """Atomically persist one completed phase result under its readable name."""
        self.state[phase] = result
        self.logger.write_json("state.json", self.state)
        self.logger.message("phase_complete", f"Completed phase {phase}.", phase=phase)
