"""Global context: lock resumable phase ordering and run-identity safety contracts.

Sources:
- https://docs.python.org/3/library/unittest.mock.html
- https://docs.python.org/3/library/hashlib.html
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from minibug_rl.config import load_run_config
from minibug_rl.context import PipelineContext
from minibug_rl.pipeline import PHASES, run_phases
from minibug_rl.run_logging import RunLogger


def _context(tmp_path: Path) -> PipelineContext:
    """Open a real context with repository data and isolated test logs."""
    repository = Path(__file__).resolve().parents[1]
    config = load_run_config(repository / "configs" / "smoke.toml", environ={})
    logger = RunLogger.create(tmp_path, run_id="pipeline-test")
    return PipelineContext.open(config, logger)


def test_pipeline_runs_in_requested_order_and_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Call named phases once, skip recorded work, and rerun only under force."""
    context = _context(tmp_path)
    calls: list[str] = []

    def fake_phase(name: str) -> Any:
        """Create a phase function that records and returns its own name."""

        def run(selected: PipelineContext) -> dict[str, Any]:
            """Record completion exactly as a production phase must."""
            calls.append(name)
            result = {"name": name}
            selected.record(name, result)
            return result

        return run

    for name in PHASES:
        monkeypatch.setitem(PHASES, name, fake_phase(name))
    selected = list(PHASES)

    run_phases(context, selected)
    run_phases(context, selected)
    run_phases(context, selected, force=True)

    assert calls == selected + selected
    context.logger.close()


def test_individual_phase_requires_prior_phase(tmp_path: Path) -> None:
    """Fail before an isolated later phase can consume absent or stale artifacts."""
    context = _context(tmp_path)

    with pytest.raises(RuntimeError, match="requires completed phase 'preflight'"):
        run_phases(context, ["prepare"])

    context.logger.close()


def test_resume_rejects_changed_resolved_configuration(tmp_path: Path) -> None:
    """Bind saved state to public config and data rather than only a reusable run ID."""
    first = _context(tmp_path)
    first.logger.close()
    state_path = first.logger.directory / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["_identity"]["config_sha256"] = "0" * 64
    state_path.write_text(json.dumps(state), encoding="utf-8")
    logger = RunLogger.create(tmp_path, run_id="pipeline-test")

    with pytest.raises(ValueError, match="does not match the resolved configuration"):
        PipelineContext.open(first.config, logger)

    logger.close()


def test_context_requires_an_immutable_prepared_sandbox_image(tmp_path: Path) -> None:
    """Expose only the Docker digest recorded by preparation to later phases."""
    # A new context has no trusted image until the prepare phase completes.
    context = _context(tmp_path)
    with pytest.raises(RuntimeError, match="prepared sandbox image"):
        context.prepared_sandbox_image()

    # Docker image IDs are algorithm-qualified immutable content identifiers.
    image_id = "sha256:" + "a" * 64
    context.state["prepare"] = {"sandbox_image_id": image_id}

    # Training and evaluation consume the immutable ID rather than the mutable local tag.
    assert context.prepared_sandbox_image() == image_id
    context.logger.close()
