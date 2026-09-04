"""Global context: thin readable wrapper over independently testable pipeline phases.

Source: https://docs.python.org/3/tutorial/modules.html
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from minibug_rl.context import PipelineContext
from minibug_rl.phases.baseline import run_baseline
from minibug_rl.phases.evaluate import run_evaluate
from minibug_rl.phases.export import run_export
from minibug_rl.phases.preflight import run_preflight
from minibug_rl.phases.prepare import run_prepare
from minibug_rl.phases.publish import run_publish
from minibug_rl.phases.smoke import run_smoke
from minibug_rl.phases.train import run_train

# The wrapper exposes the exact lifecycle in the same order described by the guide.
PHASES: dict[str, Callable[[PipelineContext], dict[str, Any]]] = {
    "preflight": run_preflight,
    "prepare": run_prepare,
    "baseline": run_baseline,
    "smoke": run_smoke,
    "train": run_train,
    "evaluate": run_evaluate,
    "export": run_export,
    "publish": run_publish,
}

# Each isolated CLI phase requires durable evidence from its immediate predecessor.
_PREREQUISITES: dict[str, tuple[str, ...]] = {
    "preflight": (),
    "prepare": ("preflight",),
    "baseline": ("prepare",),
    "smoke": ("baseline",),
    "train": ("smoke",),
    "evaluate": ("baseline", "smoke"),
    "export": ("evaluate",),
    "publish": ("export",),
}


def _require_prerequisites(context: PipelineContext, phase: str) -> None:
    """Reject out-of-order individual phases before they allocate or publish."""
    missing = [name for name in _PREREQUISITES[phase] if name not in context.state]
    if missing:
        required = "', '".join(missing)
        raise RuntimeError(f"Phase {phase!r} requires completed phase '{required}'")


def run_phases(
    context: PipelineContext,
    phases: Iterable[str],
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Run named phases in order while safely skipping completed resumable work."""
    results: dict[str, Any] = {}
    for phase in phases:
        if phase not in PHASES:
            raise ValueError(f"Unknown pipeline phase {phase!r}")
        if phase in context.state and not force:
            context.logger.message(
                "phase_skipped",
                f"Skipping completed phase {phase}; use --force to rerun it.",
                phase=phase,
            )
            results[phase] = context.state[phase]
            continue
        _require_prerequisites(context, phase)
        results[phase] = PHASES[phase](context)
    return results
