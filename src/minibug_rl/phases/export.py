"""Global context: turn a selected adapter into local portable model artifacts.

Source: https://huggingface.co/docs/peft/v0.20.0/package_reference/peft_model
"""

from __future__ import annotations

from typing import Any

from minibug_rl.artifacts import export_artifacts
from minibug_rl.context import PipelineContext


def run_export(context: PipelineContext) -> dict[str, Any]:
    """Merge, save, and reload both trained artifact forms locally."""
    context.logger.message("phase_start", "Exporting selected model artifacts.", phase="export")
    evaluation = context.state.get("evaluate")
    if not isinstance(evaluation, dict):
        raise RuntimeError("The evaluate phase must finish before export")
    result = export_artifacts(context.config, context.logger, evaluation)
    context.record("export", result)
    return result
