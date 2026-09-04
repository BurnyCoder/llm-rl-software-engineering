"""Global context: run the source-committed main local GRPO profile.

Source: https://github.com/huggingface/trl/blob/v1.12.0/trl/trainer/grpo_trainer.py
"""

from __future__ import annotations

from typing import Any

from minibug_rl.context import PipelineContext
from minibug_rl.training import run_training


def run_train(context: PipelineContext) -> dict[str, Any]:
    """Train and save the main candidate adapter under the timestamped run."""
    destination = (
        context.config.project.artifact_root / "runs" / context.logger.directory.name / "main"
    )
    context.logger.message("phase_start", "Running the main GRPO experiment.", phase="train")
    result = run_training(
        context.config,
        context.logger,
        destination,
        sandbox_image=context.prepared_sandbox_image(),
    )
    context.record("train", result)
    return result
