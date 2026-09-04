"""Global context: measure untouched base validation behavior before any RL update.

Source: https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/generation/utils.py
"""

from __future__ import annotations

from typing import Any

from minibug_rl.context import PipelineContext
from minibug_rl.evaluation import evaluate_internal


def run_baseline(context: PipelineContext) -> dict[str, Any]:
    """Evaluate the pinned base only on validation without using final model outcomes."""
    context.logger.message("phase_start", "Evaluating the untouched base model.", phase="baseline")
    result = evaluate_internal(
        context.config,
        context.logger,
        split="validation",
        label="base-validation",
        sandbox_image=context.prepared_sandbox_image(),
    )
    context.record("baseline", result)
    return result
