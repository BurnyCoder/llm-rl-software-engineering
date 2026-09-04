"""Global context: run a two-step GRPO gate before allocating the main experiment.

Source: https://github.com/huggingface/trl/blob/v1.12.0/trl/trainer/grpo_trainer.py
"""

from __future__ import annotations

import gc
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from minibug_rl.context import PipelineContext
from minibug_rl.modeling import load_transformers_model
from minibug_rl.training import run_training


def run_smoke(context: PipelineContext) -> dict[str, Any]:
    """Derive the bounded smoke profile and prove adapter save/reload inputs exist."""
    base = context.config
    smoke_model = replace(
        base.model,
        max_completion_length=min(base.model.max_completion_length, 128),
    )
    smoke_training = replace(
        base.training,
        max_steps=2,
        num_generations=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        save_steps=1,
    )
    smoke_config = replace(base, model=smoke_model, training=smoke_training)
    destination = base.project.artifact_root / "runs" / context.logger.directory.name / "smoke"
    context.logger.message("phase_start", "Running the two-step GRPO smoke gate.", phase="smoke")
    result = run_training(
        smoke_config,
        context.logger,
        destination,
        sandbox_image=context.prepared_sandbox_image(),
    )
    adapter_path = Path(str(result["adapter_directory"]))
    # A real PEFT reload catches incomplete checkpoints before the 100-step allocation.
    reloaded = load_transformers_model(
        smoke_config,
        adapter_path=adapter_path,
        device="cuda",
        for_training=False,
    )
    reloaded.eval()
    result["adapter_reload_verified"] = True
    del reloaded
    gc.collect()
    torch.cuda.empty_cache()
    context.record("smoke", result)
    return result
