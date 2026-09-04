"""Global context: construct and run current TRL GRPO with a trainable LoRA adapter.

Sources:
- https://huggingface.co/docs/trl/grpo_trainer
- https://huggingface.co/docs/trl/main/en/peft_integration
- https://huggingface.co/docs/peft/package_reference/lora
"""

from __future__ import annotations

import gc
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset  # type: ignore[import-untyped]
from peft import LoraConfig
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments, set_seed
from transformers.trainer_utils import get_last_checkpoint
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.grpo_trainer import GRPOTrainer

from minibug_rl.config import RunConfig
from minibug_rl.modeling import (
    load_tokenizer,
    load_transformers_model,
    trainable_parameter_counts,
    validate_prompt_lengths,
)
from minibug_rl.reward import build_grpo_reward
from minibug_rl.run_logging import RunLogger, utc_timestamp
from minibug_rl.sandbox import DockerSandbox
from minibug_rl.task_data import load_tasks, tasks_for_split, training_rows


def build_lora_config(config: RunConfig) -> LoraConfig:
    """Create the explicit trainable adapter missing from DebugArena's short snippet."""
    # `all-linear` follows PEFT's architecture-independent QLoRA-style targeting guidance.
    return LoraConfig(
        task_type="CAUSAL_LM",
        r=config.training.lora_rank,
        lora_alpha=config.training.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        target_modules="all-linear",
        revision=config.model.revision,
    )


def build_grpo_config(
    config: RunConfig,
    output_directory: str | Path,
    *,
    run_name: str,
    trackio_space_id: str | None = None,
) -> GRPOConfig:
    """Map the checked local profile to fields supported by installed TRL 1.12."""
    # Prompt length is deliberately absent because current GRPOConfig has no such field.
    return GRPOConfig(
        output_dir=str(Path(output_directory).expanduser().resolve()),
        run_name=run_name,
        max_steps=config.training.max_steps,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        # Transformers 5 folds ratios into float `warmup_steps`; values below one are ratios.
        warmup_steps=config.training.warmup_ratio,
        max_grad_norm=config.training.max_grad_norm,
        num_generations=config.training.num_generations,
        max_completion_length=config.model.max_completion_length,
        temperature=config.training.temperature,
        top_p=config.training.top_p,
        top_k=0,
        repetition_penalty=1.0,
        beta=0.0,
        loss_type="dr_grpo",
        scale_rewards="none",
        mask_truncated_completions=True,
        remove_unused_columns=False,
        use_vllm=False,
        use_cache=False,
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="no",
        logging_strategy="steps",
        logging_steps=1,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=config.training.save_steps,
        save_total_limit=2,
        seed=config.training.seed,
        data_seed=config.training.seed,
        dataloader_num_workers=0,
        report_to="trackio",
        project="minibug-rl",
        trackio_space_id=trackio_space_id,
        log_completions=False,
    )


def _logged_values(history: list[dict[str, Any]], key: str) -> list[float]:
    """Collect one numeric trainer metric across step records for compact summaries."""
    return [
        float(record[key])
        for record in history
        if isinstance(record.get(key), (int, float))
        and not isinstance(record.get(key), bool)
    ]


def _trainable_snapshot(model: Any) -> dict[str, torch.Tensor]:
    """Copy small LoRA tensors to CPU so a completed run can prove an update occurred."""
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _adapter_change(
    model: Any,
    before: dict[str, torch.Tensor],
) -> tuple[int, float]:
    """Count changed trainable tensors and calculate their aggregate L2 delta."""
    changed = 0
    squared_delta = 0.0
    for name, parameter in model.named_parameters():
        if name not in before:
            continue
        current = parameter.detach().cpu()
        if not torch.equal(before[name], current):
            changed += 1
        difference = current.float() - before[name].float()
        squared_delta += float(torch.sum(difference * difference))
    return changed, math.sqrt(squared_delta)


class AuditCallback(TrainerCallback):
    """Mirror trainer metrics and stop a persistently signal-free GRPO run."""

    def __init__(self, logger: RunLogger, zero_variance_limit: int = 20) -> None:
        """Store the run logger and consecutive zero-variance health threshold."""
        self.logger = logger
        self.zero_variance_limit = zero_variance_limit
        self.zero_variance_streak = 0
        self.stopped_for_reward_collapse = False

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> TrainerControl:
        """Persist numeric metrics and request a stop after 20 collapsed reward groups."""
        # The callback API supplies unused trainer arguments for all event hooks.
        del args, kwargs
        current = logs or {}
        numeric = {
            key: float(value)
            for key, value in current.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        if numeric:
            self.logger.metric(int(state.global_step), **numeric)
        # Current TRL logs this fraction directly; tolerate namespaced variants.
        zero_fraction = next(
            (value for key, value in numeric.items() if key.endswith("frac_reward_zero_std")),
            None,
        )
        if zero_fraction is not None:
            self.zero_variance_streak = (
                self.zero_variance_streak + 1 if zero_fraction > 0.8 else 0
            )
        if self.zero_variance_streak >= self.zero_variance_limit:
            self.logger.message(
                "reward_collapse",
                "Stopping because more than 80% of groups had zero reward variance for 20 logs.",
                step=int(state.global_step),
            )
            self.stopped_for_reward_collapse = True
            control.should_training_stop = True
        return control


def run_training(
    config: RunConfig,
    logger: RunLogger,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Run one local GRPO experiment and save a reloadable adapter plus exact state."""
    destination = Path(output_directory).expanduser().resolve()
    if (destination / "training_result.json").exists():
        # A forced rerun receives a new recoverable directory instead of overwriting evidence.
        destination = destination.with_name(f"{destination.name}-rerun-{utc_timestamp()}")
    destination.mkdir(parents=True, exist_ok=True)
    # TRL attaches PEFT before Trainer's own seed call, so seed before model construction.
    set_seed(config.training.seed)
    tasks = tasks_for_split(load_tasks(config.project.data_file), "train")
    # Explicit model/tokenizer loading pins both artifacts to the declared base revision.
    tokenizer = load_tokenizer(config)
    model = load_transformers_model(config, for_training=True)
    peft_config = build_lora_config(config)
    lengths = validate_prompt_lengths(tasks, tokenizer, config.model.max_prompt_length)
    dataset = Dataset.from_list(training_rows(tasks))
    arguments = build_grpo_config(
        config,
        destination / "checkpoints",
        # A phase suffix prevents Trackio from merging smoke and main step timelines.
        run_name=f"{logger.directory.name}-{destination.name}",
        trackio_space_id=os.getenv("TRACKIO_SPACE_ID") or None,
    )
    reward = build_grpo_reward(DockerSandbox(image=config.project.sandbox_image), logger)
    callback = AuditCallback(logger)
    checkpoint_directory = destination / "checkpoints"
    resume_checkpoint = (
        # Transformers does not expose annotations for this otherwise stable utility.
        get_last_checkpoint(str(checkpoint_directory))  # type: ignore[no-untyped-call]
        if checkpoint_directory.exists()
        else None
    )
    logger.message(
        "training_start",
        "Starting local GRPO training.",
        backend="transformers-peft",
        tasks=len(tasks),
        max_steps=config.training.max_steps,
        generations=config.training.num_generations,
        resume_checkpoint=resume_checkpoint,
    )
    # GRPOTrainer applies `peft_config` before optimizer creation for the standard backend.
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward,
        args=arguments,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[callback],
    )
    parameter_counts = trainable_parameter_counts(trainer.model)
    trainable_before = _trainable_snapshot(trainer.model)
    logger.message(
        "trainable_parameters",
        "Attached trainable policy parameters.",
        **parameter_counts,
    )
    torch.cuda.reset_peak_memory_stats()
    result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    changed_tensors, adapter_delta_l2 = _adapter_change(trainer.model, trainable_before)
    if int(trainer.state.global_step) != config.training.max_steps:
        raise RuntimeError(
            f"Training stopped at step {trainer.state.global_step}; "
            f"expected {config.training.max_steps}"
        )
    if not math.isfinite(float(result.training_loss)):
        raise RuntimeError("Training produced a non-finite loss")
    if changed_tensors == 0 or adapter_delta_l2 == 0.0:
        raise RuntimeError("Training completed without changing any LoRA tensor")
    adapter_directory = destination / "adapter"
    trainer.save_model(str(adapter_directory))
    tokenizer.save_pretrained(adapter_directory)
    adapter_config = json.loads(
        (adapter_directory / "adapter_config.json").read_text(encoding="utf-8")
    )
    if adapter_config.get("revision") != config.model.revision:
        raise RuntimeError("Saved adapter does not retain the pinned base-model revision")
    peak_memory = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    history = trainer.state.log_history
    history_path = destination / "trainer_log_history.json"
    history_path.write_text(
        json.dumps(history, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    clipped_ratios = _logged_values(history, "completions/clipped_ratio")
    reward_stds = _logged_values(history, "reward_std")
    summary = {
        "adapter_directory": str(adapter_directory),
        "global_step": int(trainer.state.global_step),
        "training_loss": float(result.training_loss),
        "peak_vram_bytes": peak_memory,
        "prompt_tokens_max": max(lengths.values()),
        "prompt_tokens_min": min(lengths.values()),
        "parameters": parameter_counts,
        "trainable_tensors_changed": changed_tensors,
        "adapter_delta_l2": adapter_delta_l2,
        "reward_zero_variance_streak": callback.zero_variance_streak,
        "stopped_for_reward_collapse": callback.stopped_for_reward_collapse,
        "maximum_clipped_ratio": max(clipped_ratios, default=0.0),
        "maximum_reward_std": max(reward_stds, default=0.0),
    }
    (destination / "training_result.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.message("training_complete", "Saved the trained LoRA adapter.", **summary)
    # Release optimizer, rollout, and model allocations before validation reloads the adapter.
    del trainer, model, tokenizer, dataset, trainable_before
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary
