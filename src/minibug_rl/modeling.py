"""Global context: load pinned tokenizer/model objects for training, evaluation, and export.

Sources:
- https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct
- https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained
- https://huggingface.co/docs/transformers/attention_interface
- https://huggingface.co/docs/peft/package_reference/peft_model
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from minibug_rl.config import RunConfig
from minibug_rl.prompts import build_messages, render_messages
from minibug_rl.schemas import RepairTask


def torch_dtype(name: str) -> torch.dtype:
    """Map the checked textual precision to a concrete PyTorch dtype."""
    # Keep accepted spellings intentionally narrow so a typo cannot silently use FP32.
    mapping = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    try:
        return mapping[name]
    except KeyError as error:
        raise ValueError(f"Unsupported model dtype {name!r}") from error


def load_tokenizer(config: RunConfig) -> Any:
    """Load the tokenizer at the exact base revision and configure left padding for GRPO."""
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.base_model,
        revision=config.model.revision,
        trust_remote_code=False,
    )
    # Decoder-only batched generation must pad on the left to align the newest tokens.
    tokenizer.padding_side = "left"
    # Qwen supplies an EOS token; reusing it as padding avoids changing embeddings.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_transformers_model(
    config: RunConfig,
    *,
    adapter_path: str | Path | None = None,
    device: str | torch.device | None = None,
    for_training: bool = False,
) -> Any:
    """Load pinned BF16 Qwen and optionally attach a saved PEFT adapter."""
    model: Any = AutoModelForCausalLM.from_pretrained(
        config.model.base_model,
        revision=config.model.revision,
        dtype=torch_dtype(config.model.dtype),
        attn_implementation="sdpa",
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    )
    if adapter_path is not None:
        # PEFT keeps the base immutable and loads only the small learned delta.
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=for_training)
    # Cached activations conflict with gradient checkpointing during training.
    model.config.use_cache = not for_training
    if device is not None:
        model.to(device)
    return model


def trainable_parameter_counts(model: Any) -> dict[str, int | float]:
    """Count trainable and total parameters for logs and model-card evidence."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "trainable": trainable,
        "total": total,
        "trainable_percent": (100.0 * trainable / total) if total else 0.0,
    }


def validate_prompt_lengths(
    tasks: tuple[RepairTask, ...],
    tokenizer: Any,
    maximum: int,
) -> dict[str, int]:
    """Measure native chat-template prompts because current GRPO has no truncation setting."""
    lengths: dict[str, int] = {}
    for task in tasks:
        prompt = render_messages(tokenizer, build_messages(task))
        # `add_special_tokens=False` avoids duplicating control tokens already in the template.
        token_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        lengths[task.id] = len(token_ids)
    over_limit = {task_id: length for task_id, length in lengths.items() if length > maximum}
    if over_limit:
        details = ", ".join(f"{task_id}={length}" for task_id, length in sorted(over_limit.items()))
        raise ValueError(f"Prompts exceed the {maximum}-token limit: {details}")
    return lengths
