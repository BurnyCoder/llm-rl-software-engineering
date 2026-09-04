"""Global context: verify local GPU, Docker, Hub identity, and pinned model generation.

Sources:
- https://pytorch.org/docs/stable/notes/cuda.html
- https://huggingface.co/docs/huggingface_hub/package_reference/hf_api
- https://huggingface.co/docs/transformers/main/en/main_classes/text_generation
"""

from __future__ import annotations

import gc
import os
import shutil
import subprocess
from typing import Any

import torch
from huggingface_hub import HfApi

from minibug_rl.context import PipelineContext
from minibug_rl.modeling import (
    load_tokenizer,
    load_transformers_model,
    validate_prompt_lengths,
)
from minibug_rl.prompts import build_messages, render_messages
from minibug_rl.task_data import load_tasks


def _required_command(name: str) -> str:
    """Resolve a required executable or fail before expensive allocations."""
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"Required command {name!r} is not installed or not on PATH")
    return path


def run_preflight(context: PipelineContext) -> dict[str, Any]:
    """Exercise one real pinned-model generation and capture hardware evidence."""
    logger = context.logger
    logger.message("phase_start", "Starting hardware and model preflight.", phase="preflight")
    nvidia_smi = _required_command("nvidia-smi")
    docker = _required_command("docker")
    # A daemon query proves Docker CLI presence is not a false-positive installation check.
    docker_version = subprocess.run(
        [docker, "version", "--format", "{{.Server.Version}}"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    gpu_csv = subprocess.run(
        [
            nvidia_smi,
            "--query-gpu=name,memory.total,memory.used,memory.free,compute_cap,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    if not torch.cuda.is_available():
        raise RuntimeError("The locked PyTorch build cannot access CUDA")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected GPU does not support BF16 training")
    # Run a small real BF16 kernel before attributing later failures to the model stack.
    probe = torch.ones((256, 256), device="cuda", dtype=torch.bfloat16)
    probe_sum = float((probe @ probe).float().sum().cpu())
    del probe
    api = HfApi(token=os.getenv("HF_TOKEN") or None)
    model_info = api.model_info(
        context.config.model.base_model,
        revision=context.config.model.revision,
    )
    identity = api.whoami() if os.getenv("HF_TOKEN") else {"name": "anonymous"}
    # Log only the username; the HfApi never returns the supplied token itself.
    username = str(identity.get("name", "unknown"))
    tasks = load_tasks(context.config.project.data_file)
    tokenizer = load_tokenizer(context.config)
    lengths = validate_prompt_lengths(tasks, tokenizer, context.config.model.max_prompt_length)
    torch.cuda.reset_peak_memory_stats()
    model = load_transformers_model(context.config, device="cuda", for_training=False)
    first = tasks[0]
    prompt = render_messages(tokenizer, build_messages(first))
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=16,
            do_sample=False,
            repetition_penalty=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    completion = tokenizer.decode(
        generated[0, encoded["input_ids"].shape[1] :],
        skip_special_tokens=True,
    )
    logger.generation(
        task_id=first.id,
        split=first.split,
        prompt=prompt,
        completion=completion,
        metadata={"purpose": "preflight", "max_new_tokens": 16},
    )
    peak_bytes = int(torch.cuda.max_memory_allocated())
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    if free_bytes < 768 * 2**20:
        raise RuntimeError(
            "Less than 768 MiB GPU headroom remains after the model smoke generation"
        )
    result = {
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(0),
        "bf16_supported": True,
        "bf16_probe_sum": probe_sum,
        "gpu_nvidia_smi": gpu_csv,
        "gpu_total_bytes": int(total_bytes),
        "gpu_free_after_model_bytes": int(free_bytes),
        "model_generation_peak_bytes": peak_bytes,
        "docker_server_version": docker_version,
        "hub_username": username,
        "base_model": context.config.model.base_model,
        "base_revision_requested": context.config.model.revision,
        "base_revision_resolved": model_info.sha,
        "task_count": len(tasks),
        "prompt_tokens_min": min(lengths.values()),
        "prompt_tokens_max": max(lengths.values()),
    }
    logger.write_json("environment.json", result)
    context.record("preflight", result)
    # Release inference allocations before training constructs optimizer and rollout state.
    del generated, encoded, model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return result
