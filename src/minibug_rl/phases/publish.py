"""Global context: publish and independently re-download the verified resulting model.

Sources:
- https://huggingface.co/docs/huggingface_hub/guides/upload
- https://huggingface.co/docs/huggingface_hub/guides/download
"""

from __future__ import annotations

import gc
import os
from typing import Any

import torch
from huggingface_hub import HfApi, snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from minibug_rl.context import PipelineContext
from minibug_rl.prompts import build_messages, render_messages
from minibug_rl.task_data import load_tasks


def run_publish(context: PipelineContext) -> dict[str, Any]:
    """Upload the Hub tree, download its immutable revision, and run one generation."""
    logger = context.logger
    logger.message("phase_start", "Publishing the verified model to Hugging Face.", phase="publish")
    exported = context.state.get("export")
    if not isinstance(exported, dict):
        raise RuntimeError("The export phase must finish before publication")
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required for the explicitly requested model publication")
    api = HfApi(token=token)
    username = str(api.whoami()["name"])
    namespace = context.config.model.hub_model_id.split("/", 1)[0]
    if username.casefold() != namespace.casefold():
        raise RuntimeError(
            f"Authenticated Hub user {username!r} does not own namespace {namespace!r}"
        )
    api.create_repo(
        repo_id=context.config.model.hub_model_id,
        repo_type="model",
        private=False,
        exist_ok=True,
    )
    commit = api.upload_folder(
        repo_id=context.config.model.hub_model_id,
        repo_type="model",
        folder_path=str(exported["hub_directory"]),
        commit_message="Publish measured MiniBug-RL model, adapter, and results",
    )
    verification_directory = (
        context.config.project.artifact_root / "hub-verification" / str(commit.oid)
    )
    snapshot_path = snapshot_download(
        repo_id=context.config.model.hub_model_id,
        revision=str(commit.oid),
        local_dir=verification_directory,
        token=token,
    )
    tokenizer = AutoTokenizer.from_pretrained(snapshot_path, local_files_only=True)
    model: Any = AutoModelForCausalLM.from_pretrained(
        snapshot_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.to("cuda")
    task = load_tasks(context.config.project.data_file)[0]
    prompt = render_messages(tokenizer, build_messages(task))
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=32,
            do_sample=False,
            repetition_penalty=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    completion = tokenizer.decode(
        generated[0, encoded["input_ids"].shape[1] :], skip_special_tokens=True
    )
    logger.generation(
        task_id=task.id,
        split=task.split,
        prompt=prompt,
        completion=completion,
        metadata={"purpose": "public-hub-redownload", "revision": str(commit.oid)},
    )
    result = {
        "repo_id": context.config.model.hub_model_id,
        "url": f"https://huggingface.co/{context.config.model.hub_model_id}",
        "revision": str(commit.oid),
        "verified_snapshot": str(snapshot_path),
        "redownload_generation_completed": True,
    }
    del model, tokenizer, encoded, generated
    gc.collect()
    torch.cuda.empty_cache()
    context.record("publish", result)
    return result
