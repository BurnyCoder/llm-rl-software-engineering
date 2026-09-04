"""Global context: publish and independently re-download the verified resulting model.

Sources:
- https://github.com/huggingface/huggingface_hub/blob/48ef2781c2c4c2247431c97efcd5487e89d42732/src/huggingface_hub/hf_api.py
- https://github.com/huggingface/huggingface_hub/blob/48ef2781c2c4c2247431c97efcd5487e89d42732/src/huggingface_hub/_snapshot_download.py
- https://docs.python.org/3.12/library/hashlib.html#hashlib.file_digest
"""

from __future__ import annotations

import gc
import hashlib
import os
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import HfApi, snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from minibug_rl.artifacts import PUBLICATION_ALLOWLIST, validate_publication_tree
from minibug_rl.context import PipelineContext
from minibug_rl.prompts import build_messages, render_messages
from minibug_rl.task_data import load_tasks


def _safetensors_sha256(directory: Path) -> dict[str, str]:
    """Hash every recursively discovered Safetensors file by portable relative path."""
    hashes: dict[str, str] = {}
    for path in sorted(directory.rglob("*.safetensors")):
        if path.is_file():
            with path.open("rb") as handle:
                hashes[path.relative_to(directory).as_posix()] = hashlib.file_digest(
                    handle, "sha256"
                ).hexdigest()
    return hashes


def _verify_safetensors(local_directory: Path, downloaded_directory: Path) -> dict[str, str]:
    """Require identical complete weight-file sets and SHA-256 values after upload."""
    local_hashes = _safetensors_sha256(local_directory)
    if not local_hashes:
        raise RuntimeError("Local publication tree contains no Safetensors weight files")
    downloaded_hashes = _safetensors_sha256(downloaded_directory)
    if set(local_hashes) != set(downloaded_hashes):
        missing = sorted(set(local_hashes) - set(downloaded_hashes))
        unexpected = sorted(set(downloaded_hashes) - set(local_hashes))
        raise RuntimeError(
            "Downloaded Safetensors file set differs from the local publication tree: "
            f"missing={missing}, unexpected={unexpected}"
        )
    mismatched = [
        relative_path
        for relative_path, digest in local_hashes.items()
        if downloaded_hashes[relative_path] != digest
    ]
    if mismatched:
        raise RuntimeError(f"Downloaded Safetensors hash mismatch: {sorted(mismatched)}")
    return local_hashes


def _validate_remote_publication_files(remote_files: list[str]) -> tuple[str, ...]:
    """Require all reviewed files and reject every non-Hub-managed remote extra."""
    actual_files = set(remote_files)
    # The Hub retains this repository metadata even when a matching delete pattern is used.
    hub_managed_files = {".gitattributes"}
    missing_files = sorted(PUBLICATION_ALLOWLIST - actual_files)
    unexpected_files = sorted(actual_files - PUBLICATION_ALLOWLIST - hub_managed_files)
    if missing_files or unexpected_files:
        raise RuntimeError(
            "Published Hub tree differs from its reviewed allowlist: "
            f"missing={missing_files}, unexpected={unexpected_files}"
        )
    return tuple(sorted(actual_files))


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
    hub_directory = Path(str(exported["hub_directory"]))
    # Validate again at the remote-write boundary in case local artifacts changed after export.
    publication_files = validate_publication_tree(hub_directory)
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
    # Lock the write to the inspected branch head so concurrent changes cannot be erased.
    parent_commit = str(api.model_info(context.config.model.hub_model_id).sha)
    commit = api.upload_folder(
        repo_id=context.config.model.hub_model_id,
        repo_type="model",
        folder_path=str(hub_directory),
        commit_message="Publish measured MiniBug-RL model, adapter, and results",
        parent_commit=parent_commit,
        allow_patterns=sorted(publication_files),
        # upload_folder otherwise leaves old remote files untouched; replace the reviewed tree.
        delete_patterns=["*", "**/*"],
    )
    remote_files = _validate_remote_publication_files(
        api.list_repo_files(
            repo_id=context.config.model.hub_model_id,
            repo_type="model",
            revision=str(commit.oid),
        )
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
    # Publication cannot complete until all merged and adapter weight bytes round-trip.
    safetensors_sha256 = _verify_safetensors(
        hub_directory,
        Path(snapshot_path),
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
    # Persist the complete generated value even when the publication smoke check rejects it.
    logger.generation(
        task_id=task.id,
        split=task.split,
        prompt=prompt,
        completion=completion,
        metadata={"purpose": "public-hub-redownload", "revision": str(commit.oid)},
    )
    if not isinstance(completion, str):
        raise RuntimeError("Exact-revision Hub tokenizer returned a non-string generation")
    if not completion.strip():
        raise RuntimeError("Exact-revision Hub reload produced an empty verification generation")
    result = {
        "repo_id": context.config.model.hub_model_id,
        "url": f"https://huggingface.co/{context.config.model.hub_model_id}",
        "revision": str(commit.oid),
        "verified_snapshot": str(snapshot_path),
        "safetensors_sha256": safetensors_sha256,
        "publication_files": publication_files,
        "remote_files": remote_files,
        "redownload_generation_completed": True,
    }
    del model, tokenizer, encoded, generated
    gc.collect()
    torch.cuda.empty_cache()
    context.record("publish", result)
    return result
