"""Global context: assemble portable local and Hugging Face model artifacts.

Sources:
- https://huggingface.co/docs/peft/package_reference/peft_model#peft.PeftModel.merge_and_unload
- https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.save_pretrained
- https://huggingface.co/docs/hub/model-cards
"""

from __future__ import annotations

import gc
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import torch

from minibug_rl.config import RunConfig
from minibug_rl.modeling import load_tokenizer, load_transformers_model
from minibug_rl.prompts import build_messages, render_messages
from minibug_rl.run_logging import RunLogger, utc_timestamp
from minibug_rl.task_data import load_tasks


def _backup_existing(path: Path) -> None:
    """Move an older final artifact aside rather than destructively deleting it."""
    if path.exists():
        backup = path.with_name(f"{path.name}.previous-{utc_timestamp()}")
        path.replace(backup)


def _copy_directory(source: Path, destination: Path) -> None:
    """Copy one verified model directory while rejecting an existing destination."""
    shutil.copytree(source, destination)


def _model_card(config: RunConfig, evaluation: dict[str, Any], source_commit: str) -> str:
    """Render a factual card from measured JSON rather than hand-entered performance claims."""
    base = evaluation["base_test"]["summary"]
    selected = evaluation["selected_test"]["summary"]
    interval = evaluation["test_interval"]
    external_base = evaluation["external_base"]["summary"]
    external_selected = evaluation["external_selected"]["summary"]
    external_interval = evaluation["external_interval"]
    external_tasks = int(external_selected["tasks"])
    external_timeout = float(external_selected["sandbox_timeout_seconds"])
    outcome = "passed" if evaluation["learning_success"] else "did not pass"
    sampled_k = int(selected["sampled_k"])
    base_sampled = float(base[f"sampled_pass_at_{int(base['sampled_k'])}"])
    selected_sampled = float(selected[f"sampled_pass_at_{sampled_k}"])
    source_url = f"https://github.com/BurnyCoder/llm-rl-software-engineering/commit/{source_commit}"
    lines = [
        "---",
        f"base_model: {config.model.base_model}",
        "library_name: transformers",
        "license: apache-2.0",
        "pipeline_tag: text-generation",
        "tags:",
        "- reinforcement-learning",
        "- grpo",
        "- code",
        "- peft",
        "---",
        "",
        "# Qwen2.5-Coder-0.5B MiniBug-RL",
        "",
        "This is the real merged output of the MiniBug-RL hidden-unit-test GRPO ",
        "experiment. It targets single-function Python repair; it is not a ",
        "repository-scale autonomous software-engineering agent.",
        "",
        "## Exact lineage",
        "",
        f"- Base: `{config.model.base_model}` at `{config.model.revision}`",
        f"- Source: {source_url}",
        (
            "- Trainer: TRL GRPO with LoRA rank "
            f"{config.training.lora_rank}, alpha {config.training.lora_alpha}, "
            f"learning rate {config.training.learning_rate}, seed {config.training.seed}"
        ),
        (
            f"- Selection: `{evaluation['selected_candidate']}` chosen only on the "
            "12-task validation split"
        ),
        "",
        "## Frozen final-test result",
        "",
        "| Metric | Base | Selected model |",
        "|---|---:|---:|",
        (
            f"| Greedy pass@1 | {base['greedy_pass_at_1']:.4f} | "
            f"{selected['greedy_pass_at_1']:.4f} |"
        ),
        (
            "| Greedy hidden-test fraction | "
            f"{base['greedy_hidden_test_fraction']:.4f} | "
            f"{selected['greedy_hidden_test_fraction']:.4f} |"
        ),
        (f"| Sampled pass@{sampled_k} | {base_sampled:.4f} | {selected_sampled:.4f} |"),
        "",
        (
            "Paired hidden-test-fraction difference: "
            f"`{interval['mean_difference']:.4f}` with bootstrap 95% interval "
            f"`[{interval['lower_95']:.4f}, {interval['upper_95']:.4f}]`."
        ),
        (
            f"The pre-registered validation learning gate **{outcome}**. These small "
            "synthetic-split measurements should not be generalized to SWE-bench."
        ),
        "",
        "## Frozen external HumanEvalFix result",
        "",
        "| Metric | Base | Selected model |",
        "|---|---:|---:|",
        (
            f"| Greedy pass@1 ({external_tasks} Python repairs) | "
            f"{external_base['pass_at_1']:.4f} | "
            f"{external_selected['pass_at_1']:.4f} |"
        ),
        f"| Timeouts | {int(external_base['timeouts'])} | {int(external_selected['timeouts'])} |",
        "",
        (
            f"Protocol: `{external_selected['benchmark']}` at immutable dataset revision "
            f"`{external_selected['benchmark_revision']}`, prompt variant "
            f"`{external_selected['prompt_variant']}`, greedy `n=1`."
        ),
        (
            "Paired pass@1 difference: "
            f"`{external_interval['mean_difference']:.4f}` with bootstrap 95% interval "
            f"`[{external_interval['lower_95']:.4f}, {external_interval['upper_95']:.4f}]`."
        ),
        (
            "HumanEvalFix was never used for training, reward design, checkpoint selection, "
            "or tuning. Public benchmark contamination may still affect both policies."
        ),
        (
            f"Candidates ran under MiniBug-RL's hardened {external_timeout:g}-second Docker "
            "deadline rather than the pinned BigCode Python harness's 10-second limit; treat "
            "this as a MiniBug-sandbox measurement, not a directly comparable leaderboard score."
        ),
        f"Sandbox image: `{external_selected['sandbox_image']}`.",
        "",
        "## Load the resulting model",
        "",
        "```python",
        "from transformers import AutoModelForCausalLM, AutoTokenizer",
        "",
        f'model_id = "{config.model.hub_model_id}"',
        "tokenizer = AutoTokenizer.from_pretrained(model_id)",
        'model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto")',
        "```",
        "",
        "The separately loadable LoRA adapter and tokenizer are in `adapter/`.",
        "Candidate-level measurements and the resolved configuration are included.",
    ]
    return "\n".join(lines) + "\n"


def _verification_generation(
    config: RunConfig,
    logger: RunLogger,
    model_path: str | Path,
    *,
    label: str,
    adapter_path: str | Path | None = None,
) -> str:
    """Reload one saved form and run a deterministic generation before publication."""
    tokenizer = load_tokenizer(config) if adapter_path is not None else None
    if adapter_path is None:
        # A merged directory is a standard Transformers model independent of the base path.
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=False)
        model: Any = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            trust_remote_code=False,
        )
        model.to("cuda")
    else:
        model = load_transformers_model(
            config,
            adapter_path=adapter_path,
            device="cuda",
            for_training=False,
        )
    task = load_tasks(config.project.data_file)[0]
    assert tokenizer is not None
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
        metadata={"purpose": label},
    )
    del model, tokenizer, encoded, generated
    gc.collect()
    torch.cuda.empty_cache()
    return str(completion)


def export_artifacts(
    config: RunConfig,
    logger: RunLogger,
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    """Save adapter, merged weights, Hub tree, measured results, and reload evidence."""
    final_directory = config.project.artifact_root / "final"
    _backup_existing(final_directory)
    final_directory.mkdir(parents=True)
    selected_adapter = Path(str(evaluation["selected_adapter"])).resolve()
    adapter_directory = final_directory / "adapter"
    merged_directory = final_directory / "merged"
    hub_directory = final_directory / "hub"
    _copy_directory(selected_adapter, adapter_directory)
    # Merge on CPU to keep scarce GPU memory free and preserve a standard model layout.
    model = load_transformers_model(config, adapter_path=selected_adapter, device="cpu")
    merged = model.merge_and_unload()
    merged.save_pretrained(merged_directory, safe_serialization=True, max_shard_size="2GB")
    tokenizer = load_tokenizer(config)
    tokenizer.save_pretrained(merged_directory)
    del merged, model, tokenizer
    gc.collect()
    # The Hub root contains the merged model so AutoModel can load the repository directly.
    _copy_directory(merged_directory, hub_directory)
    _copy_directory(adapter_directory, hub_directory / "adapter")
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    (hub_directory / "results.json").write_text(
        json.dumps(evaluation, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (hub_directory / "training_config.json").write_text(
        json.dumps(config.public_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (hub_directory / "README.md").write_text(
        _model_card(config, evaluation, source_commit),
        encoding="utf-8",
    )
    adapter_completion = _verification_generation(
        config,
        logger,
        config.model.base_model,
        label="local-adapter-reload",
        adapter_path=adapter_directory,
    )
    merged_completion = _verification_generation(
        config,
        logger,
        merged_directory,
        label="local-merged-reload",
    )
    outputs_match = adapter_completion == merged_completion
    if not outputs_match:
        logger.message(
            "reload_output_difference",
            "Both artifact forms loaded, but their greedy verification text differed.",
        )
    result = {
        "final_directory": str(final_directory),
        "adapter_directory": str(adapter_directory),
        "merged_directory": str(merged_directory),
        "hub_directory": str(hub_directory),
        "source_commit": source_commit,
        "reload_outputs_match": outputs_match,
    }
    (final_directory / "export_result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result
