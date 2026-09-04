"""Global context: load and validate the checked, non-secret MiniBug-RL run profile.

Sources:
- https://docs.python.org/3/library/tomllib.html
- https://huggingface.co/docs/trl/main/en/grpo_trainer
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class ConfigurationError(ValueError):
    """Identify invalid profiles before model allocation or external publication."""


@dataclass(frozen=True)
class ProjectConfig:
    """Hold repository paths and the immutable sandbox image tag."""

    name: str
    data_file: Path
    run_root: Path
    artifact_root: Path
    sandbox_image: str


@dataclass(frozen=True)
class ModelConfig:
    """Hold model identity, publication target, precision, and context limits."""

    base_model: str
    revision: str
    hub_model_id: str
    dtype: str
    max_prompt_length: int
    max_completion_length: int


@dataclass(frozen=True)
class TrainingConfig:
    """Hold the exact single-GPU GRPO and LoRA experiment settings."""

    seed: int
    max_steps: int
    num_generations: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    temperature: float
    top_p: float
    save_steps: int
    lora_rank: int = 16
    lora_alpha: int = 32
    warmup_ratio: float = 0.05
    max_grad_norm: float = 0.1


@dataclass(frozen=True)
class EvaluationConfig:
    """Hold deterministic evaluation and uncertainty-estimation settings."""

    sample_generations: int
    bootstrap_samples: int


@dataclass(frozen=True)
class RunConfig:
    """Combine validated public configuration sections without any credentials."""

    project: ProjectConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig

    def public_dict(self) -> dict[str, Any]:
        """Serialize paths and hyperparameters for reproducibility logs."""
        # Dataclass conversion cannot include HF_TOKEN because the schema has no secret field.
        data = asdict(self)
        for key in ("data_file", "run_root", "artifact_root"):
            data["project"][key] = str(data["project"][key])
        return data


def _required(section: Mapping[str, Any], name: str, section_name: str) -> Any:
    """Read a mandatory TOML value with a precise operator-facing error."""
    # Explicit membership avoids treating valid zero-like values as missing.
    if name not in section:
        raise ConfigurationError(f"Missing [{section_name}].{name}")
    return section[name]


def _positive(value: Any, label: str) -> int:
    """Validate positive integral sizes used by batches, steps, and token limits."""
    # Reject booleans because `bool` is an `int` subclass but not a meaningful size.
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigurationError(f"{label} must be a positive integer")
    return value


def _section(document: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    """Return one required table from the parsed TOML document."""
    # A mapping check catches misspelled or scalar section definitions early.
    value = document.get(name)
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"Missing or invalid [{name}] section")
    return value


def load_run_config(
    path: str | Path,
    *,
    environ: Mapping[str, str] | None = None,
) -> RunConfig:
    """Load one TOML profile and overlay only the non-secret Hub repository name."""
    # Resolve once so every relative artifact path has an unambiguous base directory.
    config_path = Path(path).expanduser().resolve()
    # `tomllib` parses bytes and never mutates the human-authored profile.
    with config_path.open("rb") as handle:
        document = tomllib.load(handle)
    project_data = _section(document, "project")
    model_data = _section(document, "model")
    training_data = _section(document, "training")
    evaluation_data = _section(document, "evaluation")
    # Dependency injection keeps tests from reading the process environment.
    environment = os.environ if environ is None else environ
    base_directory = config_path.parent

    def resolve_path(raw: Any) -> Path:
        """Resolve profile-relative paths while preserving explicit absolute paths."""
        candidate = Path(str(raw)).expanduser()
        return (candidate if candidate.is_absolute() else base_directory / candidate).resolve()

    project = ProjectConfig(
        name=str(_required(project_data, "name", "project")),
        data_file=resolve_path(_required(project_data, "data_file", "project")),
        run_root=resolve_path(_required(project_data, "run_root", "project")),
        artifact_root=resolve_path(_required(project_data, "artifact_root", "project")),
        sandbox_image=str(_required(project_data, "sandbox_image", "project")),
    )
    model = ModelConfig(
        base_model=str(_required(model_data, "base_model", "model")),
        revision=str(_required(model_data, "revision", "model")),
        hub_model_id=environment.get(
            "HF_MODEL_REPO",
            str(_required(model_data, "hub_model_id", "model")),
        ),
        dtype=str(_required(model_data, "dtype", "model")),
        max_prompt_length=_positive(
            _required(model_data, "max_prompt_length", "model"),
            "model.max_prompt_length",
        ),
        max_completion_length=_positive(
            _required(model_data, "max_completion_length", "model"),
            "model.max_completion_length",
        ),
    )
    training = TrainingConfig(
        seed=int(_required(training_data, "seed", "training")),
        max_steps=_positive(
            _required(training_data, "max_steps", "training"),
            "training.max_steps",
        ),
        num_generations=_positive(
            _required(training_data, "num_generations", "training"),
            "training.num_generations",
        ),
        per_device_train_batch_size=_positive(
            _required(training_data, "per_device_train_batch_size", "training"),
            "training.per_device_train_batch_size",
        ),
        gradient_accumulation_steps=_positive(
            _required(training_data, "gradient_accumulation_steps", "training"),
            "training.gradient_accumulation_steps",
        ),
        learning_rate=float(_required(training_data, "learning_rate", "training")),
        temperature=float(_required(training_data, "temperature", "training")),
        top_p=float(_required(training_data, "top_p", "training")),
        save_steps=_positive(
            _required(training_data, "save_steps", "training"),
            "training.save_steps",
        ),
        lora_rank=_positive(training_data.get("lora_rank", 16), "training.lora_rank"),
        lora_alpha=_positive(training_data.get("lora_alpha", 32), "training.lora_alpha"),
        warmup_ratio=float(training_data.get("warmup_ratio", 0.05)),
        max_grad_norm=float(training_data.get("max_grad_norm", 0.1)),
    )
    evaluation = EvaluationConfig(
        sample_generations=_positive(
            _required(evaluation_data, "sample_generations", "evaluation"),
            "evaluation.sample_generations",
        ),
        bootstrap_samples=_positive(
            _required(evaluation_data, "bootstrap_samples", "evaluation"),
            "evaluation.bootstrap_samples",
        ),
    )
    # TRL forms complete relative-reward groups from the effective optimizer batch.
    effective_batch = training.per_device_train_batch_size * training.gradient_accumulation_steps
    if effective_batch % training.num_generations != 0:
        raise ConfigurationError(
            "training effective batch must be divisible by training.num_generations"
        )
    # Sampling probabilities and warmup ratios have closed meaningful ranges.
    if training.temperature <= 0.0:
        raise ConfigurationError("training.temperature must be greater than zero")
    if not 0.0 < training.top_p <= 1.0:
        raise ConfigurationError("training.top_p must be in (0, 1]")
    if not 0.0 <= training.warmup_ratio < 1.0:
        raise ConfigurationError("training.warmup_ratio must be in [0, 1)")
    # Return one frozen object used unchanged by all pipeline phases.
    return RunConfig(project, model, training, evaluation)
