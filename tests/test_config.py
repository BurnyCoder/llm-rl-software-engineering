"""Global context: lock the public TOML configuration contract before implementation.

Sources:
- https://docs.python.org/3/library/tomllib.html
- https://github.com/huggingface/trl/blob/v1.12.0/trl/trainer/grpo_trainer.py
"""

from pathlib import Path

import pytest

from minibug_rl.config import ConfigurationError, load_run_config


def test_load_run_config_resolves_paths_and_environment_override(tmp_path: Path) -> None:
    """A checked TOML profile remains portable while public operator values may use `.env`."""
    # Keep the fixture minimal so every asserted field belongs to the public contract.
    config_path = tmp_path / "local.toml"
    # TOML is used because Python 3.12 reads it without an extra parser dependency.
    config_path.write_text(
        """
[project]
name = "test-run"
data_file = "data/tasks.json"
run_root = "logs"
artifact_root = "artifacts"
sandbox_image = "minibug-runner:test"

[model]
base_model = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
revision = "pinned-revision"
hub_model_id = "owner/default"
dtype = "bfloat16"
max_prompt_length = 512
max_completion_length = 128

[training]
seed = 42
max_steps = 2
num_generations = 2
per_device_train_batch_size = 1
gradient_accumulation_steps = 2
learning_rate = 0.00001
temperature = 0.9
top_p = 0.95
save_steps = 1

[evaluation]
sample_generations = 2
bootstrap_samples = 100
""".strip(),
        encoding="utf-8",
    )

    # Only non-secret settings are accepted from the environment overlay.
    config = load_run_config(
        config_path,
        environ={"HF_MODEL_REPO": "owner/override", "HF_TOKEN": "must-not-appear"},
    )

    # Relative artifact paths resolve against the configuration's repository directory.
    assert config.project.data_file == (tmp_path / "data/tasks.json").resolve()
    assert config.model.hub_model_id == "owner/override"
    # Tokens never become configuration fields that could be serialized into run logs.
    assert "must-not-appear" not in repr(config)


def test_grpo_effective_batch_must_be_divisible_by_generation_count(tmp_path: Path) -> None:
    """Reject a configuration that TRL cannot distribute into complete rollout groups."""
    # This fixture deliberately gives an effective batch of two for four generations.
    config_path = tmp_path / "invalid.toml"
    config_path.write_text(
        """
[project]
name = "invalid"
data_file = "tasks.json"
run_root = "logs"
artifact_root = "artifacts"
sandbox_image = "runner:test"
[model]
base_model = "model"
revision = "revision"
hub_model_id = "owner/model"
dtype = "bfloat16"
max_prompt_length = 128
max_completion_length = 64
[training]
seed = 42
max_steps = 2
num_generations = 4
per_device_train_batch_size = 1
gradient_accumulation_steps = 2
learning_rate = 0.00001
temperature = 0.9
top_p = 0.95
save_steps = 1
[evaluation]
sample_generations = 2
bootstrap_samples = 100
""".strip(),
        encoding="utf-8",
    )

    # Fail before allocating a model or GPU for a configuration TRL would reject later.
    with pytest.raises(ConfigurationError, match="divisible"):
        load_run_config(config_path, environ={})
