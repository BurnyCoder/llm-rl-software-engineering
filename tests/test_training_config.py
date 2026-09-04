"""Global context: verify the exact current TRL configuration before GPU allocation.

Source: https://huggingface.co/docs/trl/grpo_trainer
"""

from pathlib import Path
from types import SimpleNamespace

from minibug_rl.config import load_run_config
from minibug_rl.run_logging import RunLogger
from minibug_rl.training import AuditCallback, build_grpo_config, build_lora_config


def test_smoke_profile_maps_only_to_supported_current_grpo_fields(tmp_path: Path) -> None:
    """Prevent obsolete guide arguments such as `max_prompt_length` reaching TRL."""
    # The checked profile is portable because its own relative paths resolve consistently.
    profile = Path(__file__).resolve().parents[1] / "configs" / "smoke.toml"
    config = load_run_config(profile, environ={})

    arguments = build_grpo_config(config, tmp_path, run_name="unit-smoke")

    assert arguments.max_steps == 2
    assert arguments.num_generations == 2
    assert arguments.max_completion_length == 128
    assert arguments.top_k == 0
    assert arguments.repetition_penalty == 1.0
    assert arguments.loss_type == "dr_grpo"
    assert arguments.mask_truncated_completions is True
    assert arguments.beta == 0.0
    assert arguments.use_vllm is False
    assert not hasattr(arguments, "max_prompt_length")


def test_lora_profile_trains_all_linear_layers_with_no_dropout() -> None:
    """Attach the missing trainable adapter that DebugArena's short snippet omits."""
    profile = Path(__file__).resolve().parents[1] / "configs" / "smoke.toml"
    config = load_run_config(profile, environ={})

    lora = build_lora_config(config)

    assert lora.r == 16
    assert lora.lora_alpha == 32
    assert lora.lora_dropout == 0.0
    assert lora.target_modules == "all-linear"
    assert lora.revision == config.model.revision


def test_audit_callback_stops_and_retains_reward_collapse_state(tmp_path: Path) -> None:
    """Stop 20 low-variance groups and do not erase evidence on a summary log."""
    logger = RunLogger.create(tmp_path, run_id="callback-test")
    callback = AuditCallback(logger)
    state = SimpleNamespace(global_step=0)
    control = SimpleNamespace(should_training_stop=False)

    for step in range(1, 21):
        state.global_step = step
        callback.on_log(
            SimpleNamespace(),
            state,
            control,
            logs={"frac_reward_zero_std": 0.9},
        )
    callback.on_log(SimpleNamespace(), state, control, logs={"train_runtime": 1.0})

    assert callback.zero_variance_streak == 20
    assert callback.stopped_for_reward_collapse is True
    assert control.should_training_stop is True
    logger.close()
