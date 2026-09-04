"""Global context: verify the exact current TRL configuration before GPU allocation.

Sources:
- https://github.com/huggingface/trl/blob/v1.12.0/trl/trainer/grpo_trainer.py
- https://docs.python.org/3.12/library/math.html#math.isfinite
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from minibug_rl.config import load_run_config
from minibug_rl.run_logging import RunLogger
from minibug_rl.training import (
    AuditCallback,
    NonNullTrackioCallback,
    _require_finite_training_evidence,
    build_grpo_config,
    build_lora_config,
)


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
    assert arguments.scale_rewards == "none"
    assert arguments.mask_truncated_completions is True
    assert arguments.beta == 0.0
    assert arguments.use_vllm is False
    assert arguments.report_to == []
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


def test_audit_callback_stops_without_serializing_non_finite_metrics(tmp_path: Path) -> None:
    """Retain metric names as valid JSON while excluding NaN and infinity values."""
    # A real logger proves both the durable diagnostic and metrics stream stay parseable.
    logger = RunLogger.create(tmp_path, run_id="non-finite-callback")
    callback = AuditCallback(logger)
    state = SimpleNamespace(global_step=7)
    control = SimpleNamespace(should_training_stop=False)

    # One finite sibling proves useful metrics remain available when bad values are filtered.
    callback.on_log(
        SimpleNamespace(),
        state,
        control,
        logs={"loss": float("nan"), "reward": float("inf"), "learning_rate": 1e-5},
    )
    logger.close()

    assert callback.stopped_for_non_finite_metrics is True
    assert callback.non_finite_metric_names == ["loss", "reward"]
    assert control.should_training_stop is True
    # The event contains only string names, so standard JSON never receives NaN or Infinity.
    run_records = [
        json.loads(line)
        for line in (logger.directory / "run.log").read_text(encoding="utf-8").splitlines()
    ]
    assert run_records == [
        {
            "timestamp": run_records[0]["timestamp"],
            "event": "non_finite_training_metric",
            "message": "Stopping because trainer metrics contained non-finite values.",
            "step": 7,
            "metric_names": ["loss", "reward"],
        }
    ]
    metric_text = (logger.directory / "metrics.jsonl").read_text(encoding="utf-8")
    assert "NaN" not in metric_text
    assert "Infinity" not in metric_text
    assert json.loads(metric_text)["learning_rate"] == pytest.approx(1e-5)


@pytest.mark.parametrize(
    ("history", "training_loss", "adapter_delta_l2", "field"),
    [
        ([{"loss": float("nan")}], 0.1, 1.0, "history[0].loss"),
        ([{"reward": float("inf")}], 0.1, 1.0, "history[0].reward"),
        ([{"loss": 0.1}], float("-inf"), 1.0, "training_loss"),
        ([{"loss": 0.1}], 0.1, float("nan"), "adapter_delta_l2"),
    ],
)
def test_post_train_finite_guard_rejects_every_non_finite_source(
    history: list[dict[str, float]],
    training_loss: float,
    adapter_delta_l2: float,
    field: str,
) -> None:
    """Block corrupt trainer evidence and adapter updates before final artifact saving."""
    with pytest.raises(RuntimeError, match="non-finite") as error:
        _require_finite_training_evidence(history, training_loss, adapter_delta_l2)
    assert field in str(error.value)


def test_post_train_finite_guard_accepts_finite_numeric_history() -> None:
    """Allow ordinary finite metrics, including integral step counters and booleans."""
    _require_finite_training_evidence(
        [{"loss": 0.25, "epoch": 1, "is_final": True, "label": "train"}],
        0.25,
        1.5,
    )


def test_trackio_callback_drops_none_without_losing_defined_metrics() -> None:
    """Avoid Trackio's pyplot misclassification when TRL logs undefined extrema."""
    captured: list[dict[str, float | int]] = []
    callback = NonNullTrackioCallback()
    # Bypass network-free callback setup so this unit test observes only metric filtering.
    callback._initialized = True
    callback._trackio = SimpleNamespace(log=captured.append)

    callback.on_log(
        SimpleNamespace(),
        SimpleNamespace(is_world_process_zero=True, global_step=2),
        SimpleNamespace(),
        logs={"loss": 0.0, "clip_ratio/low_min": None, "reward": -0.3},
    )

    assert captured == [{"train/loss": 0.0, "train/reward": -0.3, "train/global_step": 2}]
