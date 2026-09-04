"""Specify that the generated model card renders measured evaluation metadata.

Global context: published claims must come from immutable result artifacts rather than
hardcoded assumptions that can drift from the protocol actually executed.

Source: https://huggingface.co/docs/hub/model-cards
"""

from __future__ import annotations

# Paths load the checked smoke profile without inventing a second configuration fixture.
from pathlib import Path

# The focused renderer is pure text generation and performs no model or file mutation.
from minibug_rl.artifacts import _model_card

# The public loader supplies realistic model lineage and training hyperparameters.
from minibug_rl.config import load_run_config


def test_model_card_renders_external_task_count_protocol_timeout_and_failures() -> None:
    """Keep benchmark scope and sandbox caveats tied to measured JSON fields."""
    # Load the small checked profile only to populate stable lineage/configuration fields.
    repository = Path(__file__).resolve().parents[1]
    config = load_run_config(repository / "configs" / "smoke.toml", environ={})
    # Internal summaries provide the table fields required by the shared renderer.
    internal_base = {
        "sampled_k": 2,
        "sampled_pass_at_2": 0.25,
        "greedy_pass_at_1": 0.25,
        "greedy_hidden_test_fraction": 0.5,
    }
    internal_selected = {
        "sampled_k": 2,
        "sampled_pass_at_2": 0.5,
        "greedy_pass_at_1": 0.5,
        "greedy_hidden_test_fraction": 0.75,
    }
    # Deliberately non-production values prove the card does not hardcode 164 or 3 seconds.
    external_base = {
        "tasks": 3,
        "pass_at_1": 1 / 3,
        "timeouts": 1,
        "benchmark": "bigcode/humanevalpack",
        "benchmark_revision": "dataset-pin",
        "prompt_variant": "humanevalfixdocs-python",
        "sandbox_timeout_seconds": 7.5,
        "sandbox_image": "sha256:" + "e" * 64,
    }
    external_selected = {**external_base, "pass_at_1": 2 / 3, "timeouts": 0}
    evaluation = {
        "base_test": {"summary": internal_base},
        "selected_test": {"summary": internal_selected},
        "test_interval": {"mean_difference": 0.25, "lower_95": 0.0, "upper_95": 0.5},
        "external_base": {"summary": external_base},
        "external_selected": {"summary": external_selected},
        "external_interval": {
            "mean_difference": 1 / 3,
            "lower_95": 0.0,
            "upper_95": 2 / 3,
        },
        "selected_candidate": "train",
        "learning_success": True,
    }

    # A synthetic commit is inert but makes the lineage URL structurally complete.
    card = _model_card(config, evaluation, "f" * 40)

    # Every assertion targets an external field whose hardcoding would misstate the run.
    assert "| Greedy pass@1 (3 Python repairs) |" in card
    assert "| Timeouts | 1 | 0 |" in card
    assert "`humanevalfixdocs-python`" in card
    assert "`dataset-pin`" in card
    assert "7.5-second Docker deadline" in card
