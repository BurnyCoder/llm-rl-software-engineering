"""Specify that the generated model card renders measured evaluation metadata.

Global context: published claims must come from immutable result artifacts rather than
hardcoded assumptions that can drift from the protocol actually executed.

Source: https://huggingface.co/docs/hub/model-cards
"""

from __future__ import annotations

# Paths load the checked smoke profile without inventing a second configuration fixture.
from pathlib import Path

# Parameterization exercises current and immutable historical result schemas equally.
import pytest

# The focused renderer is pure text generation and performs no model or file mutation.
from minibug_rl.artifacts import PUBLICATION_ALLOWLIST, _model_card, validate_publication_tree

# The public loader supplies realistic model lineage and training hyperparameters.
from minibug_rl.config import load_run_config


@pytest.mark.parametrize(
    "metric_key",
    ["observed_sampled_success_at_2", "sampled_pass_at_2"],
)
def test_model_card_renders_corrected_claims_from_current_or_historical_metric(
    metric_key: str,
) -> None:
    """Keep claims accurate while accepting current and immutable historical JSON."""
    # Load the small checked profile only to populate stable lineage/configuration fields.
    repository = Path(__file__).resolve().parents[1]
    config = load_run_config(repository / "configs" / "smoke.toml", environ={})
    # Internal summaries provide the table fields required by the shared renderer.
    internal_base = {
        "sampled_k": 2,
        metric_key: 0.25,
        "greedy_pass_at_1": 0.25,
        "greedy_hidden_test_fraction": 0.5,
    }
    internal_selected = {
        "sampled_k": 2,
        metric_key: 0.5,
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
    assert "host-enforced 7.5-second wall-clock deadline" in card
    assert "| Observed sampled success@2 | 0.2500 | 0.5000 |" in card
    assert "pre-specified in the producing source commit" in card
    assert "HumanEvalPack examples, candidate outcomes, and scores did not enter" in card
    assert "resource-limited isolated Docker" in card
    assert "[pinned BigCode Python harness]" in card
    assert "[`results.json`](./results.json)" in card
    assert "/tree/dataset-pin" in card
    for forbidden in ("Sampled pass@", "pre-registered", "hardened"):
        assert forbidden not in card


def _write_publication_tree(root: Path) -> None:
    """Create inert regular files at every reviewed Hub artifact path."""
    for relative_path in PUBLICATION_ALLOWLIST:
        destination = root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"checked fixture")


def test_publication_allowlist_accepts_only_the_exact_reviewed_tree(tmp_path: Path) -> None:
    """Make the release boundary reject extra, missing, and symlinked content."""
    hub = tmp_path / "hub"
    _write_publication_tree(hub)

    assert set(validate_publication_tree(hub)) == set(PUBLICATION_ALLOWLIST)

    unexpected = hub / "adapter" / "training_args.bin"
    unexpected.write_bytes(b"not published")
    with pytest.raises(RuntimeError, match=r"training_args\.bin"):
        validate_publication_tree(hub)
    unexpected.unlink()

    missing = hub / "results.json"
    missing.unlink()
    with pytest.raises(RuntimeError, match=r"results\.json"):
        validate_publication_tree(hub)


def test_publication_allowlist_rejects_symlinks(tmp_path: Path) -> None:
    """Prevent an allowlisted path from resolving to bytes outside the staging tree."""
    hub = tmp_path / "hub"
    _write_publication_tree(hub)
    card = hub / "README.md"
    card.unlink()
    card.symlink_to(tmp_path / "outside-card.md")

    with pytest.raises(RuntimeError, match=r"README\.md"):
        validate_publication_tree(hub)
