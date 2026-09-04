"""Global context: prevent audited public claims and evidence links from regressing.

Sources:
- https://docs.python.org/3.12/library/json.html#json.load
- https://spec.commonmark.org/0.31.2/#links
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path
from urllib.parse import unquote

# Every checked path is resolved from the repository, independent of the test process CWD.
REPOSITORY = Path(__file__).resolve().parents[1]
# CommonMark inline links are sufficient for the repository's deliberately simple link style.
INLINE_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def _public_markdown_files() -> list[Path]:
    """Return only maintained public Markdown, excluding caches and local artifacts."""
    # Root policy and landing pages are public surfaces even though they live outside docs.
    files = [REPOSITORY / "README.md", REPOSITORY / "AGENTS.md", REPOSITORY / "data/README.md"]
    # Recursive discovery includes experiment narratives without scanning ignored run logs.
    files.extend(sorted((REPOSITORY / "docs").rglob("*.md")))
    files.extend(sorted((REPOSITORY / "reports").rglob("*.md")))
    return files


def _outside_fenced_code(text: str) -> str:
    """Remove fenced code bodies so example syntax cannot become a documentation link."""
    # A line-oriented state machine handles every backtick fence used in the checked prose.
    retained: list[str] = []
    inside_fence = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            inside_fence = not inside_fence
            continue
        if not inside_fence:
            retained.append(line)
    return "\n".join(retained)


def _local_link_targets(markdown: Path) -> Iterable[Path]:
    """Resolve filesystem targets while ignoring remote URLs and same-page anchors."""
    # Decode percent escapes after extracting links from prose outside code fences.
    prose = _outside_fenced_code(markdown.read_text(encoding="utf-8"))
    for match in INLINE_LINK.finditer(prose):
        target = match.group(1).strip().removeprefix("<").removesuffix(">")
        # Scheme-bearing links and fragment-only links have no repository path to check.
        if not target or target.startswith("#") or "://" in target or target.startswith("mailto:"):
            continue
        # A fragment identifies content inside the file; existence checking concerns the file.
        file_part = unquote(target.split("#", 1)[0])
        if file_part:
            yield (markdown.parent / file_part).resolve()


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Build one JSON object while refusing ambiguous repeated member names."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def test_public_markdown_local_links_resolve() -> None:
    """Keep every maintained relative Markdown destination present in the repository."""
    missing: list[str] = []
    for markdown in _public_markdown_files():
        for target in _local_link_targets(markdown):
            if not target.exists():
                missing.append(f"{markdown.relative_to(REPOSITORY)} -> {target}")
    assert missing == []


def test_public_evidence_json_rejects_duplicate_object_keys() -> None:
    """Prove every checked evidence document has one unambiguous value per object key."""
    evidence_files = sorted((REPOSITORY / "reports/evidence").glob("*.json"))
    assert evidence_files
    for evidence_file in evidence_files:
        json.loads(
            evidence_file.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )


def test_audited_public_claim_phrases_do_not_return() -> None:
    """Block exact overclaims removed by the repository-wide evidence audit."""
    public_text = "\n".join(
        markdown.read_text(encoding="utf-8").casefold() for markdown in _public_markdown_files()
    )
    forbidden = (
        "all rollouts ran in",
        "every rollout ran in",
        "hardened docker",
        "no mounts",
        "pre-registered validation",
        "sampled pass@",
        "unopened final",
    )
    assert [phrase for phrase in forbidden if phrase in public_text] == []


def test_environment_template_is_account_neutral_and_secret_free() -> None:
    """Keep the committed environment example inert for a local-only reproduction."""
    lines = (REPOSITORY / ".env.example").read_text(encoding="utf-8").splitlines()
    active_assignments = [line for line in lines if line and not line.lstrip().startswith("#")]
    assert active_assignments == ["HF_TOKEN="]
    assert "BurnyCoder" not in "\n".join(lines)


def test_run_summary_v2_preserves_audited_metric_and_routing_facts() -> None:
    """Tie headline terminology and execution boundaries to checked machine evidence."""
    summary = json.loads(
        (REPOSITORY / "reports/evidence/run-summary.json").read_text(encoding="utf-8")
    )
    assert summary["schema_version"] == 2
    assert summary["amendment"]["measured_values_changed"] is False
    assert summary["amendment"]["training_or_evaluation_rerun"] is False
    routing = summary["training"]["reward_routing"]
    routed = (
        routing["host_parser_or_policy_rejections"]
        + routing["deterministic_reward_cache_hits"]
        + routing["valid_cache_misses_executed_in_fresh_containers"]
    )
    assert routed == routing["smoke_and_main_rollouts"] == 404
    for section in (summary["validation"], summary["final_internal_test"]):
        for system in ("base", "selected") if "selected" in section else ("base", "smoke", "train"):
            assert "observed_sampled_success_at_4" in section[system]
            assert "sampled_pass_at_4" not in section[system]
