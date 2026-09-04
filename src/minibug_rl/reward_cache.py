"""Global context: persist deterministic rewards across repeated GRPO rollouts.

Sources:
- https://docs.python.org/3/library/sqlite3.html
- https://docs.python.org/3/library/hashlib.html
- https://docs.python.org/3/library/json.html
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path

from minibug_rl.schemas import CandidateExecutor, RewardBreakdown, TestCase


def _cache_key(
    task_id: str,
    completion: str,
    function_name: str,
    tests: tuple[TestCase, ...],
) -> str:
    """Hash the exact task, hidden suite, callable, and unmodified completion."""
    # Expected values belong in this trusted host key but never in the Docker request.
    document = {
        "schema": 1,
        "task_id": task_id,
        "completion": completion,
        "function_name": function_name,
        "tests": [asdict(test) for test in tests],
    }
    encoded = json.dumps(
        document,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class RewardCache:
    """Store final deterministic reward components in a local SQLite database."""

    path: Path

    def __post_init__(self) -> None:
        """Create the private cache parent and table before the first rollout."""
        normalized = self.path.expanduser().resolve()
        normalized.parent.mkdir(parents=True, exist_ok=True)
        object.__setattr__(self, "path", normalized)
        with sqlite3.connect(normalized) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS rewards ("
                "cache_key TEXT PRIMARY KEY, reward_json TEXT NOT NULL)"
            )

    def score(
        self,
        task_id: str,
        completion: str,
        function_name: str,
        tests: tuple[TestCase, ...],
        executor: CandidateExecutor,
    ) -> tuple[RewardBreakdown, bool]:
        """Return a cached breakdown or execute and remember a new one atomically."""
        # The local import keeps this storage module independent of reward construction.
        from minibug_rl.reward import score_completion

        key = _cache_key(task_id, completion, function_name, tests)
        with sqlite3.connect(self.path) as connection:
            row = connection.execute(
                "SELECT reward_json FROM rewards WHERE cache_key = ?",
                (key,),
            ).fetchone()
        if row is not None:
            payload = json.loads(str(row[0]))
            if not isinstance(payload, dict):
                raise ValueError("Cached reward must be a JSON object")
            return RewardBreakdown(**payload), True
        breakdown = score_completion(completion, function_name, tests, executor)
        serialized = json.dumps(
            asdict(breakdown),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "INSERT OR REPLACE INTO rewards (cache_key, reward_json) VALUES (?, ?)",
                (key, serialized),
            )
        return breakdown, False
