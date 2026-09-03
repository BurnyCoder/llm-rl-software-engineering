"""Global context: timestamped terminal and append-only run evidence for every phase.

Sources:
- https://docs.python.org/3/library/logging.html
- https://jsonlines.org/
"""

from __future__ import annotations

import csv
import json
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO


def utc_timestamp() -> str:
    """Return a filesystem-safe, timezone-explicit UTC timestamp."""
    # Second precision is readable while the suffix documents the UTC time basis.
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _json_default(value: Any) -> str:
    """Serialize path-like and library-specific values without losing their text."""
    # `str` is deterministic for paths and common scalar metadata objects.
    return str(value)


class RunLogger:
    """Mirror lifecycle messages to terminal and files while preserving raw generations."""

    def __init__(self, directory: Path, terminal: TextIO | None = None) -> None:
        """Open line-buffered append-only files inside one immutable run directory."""
        # The caller constructs the directory name so resumed phases share the same run.
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        # Default to current stdout at construction, which also works with pytest capture.
        self._terminal = terminal if terminal is not None else sys.stdout
        # Line buffering makes live tailing useful without waiting for process exit.
        self._run_file = (directory / "run.log").open("a", encoding="utf-8", buffering=1)
        self._generation_file = (directory / "generations.jsonl").open(
            "a", encoding="utf-8", buffering=1
        )
        self._metric_file = (directory / "metrics.jsonl").open(
            "a", encoding="utf-8", buffering=1
        )
        # A lock prevents interleaved JSON records if reward workers log concurrently.
        self._lock = threading.Lock()

    @classmethod
    def create(
        cls,
        root: str | Path,
        *,
        run_id: str | None = None,
        terminal: TextIO | None = None,
    ) -> RunLogger:
        """Create a timestamped run directory or reopen an explicitly named run."""
        # A short fixed suffix makes automatically created IDs recognizable in reports.
        selected_id = run_id or f"{utc_timestamp()}-minibug-rl"
        return cls(Path(root).expanduser().resolve() / selected_id, terminal=terminal)

    @staticmethod
    def _record(event: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Add shared event metadata without mutating caller-owned dictionaries."""
        # ISO timestamps remain easy for humans and standard data tools to parse.
        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "event": event,
            **payload,
        }

    def message(self, event: str, message: str, **metadata: Any) -> None:
        """Write one timestamped lifecycle event to terminal and `run.log`."""
        # JSON encoding keeps arbitrary diagnostics on one unambiguous log line.
        record = self._record(event, {"message": message, **metadata})
        line = json.dumps(record, ensure_ascii=False, default=_json_default)
        with self._lock:
            print(line, file=self._terminal, flush=True)
            print(line, file=self._run_file, flush=True)

    def generation(
        self,
        *,
        task_id: str,
        split: str,
        prompt: Any,
        completion: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Persist and print an entire model prompt/completion pair without truncation."""
        # Do not slice or abbreviate either field: this file is the reproducibility record.
        record = self._record(
            "generation",
            {
                "task_id": task_id,
                "split": split,
                "prompt": prompt,
                "completion": completion,
                "metadata": metadata or {},
            },
        )
        line = json.dumps(record, ensure_ascii=False, default=_json_default)
        with self._lock:
            # Print raw fields so terminal viewers see every original newline and character.
            print(
                f"[{record['timestamp']}] generation task={task_id} split={split}\n"
                f"--- PROMPT (complete) ---\n{prompt}\n"
                f"--- COMPLETION (complete) ---\n{completion}\n"
                "--- END GENERATION ---",
                file=self._terminal,
                flush=True,
            )
            print(line, file=self._run_file, flush=True)
            print(line, file=self._generation_file, flush=True)

    def metric(self, step: int, **values: float | int) -> None:
        """Append a numerical metric snapshot to JSONL and a convenient CSV projection."""
        # JSONL tolerates evolving metric names across preflight, training, and evaluation.
        record = self._record("metric", {"step": step, **values})
        line = json.dumps(record, ensure_ascii=False, default=_json_default)
        csv_path = self.directory / "metrics.csv"
        with self._lock:
            print(line, file=self._metric_file, flush=True)
            # CSV is a secondary human-friendly view; JSONL remains authoritative.
            needs_header = not csv_path.exists() or csv_path.stat().st_size == 0
            with csv_path.open("a", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(record))
                if needs_header:
                    writer.writeheader()
                writer.writerow(record)

    def write_json(self, name: str, value: Any) -> Path:
        """Atomically write one resolved configuration or result document."""
        # Restrict names to a single file component so callers cannot escape the run directory.
        if Path(name).name != name:
            raise ValueError("name must be a plain filename")
        destination = self.directory / name
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(value, indent=2, ensure_ascii=False, default=_json_default) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
        return destination

    def close(self) -> None:
        """Flush and close all append-only streams owned by this logger."""
        # Idempotent stream closure simplifies `finally` blocks across CLI phases.
        for handle in (self._run_file, self._generation_file, self._metric_file):
            if not handle.closed:
                handle.flush()
                handle.close()

    def __enter__(self) -> RunLogger:
        """Return the logger for context-manager use."""
        return self

    def __exit__(self, *_error: object) -> None:
        """Always close files when a pipeline phase leaves its context."""
        self.close()
