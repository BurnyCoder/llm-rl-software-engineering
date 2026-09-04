"""Global context: prove prompts and generations are persisted without truncation.

Source: https://docs.python.org/3.12/library/logging.html
"""

import json
from io import StringIO
from pathlib import Path

import pytest

from minibug_rl.run_logging import RunLogger


class BrokenTerminal(StringIO):
    """Model a disconnected terminal whose first write raises immediately."""

    def write(self, text: str) -> int:
        """Reject terminal output while leaving file-backed audit streams usable."""
        raise BrokenPipeError("terminal disconnected")


def test_generation_logging_preserves_full_prompt_and_completion(
    tmp_path: Path,
    capsys: object,
) -> None:
    """Write exact text to JSONL and the real-time terminal stream."""
    # Long sentinel tails catch accidental display-oriented truncation.
    prompt = "PROMPT-BEGIN\n" + ("p" * 20_000) + "\nPROMPT-END"
    completion = "OUTPUT-BEGIN\n" + ("o" * 20_000) + "\nOUTPUT-END"
    logger = RunLogger.create(tmp_path, run_id="20260101T000000Z-test")

    generation_id = logger.generation(
        task_id="task-1",
        split="train",
        prompt=prompt,
        completion=completion,
        metadata={"reward": 1.1},
    )
    logger.generation_outcome(
        generation_id=generation_id,
        task_id="task-1",
        split="train",
        metadata={"status": "success", "reward": 1.1},
    )
    logger.close()

    # JSONL is the authoritative machine-readable audit trail.
    records = [
        json.loads(line)
        for line in (logger.directory / "generations.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    generation, outcome = records
    assert generation["event"] == "generation"
    assert generation["generation_id"] == generation_id
    assert generation["prompt"] == prompt
    assert generation["completion"] == completion
    assert outcome["event"] == "generation_outcome"
    assert outcome["generation_id"] == generation_id
    assert outcome["metadata"] == {"status": "success", "reward": 1.1}
    # Terminal logging is intentionally complete per the repository requirement.
    terminal = capsys.readouterr().out  # type: ignore[attr-defined]
    assert prompt in terminal
    assert completion in terminal


def test_generation_is_durable_before_terminal_mirroring_fails(tmp_path: Path) -> None:
    """Keep complete raw evidence even when the interactive output stream disappears."""
    logger = RunLogger.create(tmp_path, run_id="broken-terminal", terminal=BrokenTerminal())

    with pytest.raises(BrokenPipeError, match="terminal disconnected"):
        logger.generation(
            task_id="task-1",
            split="train",
            prompt="complete prompt",
            completion="complete output",
        )
    logger.close()

    record = json.loads((logger.directory / "generations.jsonl").read_text(encoding="utf-8"))
    assert record["prompt"] == "complete prompt"
    assert record["completion"] == "complete output"
    assert json.loads((logger.directory / "run.log").read_text(encoding="utf-8")) == record
